import torch
import pyro
import pyro.poutine as poutine
from pyro.contrib.util import lexpand
from pyro.infer.autoguide.utils import mean_field_entropy
import math
from torch.utils.data import Dataset
from util import profile_method

def nmc_eig(
    model,
    design,
    observation_labels,
    target_labels=None,
    N=100,
    M=10,
    M_prime=None,
    independent_priors=False,
    contrastive=False
):
    """
    Based on Pyro implementation from:

    https://github.com/pyro-ppl/pyro/blob/dev/pyro/contrib/oed/eig.py

    adding an option for contrastive sampling to convert to a lower bound

    """

    if isinstance(observation_labels, str):  # list of strings instead of strings
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    # Take N samples of the model
    expanded_design = lexpand(design, N)  # N copies of the model
    trace = poutine.trace(model).get_trace(expanded_design) # sample y_n and theta_n N times
    trace.compute_log_prob() # compute likelihood p(y_n | theta_n,0, d)

    if M_prime is not None:
        y_dict = {
            l: lexpand(trace.nodes[l]["value"], M_prime) for l in observation_labels
        }
        theta_dict = {
            l: lexpand(trace.nodes[l]["value"], M_prime) for l in target_labels
        }
        theta_dict.update(y_dict)
        # Resample M values of u and compute conditional probabilities
        # WARNING: currently the use of condition does not actually sample
        # the conditional distribution!
        # We need to use some importance weighting
        conditional_model = pyro.condition(model, data=theta_dict)
        if independent_priors:
            reexpanded_design = lexpand(design, M_prime, 1)
        else:
            # Not acceptable to use (M_prime, 1) here - other variables may occur after
            # theta, so need to be sampled conditional upon it
            reexpanded_design = lexpand(design, M_prime, N)
        retrace = poutine.trace(conditional_model).get_trace(reexpanded_design)
        retrace.compute_log_prob()
        conditional_lp = sum(
            retrace.nodes[l]["log_prob"] for l in observation_labels
        ).logsumexp(0) - math.log(M_prime)
    else:
        # This assumes that y are independent conditional on theta
        # Furthermore assume that there are no other variables besides theta

        # sum together likelihood terms for p(y_n|theta_n,0, d)
        conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels) # shape (N, num_designs)

    # calculate y_n from the model:
    y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}
    # Resample M values of theta and compute conditional probabilities
    conditional_model = pyro.condition(model, data=y_dict)
    # Using (M, 1) instead of (M, N) - acceptable to re-use thetas between ys because
    # theta comes before y in graphical model
    reexpanded_design = lexpand(design, M, 1) 
    retrace = poutine.trace(conditional_model).get_trace(reexpanded_design) # sample theta_n,m M times conditioned on y_n
    retrace.compute_log_prob() # compute likelihood p(y_n|theta_n,m, d)
    if not contrastive:
        # sum together likelihood terms for p(y_n|theta_n,m, d) with extra term from 1/M
        # returns log summed exponentials log(exp(x_1)+exp(x_2)..) of each row of the input tensor in the given dim (0)
        marginal_lp = sum(retrace.nodes[l]["log_prob"] for l in observation_labels).logsumexp(0) - math.log(M)

        #terms = conditional_lp - marginal_lp
        #nonnan = (~torch.isnan(terms)).sum(0).type_as(terms)
        #terms[torch.isnan(terms)] = 0.0
        #return terms.sum(0) / nonnan
        return _safe_mean_terms(conditional_lp - marginal_lp)[1]
    else:
        marginal_log_probs = torch.cat([lexpand(conditional_lp, 1),
                                        sum(retrace.nodes[l]["log_prob"] for l in observation_labels)]) # shape (M+1, N, num_designs)
        marginal_lp_lower = marginal_log_probs.logsumexp(0) - math.log(M+1)
        marginal_lp_upper = marginal_log_probs[1:].logsumexp(0) - math.log(M)
        return _safe_mean_terms(conditional_lp - marginal_lp_lower)[1], _safe_mean_terms(conditional_lp - marginal_lp_upper)[1]


def vnmc_eig(
    model,
    design,
    observation_labels,
    target_labels,
    num_samples,
    num_steps,
    guide,
    optim,
    return_history=False,
    final_design=None,
    final_num_samples=None,
    contrastive=False,
):
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    loss = _vnmc_eig_loss(model, guide, observation_labels, target_labels, contrastive)
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, contrastive, final_design, final_num_samples,)

def _vnmc_eig_loss(model, guide, observation_labels, target_labels, contrastive=False):
    """VNMC loss: to evaluate directly use `vnmc_eig` setting `num_steps=0`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):
        N, M = num_particles
        expanded_design = lexpand(design, N) # N copies of the model

        # Sample from p(y, theta | d)
        trace = poutine.trace(model).get_trace(expanded_design) 
        # sample y_n and theta_n N times
            # POSSIBLE BUG: only one theta is sampled ? FIXED
        y_dict = {l: lexpand(trace.nodes[l]["value"], M) for l in observation_labels}

        # Sample M times from q(theta | y, d) for each y
        reexpanded_design = lexpand(expanded_design, M)
        conditional_guide = pyro.condition(guide, data=y_dict) # condition the guide on y_n
        # sample theta_n,m M times for each y_n
        guide_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, reexpanded_design, observation_labels, target_labels)
        theta_y_dict = {l: guide_trace.nodes[l]["value"] for l in target_labels}
        theta_y_dict.update(y_dict) # theta_y_dict contains theta_n,m and y_n
        # theta_n,m has shape [M,N,n_designs]
        guide_trace.compute_log_prob() # compute q(theta_n,m | y_n, d, phi), site-wise log probabilities of each trace. (shape = batch_shape)

        if contrastive:
            # concatenate original sample theta_0 to be included in the inner sum
            theta_y_dict = {l: torch.cat([lexpand(trace.nodes[l]["value"], 1), theta_y_dict[l]], dim=0) for l in target_labels}
            y_dict = {l: torch.cat([lexpand(trace.nodes[l]["value"], 1), y_dict[l]], dim=0) for l in observation_labels}
            M += 1
            reexpanded_design = lexpand(expanded_design, M)
            conditional_guide = pyro.condition(guide, data=y_dict) # condition the guide on y_n
            guide_trace = poutine.trace(conditional_guide).get_trace(
                y_dict, reexpanded_design, observation_labels, target_labels)
            theta_y_dict = {l: guide_trace.nodes[l]["value"] for l in target_labels}
            theta_y_dict.update(y_dict)
            guide_trace.compute_log_prob()

        # Re-run that through the model to compute the joint for a given theta_n and y_n
        modelp = pyro.condition(model, data=theta_y_dict)
        # sample y_n and theta_n,m M times
        model_trace = poutine.trace(modelp).get_trace(reexpanded_design)
        model_trace.compute_log_prob() # compute joint likelihood p(y_n, theta_n,m, | d)

        terms = -sum(guide_trace.nodes[l]["log_prob"] for l in target_labels) # q(theta_nm | y_n)
        terms += sum(model_trace.nodes[l]["log_prob"] for l in target_labels) # p(theta_nm)
        terms += sum(model_trace.nodes[l]["log_prob"] for l in observation_labels) # p(y_n | theta_nm, d)

        if evaluation:
            trace.compute_log_prob() # compute likelihood p(y_n | theta_n,0, d)
            conditional_lp = sum(trace.nodes[l]["log_prob"] for l in observation_labels) # p(y_n | theta_n, d)

        # to calculate lower and upper bounds:
        if contrastive:
            # including the original sample of theta_0 from which y was sampled to get the lower bound:
            lower_terms = -terms.logsumexp(0) + math.log(M) 
            # excluding the original sample to get the upper bound:
            upper_terms = -terms[1:].logsumexp(0) + math.log(M-1)
            if evaluation:
                lower_terms += conditional_lp
                upper_terms += conditional_lp
            lower_agg_loss, lower_loss = _safe_mean_terms(lower_terms)
            upper_agg_loss, upper_loss = _safe_mean_terms(upper_terms)
            agg_loss = (lower_agg_loss, upper_agg_loss)
            loss = (lower_loss, upper_loss)
        else:
            # returns log summed exponentials log(exp(x_1)+exp(x_2)..) of each row of the input tensor in the given dim (0)
            terms = -terms.logsumexp(0) + math.log(M) 
            if evaluation:
                terms += conditional_lp
            agg_loss, loss = _safe_mean_terms(terms)

        return agg_loss, loss
    
    return loss_fn

def nf_loss(
    samples,
    context,
    guide,
    experiment,
    rank=0,
    verbose_shapes=False,
    log_probs=None,
    evaluation=False,
    chunk_size=None,
):
    """
    Computes the negative log-probability loss for a normalizing flow model given pre-sampled data.

    Parameters:
    - samples: Pre-sampled parameter values (theta).
    - context: The design tensor (batch of designs).
    - guide: The normalizing flow guide model (e.g., a Pyro or zuko flow).
    - experiment: Object with transform_input, params_to_unconstrained, cosmo_params, etc.
    - rank: The rank of the current process.
    - verbose_shapes: Whether to print tensor shapes for debugging (default: False).
    - log_probs: Dict of prior log-probs for evaluation mode.
    - evaluation: If True, return EIG-style loss (prior_entropy - posterior_entropy).
    - chunk_size: If not None, evaluate log_prob in chunks of this size along the
                  flattened batch dimension to reduce peak memory usage.

    Returns:
    - agg_loss: Aggregated loss over all samples.
    - loss: Per-sample loss, reshaped to match the original design batch.
    """
    # Store original batch shape
    batch_shape = samples.shape[:-1]  # e.g. [num_particles, num_designs]
    
    # Flatten samples and contexts for normalizing flow input
    flattened_samples = samples.view(-1, samples.shape[-1])
    flattened_context = context.view(-1, context.shape[-1])

    if verbose_shapes and rank == 0:
        print(f"Context shape: {context.shape}")
        print(f"Samples shape: {samples.shape}")
        print(f"Flattened samples shape: {flattened_samples.shape}")
        print(f"Flattened context shape: {flattened_context.shape}")
        if chunk_size is not None:
            print(f"Using chunk_size = {chunk_size}")

    # Transform to unconstrained space if needed
    if experiment.transform_input:
        y_flat = experiment.params_to_unconstrained(flattened_samples)
    else:
        y_flat = flattened_samples

    # Compute the negative log-probability
    if chunk_size is None:
        neg_log_prob = -guide(flattened_context).log_prob(y_flat)
    else:
        # Chunk along the flattened batch dimension
        neg_log_prob_chunks = []
        for y_chunk, ctx_chunk in zip(
            y_flat.split(chunk_size, dim=0),
            flattened_context.split(chunk_size, dim=0),
        ):
            neg_log_prob_chunks.append(-guide(ctx_chunk).log_prob(y_chunk))
        neg_log_prob = torch.cat(neg_log_prob_chunks, dim=0)

    # Reshape back to original batch shape
    neg_log_prob = neg_log_prob.reshape(batch_shape)

    # Compute the aggregate loss
    agg_loss, loss = _safe_mean_terms(neg_log_prob)

    if evaluation:
        if log_probs is None:
            raise ValueError("Log probabilities are not provided")
        
        # Check if experiment uses prior_flow - if so, use joint log_prob directly
        if hasattr(experiment, 'prior_flow') and experiment.prior_flow is not None:
            prior_entropy = -1 * log_probs["joint"].mean(dim=0)
        else:
            # Per-parameter log_probs from uniform priors: sum them
            prior_entropy = -1 * sum(log_probs[l].mean(dim=0) for l in experiment.cosmo_params)
        
        loss = prior_entropy - loss

    return agg_loss, loss

def posterior_loss(experiment, guide, num_particles, 
                   nflow=False, verbose_shapes=False, evaluation=False, 
                   analytic_prior=True, condition_design=True, 
                   context=True, nominal_design=False
                   ):
    # num_particles = num_samples
    if not nominal_design:
        expanded_design = lexpand(experiment.designs, num_particles)
    else:
        if getattr(experiment, "nominal_design", None) is None:
            raise ValueError("Nominal design not found in experiment")
        expanded_design = lexpand(experiment.nominal_design.reshape(1, -1), num_particles)
    if nflow:
        batch_shape = expanded_design.shape[:-1]
        # Sample y and theta from p(y, theta | d)
        trace = poutine.trace(experiment.pyro_model).get_trace(expanded_design)
        y_dict = {l: trace.nodes[l]["value"] for l in experiment.observation_labels}
        theta_dict = {l: trace.nodes[l]["value"] for l in experiment.cosmo_params}
        # combine y and design into the  input
        condition_input = _create_condition_input(design=expanded_design,
                                                y_dict=y_dict,
                                                observation_labels=experiment.observation_labels,
                                                condition_design=condition_design)
        evaluate_samples = torch.cat([theta_dict[k].unsqueeze(dim=-1) for k in experiment.cosmo_params], dim=-1)#.squeeze(dim=-1)
        # Flatten batch dimensions into one batch dimension - required for nflows :(
        flattened_samples = torch.flatten(evaluate_samples, start_dim=0, end_dim=len(batch_shape)-1)
        flattened_condition = torch.flatten(condition_input, start_dim=0, end_dim=len(batch_shape)-1)
        # Flatten posterior dimensions - if any
        #flattened_samples = torch.flatten(flattened_samples, start_dim=0, end_dim=-1)
        if verbose_shapes:
            print("design shape", experiment.designs.shape)
            print("batch_shape", batch_shape)
            print("condition_input shape", condition_input.shape)
            print("evaluate_samples shape", evaluate_samples.shape)
            print("flattened_samples shape", flattened_samples.shape)
            print("flattened_condition shape", flattened_condition.shape)
        if context:
            # for using nflows:
            #neg_log_prob = -1*guide.log_prob(inputs=flattened_samples, context=flattened_condition)

            # for using pyro normalizing flows:
            #conditioned_guide = guide.condition(flattened_condition)
            #neg_log_prob = -1*conditioned_guide.log_prob(flattened_samples)
            #neg_log_prob = neg_log_prob.reshape(batch_shape)
            #neg_log_prob = neg_log_prob.sum(dim=-1).reshape(-1, design.shape[0])

            # for using zuko:
            neg_log_prob = -1*guide(flattened_condition).log_prob(flattened_samples)
            neg_log_prob = neg_log_prob.reshape(batch_shape)
        else:
            neg_log_prob = -1*guide.log_prob(inputs=flattened_samples)
            neg_log_prob = neg_log_prob.reshape(-1, experiment.designs.shape[0])
        agg_loss, loss = _safe_mean_terms(neg_log_prob)

    else:
        # Sample from p(y, theta | d) the model
        trace = poutine.trace(experiment.pyro_model).get_trace(expanded_design) # trace: graph data structure denoting relationships amongst different pyro primitives
        y_dict = {l: trace.nodes[l]["value"] for l in experiment.observation_labels} # trace.nodes contains a collection (OrderedDict) of site names and metadata
        theta_dict = {l: trace.nodes[l]["value"] for l in experiment.cosmo_params}

        # Run through q(theta | y, d)
        conditional_guide = pyro.condition(guide, data=theta_dict) # set the sample statements in the guide to the values in theta_dict
        cond_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, expanded_design, experiment.observation_labels, experiment.cosmo_params) 
        cond_trace.compute_log_prob() # compute site-wise log probabilities of each trace. (shape = batch_shape)

        terms = -sum(cond_trace.nodes[l]["log_prob"] for l in experiment.cosmo_params) # forward pass through network and evaluate loss func
        agg_loss, loss = _safe_mean_terms(terms)

    if evaluation:
        if analytic_prior:
            #prior_entropy = model.prior_entropy() # add target labels here
            # First dimension is number of MC samples for sum - reason for zero index
            prior_entropy = sum(trace.nodes[l]["fn"].entropy().mean(dim=0) for l in experiment.cosmo_params) # shape: [num_designs]
            loss = prior_entropy - loss
        else:
            trace.compute_log_prob()
            # First dimension is number of MC samples for sum - reason for zero index
            prior_entropy = -1*sum(trace.nodes[l]["log_prob"].mean(dim=0) for l in experiment.cosmo_params)
            loss = prior_entropy - loss
    return agg_loss, loss

def posterior_eig(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim, contrastive=False,
                  return_history=False, final_design=None, final_num_samples=None, eig=True, prior_entropy_kwargs={},
                  *args, **kwargs):
                  
    if isinstance(observation_labels, str):
        observation_labels = [observation_labels]
    if isinstance(target_labels, str):
        target_labels = [target_labels] # change str to list of string

    if return_history:
        ape, history = _posterior_ape(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim,
                            return_history=return_history, contrastive=contrastive, final_design=final_design, final_num_samples=final_num_samples,
                            *args, **kwargs)
        return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs), history # calculate prior_entropy - ape (if eig=True)
    
    else:
        ape = _posterior_ape(model, design, observation_labels, target_labels, num_samples, num_steps, guide, optim,
                            return_history=return_history, contrastive=contrastive, final_design=final_design, final_num_samples=final_num_samples,
                            *args, **kwargs)
        return _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs)


# APE: average posterior entropy
def _posterior_ape(model, design, observation_labels, target_labels,
                   num_samples, num_steps, guide, optim, return_history=False, contrastive=False,
                   final_design=None, final_num_samples=None, *args, **kwargs):
    loss = _posterior_loss(model, guide, observation_labels, target_labels, *args, **kwargs) # calculate loss
    return opt_eig_ape_loss(design, loss, num_samples, num_steps, optim, return_history, contrastive, final_design, final_num_samples) # apply steps for optimization

def _posterior_loss(model, guide, observation_labels, target_labels, analytic_entropy=False):
    """Posterior loss: to evaluate directly use `posterior_eig` setting `num_steps=0`, `eig=False`."""

    def loss_fn(design, num_particles, evaluation=False, **kwargs):

        # num_particles = num_samples
        expanded_design = lexpand(design, num_particles) # expand num_particles copies to left dimension

        # Sample from p(y, theta | d) the model
        trace = poutine.trace(model).get_trace(expanded_design) # trace: graph data structure denoting relationships amongst different pyro primitives
        y_dict = {l: trace.nodes[l]["value"] for l in observation_labels} # trace.nodes contains a collection (OrderedDict) of site names and metadata
        theta_dict = {l: trace.nodes[l]["value"] for l in target_labels}

        # Run through q(theta | y, d)
        conditional_guide = pyro.condition(guide, data=theta_dict) # set the sample statements in the guide to the values in theta_dict
        cond_trace = poutine.trace(conditional_guide).get_trace(
            y_dict, expanded_design, observation_labels, target_labels) 
        cond_trace.compute_log_prob() # compute site-wise log probabilities of each trace. (shape = batch_shape)
        if evaluation and analytic_entropy:
            loss = mean_field_entropy(
                guide, [y_dict, expanded_design, observation_labels, target_labels],
                whitelist=target_labels).sum(0) / num_particles
            agg_loss = loss.sum()
        else:
            terms = -sum(cond_trace.nodes[l]["log_prob"] for l in target_labels) # forward pass through network and evaluate loss func
            agg_loss, loss = _safe_mean_terms(terms)

        return agg_loss, loss

    return loss_fn


def opt_eig_ape_loss(design, loss_fn, num_samples, num_steps, optim, return_history=False, contrastive=False,
                     final_design=None, final_num_samples=None):

    if final_design is None:
        final_design = design
    if final_num_samples is None:
        final_num_samples = num_samples

    if contrastive:
        params = None
        history_upper = []
        history_lower = []
        for step in range(num_steps):
            if params is not None:
                pyro.infer.util.zero_grads(params)
            with poutine.trace(param_only=True) as param_capture:
                agg_loss, loss = loss_fn(design, num_samples, evaluation=return_history)
            params = set(site["value"].unconstrained()
                        for site in param_capture.trace.nodes.values())
            if torch.isnan(agg_loss[0]) or torch.isnan(agg_loss[1]):
                raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
            agg_loss[0].backward(retain_graph=True)
            agg_loss[1].backward(retain_graph=True)
            if return_history:
                history_lower.append(loss[0])
                history_upper.append(loss[1])
            optim(params)
            try:
                optim.step()
            except AttributeError:
                pass

        _, loss = loss_fn(final_design, final_num_samples, evaluation=True, contrastive=True)
        if return_history:
            history = torch.stack(history_lower), torch.stack(history_upper)
            return loss, history
        else:
            return loss


    else:
        params = None
        history = []
        loss_min = None
        for step in range(num_steps):
            # zero the gradients of the parameters:
            if params is not None:
                pyro.infer.util.zero_grads(params)
            # identify the parameters that are being optimized with the loss function:
            with poutine.trace(param_only=True) as param_capture:
                agg_loss, loss = loss_fn(design, num_samples, evaluation=return_history)
            # define a set including the parameter values:
            params = set(
                site["value"].unconstrained() for site in param_capture.trace.nodes.values())
            if torch.isnan(torch.tensor([torch.isnan(site["value"]).any() 
                                        for site in param_capture.trace.nodes.values()])).any():
                print("Encountered NaN in params before optim")
            if torch.isnan(agg_loss):
                raise ArithmeticError("Encountered NaN loss in opt_eig_ape_loss")
            agg_loss.backward(retain_graph=True)
            if return_history:
                history.append(loss)
            optim(params)
            #for i in list(params):
                #if torch.isnan(i).any():
                    #print("NaN in params", "step:", step, 
                    #        "design:", design.squeeze(), 
                    #        "learning rate:", optim.pt_optim_args["lr"])
                    #print("params:", i)
            try:
                optim.step()
            except AttributeError:
                pass
    
        _, loss = loss_fn(final_design, final_num_samples, evaluation=True, contrastive=False)
        if return_history:
            return loss, torch.stack(history)
        else:
            return loss

def _eig_from_ape(model, design, target_labels, ape, eig, prior_entropy_kwargs):
    mean_field = prior_entropy_kwargs.get("mean_field", True)
    if eig:
        if mean_field:
            try:
                prior_entropy = mean_field_entropy(
                    model, [design], whitelist=target_labels
                )
            except NotImplemented:
                prior_entropy = monte_carlo_entropy(
                    model, design, target_labels, **prior_entropy_kwargs
                )
        else:
            prior_entropy = monte_carlo_entropy(
                model, design, target_labels, **prior_entropy_kwargs
            )
        return prior_entropy - ape
    else:
        return ape


def monte_carlo_entropy(model, design, target_labels, num_prior_samples=1000):
    # compute a MC estimate of the entropy of the prior distribution
    if isinstance(target_labels, str):
        target_labels = [target_labels]

    expanded_design = lexpand(design, num_prior_samples)
    trace = pyro.poutine.trace(model).get_trace(expanded_design)
    trace.compute_log_prob()
    lp = sum(trace.nodes[l]["log_prob"] for l in target_labels)
    return -lp.sum(0) / num_prior_samples

def _safe_mean_terms(terms):
    mask = torch.isnan(terms) | (terms == float("-inf")) | (terms == float("inf"))
    if terms.dtype is torch.float32:
        nonnan = (~mask).sum(0).float()
    elif terms.dtype is torch.float64:
        nonnan = (~mask).sum(0).double()
    terms[mask] = 0.0
    loss = terms.sum(0) / nonnan
    agg_loss = loss.sum()
    return agg_loss, loss

def _create_condition_input(design, y_dict, observation_labels, condition_design=True):
    ys = [design]
    for l in observation_labels:
        if y_dict[l].ndim == design.ndim:
            ys.append(y_dict[l])
        else:
            ys.append(y_dict[l].unsqueeze(dim=-1))

    if condition_design:
        return torch.cat(ys, dim=-1)
    else:
        return torch.cat(ys[1:], dim=-1)

class LikelihoodDataset(Dataset):
    def __init__(self, experiment, n_particles_per_device, designs=None, device="cuda", evaluation=False, particle_batch_size=None, profile=False, global_rank=0):
        self.experiment = experiment
        self.n_particles_per_device = n_particles_per_device
        self.device = device
        self.evaluation = evaluation
        self.profile = profile
        self.global_rank = global_rank
        if particle_batch_size is None:
            self.particle_batch_size = n_particles_per_device
        else:
            self.particle_batch_size = min(particle_batch_size, n_particles_per_device)
        if designs is None:
            self.designs = experiment.designs
        else:
            self.designs = designs

    def __len__(self):
        # We only have one "batch" of designs, so return 1
        return 1

    def __getitem__(self, idx):
        # Process particles in batches to reduce memory usage
        if self.particle_batch_size >= self.n_particles_per_device:
            # No batching needed - process all particles at once
            return self._process_particles(self.n_particles_per_device)
        else:
            # Process in batches and concatenate results
            batch_results = []
            n_batches = (self.n_particles_per_device + self.particle_batch_size - 1) // self.particle_batch_size
            
            for batch_idx in range(n_batches):
                start_idx = batch_idx * self.particle_batch_size
                end_idx = min(start_idx + self.particle_batch_size, self.n_particles_per_device)
                n_particles_in_batch = end_idx - start_idx
                
                # Process this batch
                batch_result = self._process_particles(n_particles_in_batch)
                batch_results.append(batch_result)
                
                # Clear GPU cache between batches to free memory
                if torch.cuda.is_available() and batch_idx < n_batches - 1:
                    torch.cuda.empty_cache()
            
            # Concatenate results from all batches
            samples_list, condition_input_list, log_probs_list = zip(*batch_results)
            samples = torch.cat(samples_list, dim=0)
            condition_input = torch.cat(condition_input_list, dim=0)
            
            if self.evaluation:
                log_probs = {}
                for key in log_probs_list[0].keys():
                    log_probs[key] = torch.cat([lp[key] for lp in log_probs_list], dim=0)
                return samples, condition_input, log_probs
            else:
                return samples, condition_input
    
    @profile_method
    def _sample_from_model(self, expanded_design):
        """Sample from the pyro model and return trace, y_dict, theta_dict."""
        with torch.no_grad():
            trace = poutine.trace(self.experiment.pyro_model).get_trace(expanded_design)
            y_dict = {l: trace.nodes[l]["value"] for l in self.experiment.observation_labels}
            theta_dict = {l: trace.nodes[l]["value"] for l in self.experiment.cosmo_params}
        return trace, y_dict, theta_dict

    @profile_method
    def _compute_prior_log_probs(self, samples, trace):
        """Compute log probabilities from prior (either prior_flow or uniform)."""
        if hasattr(self.experiment, 'prior_flow') and self.experiment.prior_flow is not None:
            # Compute log probabilities from prior_flow
            nominal_context = self.experiment.prior_flow_metadata['nominal_context'].to(self.device)
            batch_shape = samples.shape[:-1]
            flattened_samples = samples.view(-1, samples.shape[-1])
            expanded_context = nominal_context.unsqueeze(0).expand(flattened_samples.shape[0], -1)

            prior_transform_input = self.experiment.prior_flow_metadata['transform_input']
            if prior_transform_input:
                samples_for_log_prob = self.experiment.params_to_unconstrained(flattened_samples)
            else:
                samples_for_log_prob = flattened_samples

            prior_dist = self.experiment.prior_flow(expanded_context)
            log_prob_all = prior_dist.log_prob(samples_for_log_prob)
            log_prob_all = log_prob_all.reshape(batch_shape)
            return {"joint": log_prob_all}
        else:
            # Use standard trace.compute_log_prob() for uniform priors
            trace.compute_log_prob()
            return {l: trace.nodes[l]["log_prob"] for l in self.experiment.cosmo_params}

    @profile_method
    def _process_particles(self, n_particles):
        """Process a batch of particles and return results."""
        # Dynamically expand the designs on each access
        expanded_design = lexpand(self.designs, n_particles).to(self.device)

        # Sample from pyro model
        trace, y_dict, theta_dict = self._sample_from_model(expanded_design)

        # Extract the target samples (theta)
        samples = torch.cat([theta_dict[k].unsqueeze(dim=-1) for k in self.experiment.cosmo_params], dim=-1)

        # Create the condition input
        condition_input = _create_condition_input(
            design=expanded_design,
            y_dict=y_dict,
            observation_labels=self.experiment.observation_labels,
            condition_design=True
        )

        if self.evaluation:
            log_probs = self._compute_prior_log_probs(samples, trace)
            del trace, expanded_design, y_dict, theta_dict
            return samples, condition_input, log_probs
        else:
            del trace, expanded_design, y_dict, theta_dict
            return samples, condition_input

