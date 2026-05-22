"""Empirical galaxy SED prior from DESI spectra and EAZY template fits."""

from .build_empirical_sed_prior_kde import (
    coeffs_from_sample_row,
    load_prior_training_table,
    load_sed_prior_kde,
    sample_sed_prior,
    sample_sed_prior_kde,
    samples_to_coeffs,
    save_sed_prior_kde,
)
from .prior_sampler import (
    EmpiricalPriorPool,
    build_gpu_prior_pool,
    load_empirical_prior,
    sample_prior_batch,
    unpack_prior_rows,
)
from .simplex import (
    PARAMETERIZATION_LOGITS,
    PARAMETERIZATION_WEIGHTS,
    logits_to_weights,
    logits_to_weights_torch,
    prior_feature_names,
    prior_logit_feature_names,
    prior_weights_feature_names,
    split_feature_matrix,
    weights_to_logits,
    weights_to_logits_torch,
)
from .templates import (
    DEFAULT_PARAM_12D,
    DEFAULT_TEMPLATES_DIR,
    load_eazy_template_bank,
    load_eazy_templates,
)

__all__ = [
    "PARAMETERIZATION_LOGITS",
    "PARAMETERIZATION_WEIGHTS",
    "DEFAULT_PARAM_12D",
    "DEFAULT_TEMPLATES_DIR",
    "EmpiricalPriorPool",
    "build_gpu_prior_pool",
    "coeffs_from_sample_row",
    "load_empirical_prior",
    "load_eazy_template_bank",
    "load_eazy_templates",
    "load_prior_training_table",
    "load_sed_prior_kde",
    "logits_to_weights",
    "logits_to_weights_torch",
    "prior_feature_names",
    "prior_logit_feature_names",
    "prior_weights_feature_names",
    "sample_prior_batch",
    "split_feature_matrix",
    "weights_to_logits",
    "weights_to_logits_torch",
    "sample_sed_prior",
    "sample_sed_prior_kde",
    "samples_to_coeffs",
    "save_sed_prior_kde",
    "unpack_prior_rows",
]
