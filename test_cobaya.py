from cobaya.model import get_model
from cobaya.yaml import yaml_load

# Load the configuration
info = yaml_load(open("config.yml"))

# Initialize the model
model = get_model(info)

# Define a point in parameter space
params = {
    "H0": 67.5,
    "omega_b": 0.022,
    "omega_cdm": 0.12,
    "tau": 0.054,
    "A_s": 3.05,
    "n_s": 0.965,
    "A_planck": 1.0
}

# Compute the log-likelihood
logl = model.logposterior(params, return_prior=False)
print("Log-likelihood:", logl)