from typing import Dict
import torch
from .score_matching_utils import (
    create_log_p_theta,
    compute_score_function,
    compute_trace_jacobian,
    compute_frobenius_norm_squared,
    add_noise,
    compute_score
)

# Objective Function for Denoising Score Matching
def denoising_score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor], noise_std: float = 0.1
) -> torch.Tensor:
    """Objective function for denoising score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.
        noise_std (float): Standard deviation of the noise to add.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###


# Objective Function for Score Matching
def score_matching_objective(
    x: torch.Tensor, theta: Dict[str, torch.Tensor]
) -> torch.Tensor:
    """Objective function for score matching.

    Args:
        x (torch.Tensor): Input tensor.
        theta (Dict[str, torch.Tensor]): Parameters containing 'mean' and 'log_var'.

    Returns:
        torch.Tensor: The computed objective value.
    """
    mean = theta["mean"]
    log_var = theta["log_var"]
    ### START CODE HERE ###
    pass
    ### END CODE HERE ###