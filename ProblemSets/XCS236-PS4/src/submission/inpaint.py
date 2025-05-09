import sys
sys.path.append("..") # Adds higher directory to python modules path.
import torch
from torch import Tensor
from typing import Dict
from sampling_config import get_config

def add_forward_tnoise(
    image: Tensor, timestep: int, scheduler_data: Dict[str, Tensor]
) -> Tensor:
    """Add forward timestep noise to the image.

    Args:
        image (Tensor): The input image tensor.
        timestep (int): Current timestep.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.

    Returns:
        x_t (Tensor): The image tensor with added noise.
    """
    config = get_config()
    alpha_bar_at_t = scheduler_data["alphas_bar"][timestep]
    noise = torch.randn(image.shape, device=config.device)
    ### START CODE HERE ###
    epsilon = torch.normal(0, 1, size=image.shape)
    x_t = torch.sqrt(scheduler_data["alphas_bar"][timestep - 1]) * image + torch.sqrt(scheduler_data["alphas_bar"][timestep ]) * epsilon
    return x_t
    ### END CODE HERE ###

def apply_inpainting_mask(
        original_image: Tensor, 
        noisy_image: Tensor,
        mask: Tensor, 
        timestep, 
        scheduler_data) -> Tensor:
    """Apply the inpainting mask to the image.

    Args:
        image (Tensor): The input image tensor.
        noisy_image (Tensor): The noisy image tensor.
        mask (Tensor): The inpainting mask tensor.
        timestep (int): Current timestep.
        scheduler_data (Dict[str, Tensor]): Scheduler parameters.
    Returns:
        Tensor: The inpainted image tensor.
    
    HINT: use add_forward_tnoise to add noise to the original image.
    """
    ### START CODE HERE ###
    x_orig_noisy = add_forward_tnoise(original_image, timestep=timestep, scheduler_data=scheduler_data)
    flipped_mask = torch.logical_not(mask) 
    return x_orig_noisy * mask + noisy_image * flipped_mask
    ### END CODE HERE ###

def get_mask(image: Tensor) -> Tensor:
    """Generate a mask for the given image.

    Args:
        image (Tensor): The input image tensor.

    Returns:
        Tensor: The generated mask tensor.
    """
    # Suppose your image is [1, 3, H, W]
    config = get_config() # useful to get torch device details
    ### START CODE HERE ###
    start_row = int(image.shape[2] / 4)
    start_col = int(image.shape[3] / 4)
    mask_height = start_row * 2
    mask_width = start_col * 2
    overall_mask = torch.zeros(image.shape)
    ones_mask = torch.ones(mask_height, mask_width)
    overall_mask[:,:,start_row:start_row + mask_height, start_col:start_col + mask_width] = ones_mask
    return overall_mask
    ### END CODE HERE ###