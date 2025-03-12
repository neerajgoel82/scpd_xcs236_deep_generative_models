import torch
from torch.nn import functional as F


def loss_nonsaturating_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)    
    d_loss = None
    # You may find some or all of the below useful:
    #   - F.binary_cross_entropy_with_logits
    ### START CODE HERE ###
    discriminator_x_real_logits = d(x_real)
    d_loss_x_real = F.binary_cross_entropy_with_logits(discriminator_x_real_logits, 
                                                          torch.ones(discriminator_x_real_logits.shape))
    
    x_generated = g(z)
    discriminator_x_generated_logits = d(x_generated)
    d_loss_x_generated = F.binary_cross_entropy_with_logits(discriminator_x_generated_logits, 
                                                          torch.zeros(discriminator_x_generated_logits.shape))

    d_loss = d_loss_x_real + d_loss_x_generated
    return d_loss
    ### END CODE HERE ###

def loss_nonsaturating_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    # You may find some or all of the below useful:
    #   - F.logsigmoid
    ### START CODE HERE ###
    x_generated = g(z)
    discriminator_x_generated_logits = d(x_generated)
    g_loss = torch.mean(F.logsigmoid(discriminator_x_generated_logits)) * -1
    return g_loss
    ### END CODE HERE ###


def conditional_loss_nonsaturating_d(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_generated = torch.randint(low=0, high=2, size=(batch_size,), dtype=torch.int64)

    d_loss = None

    ### START CODE HERE ###
    discriminator_real_logits = d(x_real, y_real)
    d_loss_x_real = F.binary_cross_entropy_with_logits(discriminator_real_logits, 
                                                          torch.ones(discriminator_real_logits.shape))
    
    x_generated = g(z, y_generated)
    discriminator_generated_logits = d(x_generated, y_generated)
    d_loss_x_generated = F.binary_cross_entropy_with_logits(discriminator_generated_logits, 
                                                          torch.zeros(discriminator_generated_logits.shape))

    d_loss = d_loss_x_real + d_loss_x_generated
    return d_loss
    ### END CODE HERE ###


def conditional_loss_nonsaturating_g(g, d, x_real, y_real, *, device):
    """
    Arguments:
    - g (codebase.network.ConditionalGenerator): The generator network
    - d (codebase.network.ConditionalDiscriminator): The discriminator network
      - Note that d outputs logits
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - y_real (torch.Tensor): training data labels (64)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): nonsaturating conditional discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    y_generated = torch.randint(low=0, high=2, size=(batch_size,), dtype=torch.int64)
    g_loss = None

    ### START CODE HERE ###
    x_generated = g(z, y_generated)
    discriminator_x_generated_logits = d(x_generated, y_generated)
    g_loss = torch.mean(F.logsigmoid(discriminator_x_generated_logits)) * -1
    return g_loss
    ### END CODE HERE ###


def loss_wasserstein_gp_d(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - d_loss (torch.Tensor): wasserstein discriminator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    d_loss = None

    # You may find some or all of the below useful:
    #   - torch.rand
    #   - torch.autograd.grad(..., create_graph=True)
    ### START CODE HERE ###

    discriminator_x_real_logits = d(x_real)
    expectation_d_real = torch.mean(discriminator_x_real_logits)
    
    x_generated = g(z)
    discriminator_x_generated_logits = d(x_generated)
    expectation_d_generated = torch.mean(discriminator_x_generated_logits)

    alpha = torch.rand(batch_size).reshape(-1, 1, 1, 1)
    ones = torch.ones(x_real.shape)
    x_r_theta = (alpha * x_generated) + (ones - alpha) * x_real
    discriminator_x_r_theta_logits = d(x_r_theta)
    discriminator_x_r_theta_logits_sum = torch.sum(discriminator_x_r_theta_logits)
    gradients = torch.autograd.grad(discriminator_x_r_theta_logits_sum,  [x_r_theta], create_graph=True)
    norm = torch.norm(gradients[0])
    lambda_param = 10
    third_term = lambda_param * torch.square(norm - 1) 
    d_loss = expectation_d_generated - expectation_d_real + third_term
    return d_loss
    ### END CODE HERE ###


def loss_wasserstein_gp_g(g, d, x_real, *, device):
    """
    Arguments:
    - g (codebase.network.Generator): The generator network
    - d (codebase.network.Discriminator): The discriminator network
      - Note that d outputs value of discriminator
    - x_real (torch.Tensor): training data samples (64, 1, 28, 28)
    - device (torch.device): 'cpu' by default

    Returns:
    - g_loss (torch.Tensor): wasserstein generator loss
    """
    batch_size = x_real.shape[0]
    z = torch.randn(batch_size, g.dim_z, device=device)
    g_loss = None
    
    ### START CODE HERE ###
    x_generated = g(z)
    discriminator_x_generated_logits = d(x_generated)
    g_loss = torch.mean(discriminator_x_generated_logits) * -1
    return g_loss
    ### END CODE HERE ###