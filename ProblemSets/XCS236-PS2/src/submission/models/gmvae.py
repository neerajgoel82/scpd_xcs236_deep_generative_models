import numpy as np
import random
import torch
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn

class GMVAE(nn.Module):
    def __init__(self, nn='v1', z_dim=2, k=500, name='gmvae'):
        super().__init__()
        self.name = name
        self.k = k
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Mixture of Gaussians prior
        self.z_pre = torch.nn.Parameter(torch.randn(1, 2 * self.k, self.z_dim)
                                        / np.sqrt(self.k * self.z_dim))
        # Uniform weighting
        self.pi = torch.nn.Parameter(torch.ones(k) / k, requires_grad=False)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL and Rec decomposition
        #
        # To help you start, we have computed the mixture of Gaussians prior
        # prior = (m_mixture, v_mixture) for you, where
        # m_mixture and v_mixture each have shape (1, self.k, self.z_dim)
        #
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl_z, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        ### START CODE HERE ###
        #compute reconstruction loss
        q_phi = self.enc(x)
        z_pred = ut.sample_gaussian(q_phi[0], q_phi[1])
        x_pred_logits = self.dec(z_pred)
        log_p_theta_image_wise = ut.log_bernoulli_with_logits(x, x_pred_logits)
        rec = torch.mean(log_p_theta_image_wise) * -1

        #compute kl divergence 
        kl_image_wise = ut.log_normal(z_pred, q_phi[0], q_phi[1]) - ut.log_normal_mixture(z_pred,prior[0],prior[1])
        kl = torch.mean(kl_image_wise)

        #compute nelbo
        nelbo = torch.mean((log_p_theta_image_wise * -1) + kl_image_wise)

        #print some variables 
        #random_num = random.randint(1, 200)
        #if (random_num % 100 == 0) :
        #    print("\nnelbo: " + str(nelbo.item()) + " kl:" + str(kl.item()) + " rec:" + str(rec.item()))

        #returning all the computed values
        return nelbo,kl,rec

        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################

    def negative_iwae_bound(self, x, iw):
        """
        Computes the Importance Weighted Autoencoder Bound
        Additionally, we also compute the ELBO KL and reconstruction terms

        Args:
            x: tensor: (batch, dim): Observations
            iw: int: (): Number of importance weighted samples

        Returns:
            niwae: tensor: (): Negative IWAE bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute niwae (negative IWAE) with iw importance samples, and the KL
        # and Rec decomposition of the Evidence Lower Bound
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   niwae, kl, rec
        ################################################################################
        # We provide the learnable prior for you. Familiarize yourself with
        # this object by checking its shape.
        prior = ut.gaussian_parameters(self.z_pre, dim=1)

        ### START CODE HERE ###
        batch = x.shape[0]
        duplicated_x = ut.duplicate(x,iw)
        q_phi = self.enc(duplicated_x)

        #sampling iw samples for each q_phi mean and variance 
        z_pred = ut.sample_gaussian(q_phi[0], q_phi[1])
        x_pred_logits = self.dec(z_pred)
        log_p_theta_image_wise = ut.log_bernoulli_with_logits(ut.duplicate(x,iw), x_pred_logits)
        rec = torch.mean(log_p_theta_image_wise, 0) * -1

        #compute kl divergence 
        kl_image_wise = ut.log_normal(z_pred, q_phi[0], q_phi[1]) - ut.log_normal_mixture(z_pred,prior[0],prior[1])
        kl = torch.mean(kl_image_wise, 0)

        #compute nelbo
        combined_nelbo = log_p_theta_image_wise - kl_image_wise 
        combined_nelbo = combined_nelbo.reshape(iw, batch)
        my_log_mean_exp = ut.log_mean_exp(combined_nelbo, 0)
        nelbo = torch.mean(my_log_mean_exp)  * -1 

        #returning all the computed values
        return nelbo,kl,rec

        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        m, v = ut.gaussian_parameters(self.z_pre.squeeze(0), dim=0)
        idx = torch.distributions.categorical.Categorical(self.pi).sample((batch,))
        m, v = m[idx], v[idx]
        return ut.sample_gaussian(m, v)

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
