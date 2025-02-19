import torch
from torch import nn
import os
script_directory = os.path.dirname(os.path.abspath(__file__))

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

class VAE(nn.Module):
    def __init__(self, nn='v1', name='vae', z_dim=2):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim) # from ./nns/v1.py

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

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
        # Note that nelbo = kl + rec
        #
        # Outputs should all be tensor scalars
        #
        # Return:
        #   nelbo, kl, rec
        ################################################################################
        ### START CODE HERE ###

        batch = x.shape[0]

        #compute reconstruction loss
        q_phi = self.enc(x)
        print("q_phi mean shape: [" + str(q_phi[0].shape) + "]")
        print("q_phi variance shape: [" + str(q_phi[1].shape) + "]")

        z_pred = ut.sample_gaussian(q_phi[0], q_phi[1])
        print("z_pred shape: [" + str(z_pred.shape) + "]")
        
        
        x_pred_logits = self.compute_sigmoid_given(z_pred)
        print("x_pred_logits shape: [" + str(x_pred_logits.shape) + "]")

        log_p_theta = ut.log_bernoulli_with_logits(x, x_pred_logits)
        print("log_p_theta shape: [" + str(log_p_theta.shape) + "]")
        
        rec = torch.mean(log_p_theta) * -1

        #compute kl divergence 
        z_prior_means = self.z_prior[0].expand(batch, self.z_dim)
        z_prior_variances = self.z_prior[1].expand(batch, self.z_dim)
        kl_image_wise = ut.kl_normal(q_phi[0], q_phi[1], z_prior_means, z_prior_variances)
        kl = torch.mean(kl_image_wise) * -1

        #compute nelbo
        nelbo = kl + rec

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
        #
        # HINT: The summation over m may seem to prevent us from 
        # splitting the ELBO into the KL and reconstruction terms, but instead consider 
        # calculating log_normal w.r.t prior and q
        ################################################################################
        ### START CODE HERE ###
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################
        raise NotImplementedError

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
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
