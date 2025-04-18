import numpy as np
import torch
import torch.utils.data
import os
script_directory = os.path.dirname(os.path.abspath(__file__))
import random

if 'solution' in script_directory:
    from solution import utils as ut
    from solution.models import nns
else:
    from submission import utils as ut
    from submission.models import nns

from torch import nn
from torch.nn import functional as F

class SSVAE(nn.Module):
    def __init__(self, nn='v1', name='ssvae', gen_weight=1, class_weight=100):
        super().__init__()
        self.name = name
        self.z_dim = 64
        self.y_dim = 10
        self.gen_weight = gen_weight
        self.class_weight = class_weight
        nn = getattr(nns, nn)
        self.enc = nn.Encoder(self.z_dim, self.y_dim) # from ./nns/v1.py
        self.dec = nn.Decoder(self.z_dim, self.y_dim) # from ./nns/v1.py
        self.cls = nn.Classifier(self.y_dim)

        # Set prior as fixed parameter attached to Module
        self.z_prior_m = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
        self.z_prior_v = torch.nn.Parameter(torch.ones(1), requires_grad=False)
        self.z_prior = (self.z_prior_m, self.z_prior_v)

        #creating y_prior
        self.y_prior_pi = torch.nn.Parameter(torch.ones(self.y_dim) / self.y_dim, requires_grad=False)
        self.y_prior_log_pi = torch.log(self.y_prior_pi)

    def negative_elbo_bound(self, x):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl_z: tensor: (): ELBO KL divergence to prior for latent variable z
            kl_y: tensor: (): ELBO KL divergence to prior for labels y
            rec: tensor: (): ELBO Reconstruction term
        """
        ################################################################################
        # TODO: Modify/complete the code here
        # Compute negative Evidence Lower Bound and its KL_Z, KL_Y and Rec decomposition
        #
        # To assist you in the vectorization of the summation over y, we have
        # the computation of q(y | x) = `y_prob` and some tensor tiling code for you.
        #
        # Note that nelbo = kl_z + kl_y + rec
        #
        # Outputs should all be tensor scalars

        # Return:
        #   nelbo, kl_z, kl_y, rec

        # HINT 1: The function `kl_normal` and `kl_cat` will be useful from `utils.py`

        # HINT 2: Start by computing KL_Y, KL_Z, and the rec terms. Remember that for
        # KL_Z and rec we have the additional y_dim necessary for the "expectation" w/r
        # to the values of y.

        # HINT 3: When computing the expectation w/r to the distribution q(y | x),
        # KL_Z and rec have values **grouped by y_dim**, so you should reshape
        # to (y_dim,batch) NOT (batch, y_dim). You can take the transpose of y_prob
        # when calculating the expectation then.

        # HINT 4: Try making use of log_bernoulli_with_logits in your calculation of rec
        # from utils.py
        ################################################################################
        y_logits = self.cls(x)
        y_logprob = F.log_softmax(y_logits, dim=1)
        y_prob = torch.softmax(y_logprob, dim=1) # (batch, y_dim)

        # Duplicate y based on x's batch size. Then duplicate x
        # This enumerates all possible combination of x with labels (0, 1, ..., 9)
        y = np.repeat(np.arange(self.y_dim), x.size(0))
        y = x.new(np.eye(self.y_dim)[y])
        x = ut.duplicate(x, self.y_dim)
        
        ### START CODE HERE ###

        #compute kl_y
        kl_y_image_wise = ut.kl_cat(y_prob, y_logprob, self.y_prior_log_pi)
        kl_y = torch.mean(kl_y_image_wise)

        #compute reconstruction loss
        y_prob_repeated = ut.duplicate(y_prob, self.y_dim)
        q_phi = self.enc(x, y_prob_repeated)
        z_pred = ut.sample_gaussian(q_phi[0], q_phi[1])
        x_pred_logits = self.dec(z_pred, y_prob_repeated)
        log_p_theta_image_wise = ut.log_bernoulli_with_logits(x, x_pred_logits)
        neg_log_p_theta_image_wise = log_p_theta_image_wise * -1
        rec = torch.mean(neg_log_p_theta_image_wise)

        #compute kl_z
        batch = x.shape[0]
        z_prior_means = self.z_prior[0].expand(batch, self.z_dim)
        z_prior_variances = self.z_prior[1].expand(batch, self.z_dim)
        kl_z_image_wise = ut.kl_normal(q_phi[0], q_phi[1], z_prior_means, z_prior_variances)
        kl_z = torch.mean(kl_z_image_wise)

        #compute nelbo
        nelbo = kl_y + kl_z + rec

        #printing outputs randomly 
        print_val = random.randint(1, 200)
        if (print_val % 100) == 0: 
            print("\n NELBO: " + str(nelbo.item()) + 
                  " KL_y: " + str(kl_y.item())  + 
                  "KL_z: " +  str(kl_z.item()) + 
                  "Rec: " + str(rec.item()))

        #return the required fields
        return nelbo, kl_z, kl_y, rec 
    
        ### END CODE HERE ###
        ################################################################################
        # End of code modification
        ################################################################################

    def classification_cross_entropy(self, x, y):
        y_logits = self.cls(x)
        return F.cross_entropy(y_logits, y.argmax(1))

    def loss(self, x, xl, yl):
        if self.gen_weight > 0:
            nelbo, kl_z, kl_y, rec = self.negative_elbo_bound(x)
        else:
            nelbo, kl_z, kl_y, rec = [0] * 4
        ce = self.classification_cross_entropy(xl, yl)
        loss = self.gen_weight * nelbo + self.class_weight * ce

        summaries = dict((
            ('train/loss', loss),
            ('class/ce', ce),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl_z),
            ('gen/kl_y', kl_y),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def compute_sigmoid_given(self, z, y):
        logits = self.dec(z, y)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(self.z_prior[0].expand(batch, self.z_dim),
                                  self.z_prior[1].expand(batch, self.z_dim))

    def sample_x_given(self, z, y):
        return torch.bernoulli(self.compute_sigmoid_given(z, y))
