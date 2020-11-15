from torch import nn
import numpy as np
import torch
from torch.nn import functional as F
eps = 1e-8

class Tacotron2Loss(nn.Module):
    def __init__(self, hp, update_step):
        super(Tacotron2Loss, self).__init__()
        self.expressive_classes = hp.emotion_classes
        self.speaker_classes = hp.speaker_classes
        self.cat_lambda = hp.cat_lambda
        self.speaker_encoder_type = hp.speaker_encoder_type
        self.expressive_encoder_type = hp.expressive_encoder_type
        self.update_step = update_step
        self.kl_lambda = hp.kl_lambda
        self.kl_incr = hp.kl_incr
        self.kl_step = hp.kl_step
        self.kl_step_after = hp.kl_step_after
        self.kl_max_step = hp.kl_max_step
        
        self.cat_incr = hp.cat_incr
        self.cat_step = hp.cat_step
        self.cat_step_after = hp.cat_step_after
        self.cat_max_step = hp.cat_max_step
    
    def indices_to_one_hot(self, data, n_classes):
        targets = np.array(data).reshape(-1)
        return torch.from_numpy(np.eye(n_classes)[targets]).cuda()
    
    def KL_loss(self, mu, var):
        return torch.mean(0.5 * torch.sum(torch.exp(var) + mu**2 - 1. - var, 1))
    
    def gaussian_loss(self, z, z_mu, z_var, z_mu_prior, z_var_prior):
        loss = self.log_normal(z, z_mu, z_var) - self.log_normal(z, z_mu_prior, z_var_prior)

        return loss.mean()
    
    def get_encoder_loss(self, id_, prob_, classes_, cat_lambda, kl_lambda, encoder_type):
        cat_target = self.indices_to_one_hot(id_, classes_)

        if (encoder_type == 'gst' or encoder_type == 'x-vector') and cat_lambda != 0.0:
            loss = cat_lambda*(-self.entropy(cat_target, prob_) - np.log(0.1))
        elif (encoder_type == 'vae' or encoder_type == 'gst_vae') and (cat_lambda != 0.0 or kl_lambda !=0.0):
            loss = cat_lambda*(-self.entropy(cat_target, prob_[2]) - np.log(0.1)) + kl_lambda*self.KL_loss(prob_[0], prob_[1])
        elif encoder_type == 'gmvae' and (cat_lambda != 0.0 or kl_lambda !=0.0) :
            loss = self.gaussian_loss(prob_[0], prob_[1], prob_[2], prob_[2], prob_[3])*kl_lambda + (-self.entropy(cat_target, prob_[4]) - np.log(0.1))*cat_lambda
        else:
            loss = 0.0

        return loss
    
    def update_lambda(self, iteration):
        iteration += 1        
        if self.update_step%iteration == 0:
            self.kl_lambda = self.kl_lambda + self.kl_incr
            self.cat_lambda = self.cat_lambda + self.cat_incr
                          
        if iteration <= self.kl_max_step and iteration%self.kl_step == 0:
            kl_lambda = self.kl_lambda
        elif iteration > self.kl_max_step and iteration%self.kl_step_after == 0:
            kl_lambda = self.kl_lambda
        else:
            kl_lambda = 0.0

                          
        if iteration <= self.cat_max_step and iteration%self.cat_step == 0:
            cat_lambda = self.cat_lambda
        elif iteration > self.cat_max_step and iteration%self.cat_step_after == 0:
            cat_lambda = self.cat_lambda
        else:
            cat_lambda = 0.0
        
        return kl_lambda, cat_lambda
    
    def log_normal(self, x, mu, var):
        if eps > 0.0:
            var = var + eps
        return -0.5 * torch.sum(np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var, dim=-1)
    
    def entropy(self, logits, targets):
        log_q = F.log_softmax(logits, dim=-1)
        return -torch.mean(torch.sum(targets * log_q, dim=-1))
    
    
    def forward(self, iteration, model_output, targets, s_id, e_id):
                          
        kl_lambda, cat_lambda = self.update_lambda(iteration)                  
        
        mel_target, gate_target = targets[0], targets[1]
        mel_target.requires_grad = False
        gate_target.requires_grad = False
        gate_target = gate_target.view(-1, 1)
        
        # tacotron losses
        mel_out, mel_out_postnet, gate_out, _, s_prob, e_prob = model_output
        gate_out = gate_out.view(-1, 1)
        mel_loss = nn.MSELoss()(mel_out, mel_target) + nn.MSELoss()(mel_out_postnet, mel_target)
        gate_loss = nn.BCEWithLogitsLoss()(gate_out, gate_target)

        #speaker_encoder_loss
        speaker_loss = self.get_encoder_loss(s_id, s_prob, self.speaker_classes, cat_lambda, kl_lambda, self.speaker_encoder_type)
                          
        #expressive_encoder_loss
        expressive_loss = self.get_encoder_loss(e_id, e_prob, self.expressive_classes, cat_lambda, kl_lambda, self.expressive_encoder_type)
                          
        return mel_loss + gate_loss + speaker_loss + expressive_loss
    
        
