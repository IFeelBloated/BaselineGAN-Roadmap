# Copyright (c) 2021, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Loss functions."""

import torch
from torch_utils import training_stats

#----------------------------------------------------------------------------

class StyleGAN2Loss:
    def __init__(self, device, G, D, augment_pipe=None, r1_gamma=10, style_mixing_prob=0, pl_weight=0, pl_batch_shrink=2, pl_decay=0.01, pl_no_weight_grad=False, blur_init_sigma=0, blur_fade_kimg=0):
        super().__init__()
        self.G                  = G
        self.D                  = D
        self.augment_pipe       = augment_pipe
        self.r1_gamma           = r1_gamma

    def run_G(self, z, c, update_emas=False):
        return self.G(z, c, update_emas=update_emas)

    def run_D(self, img, c, update_emas=False):
        if self.augment_pipe is not None:
            img = self.augment_pipe(img)
        logits = self.D(img, c, update_emas=update_emas)
        return logits

    def accumulate_gradients(self, phase, real_img, real_c, gen_z, gain, cur_nimg):
        if phase == 'G':
            gen_img = self.run_G(gen_z, real_c)
            gen_logits = self.run_D(gen_img, real_c)
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            loss_G = torch.nn.functional.softplus(-gen_logits)
            training_stats.report('Loss/G/loss', loss_G)
            loss_G.mean().mul(gain).backward()

        if phase == 'D':
            gen_img = self.run_G(gen_z, real_c, update_emas=True)
            gen_logits = self.run_D(gen_img, real_c, update_emas=True)
            training_stats.report('Loss/scores/fake', gen_logits)
            training_stats.report('Loss/signs/fake', gen_logits.sign())
            
            real_img_tmp = real_img.detach().requires_grad_(True)
            real_logits = self.run_D(real_img_tmp, real_c)
            training_stats.report('Loss/scores/real', real_logits)
            training_stats.report('Loss/signs/real', real_logits.sign())
            
            r1_grads = torch.autograd.grad(outputs=[real_logits.sum()], inputs=[real_img_tmp], create_graph=True, only_inputs=True)[0]
            r1_penalty = r1_grads.square().sum([1,2,3])
            training_stats.report('Loss/r1_penalty', r1_penalty)
            
            loss_D = torch.nn.functional.softplus(gen_logits) + torch.nn.functional.softplus(-real_logits) + r1_penalty * (self.r1_gamma / 2)
            training_stats.report('Loss/D/loss', loss_D)
            loss_D.mean().mul(gain).backward()
#----------------------------------------------------------------------------
