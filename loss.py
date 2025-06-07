import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class LossComputer:
    def __init__(self, is_robust, n_groups, group_counts, alpha=None, gamma=0.1, adj=None, min_var_weight=0, step_size=0.01, use_ema_loss=False, normalize_loss=False, btl=False, group_str='', bias_correction=False, uniform_sampling=False):
        self.is_robust = is_robust
        self.gamma = gamma
        self.alpha = alpha
        self.min_var_weight = min_var_weight
        self.step_size = step_size
        self.use_ema_loss = use_ema_loss
        self.normalize_loss = normalize_loss
        self.btl = btl

        self.n_groups = n_groups # n
        self.group_counts = group_counts.cuda() # [n1, n2, n3]
        self.group_frac = self.group_counts / self.group_counts.sum() # [n1/n, n2/n, n3/n]
        self.group_str = group_str # ['group1', 'group2', 'group3']

        if adj is not None: # size=|n_group|
            self.adj = torch.from_numpy(adj).float().cuda()  # # [0.0,0.0,...] size=|n_group|
        else: # [0,0,...]
            self.adj = torch.zeros(self.n_groups).float().cuda()

        if is_robust:
            assert alpha, 'alpha must be specified'

        self.adv_weights = torch.ones(self.n_groups).cuda()/self.n_groups # [1/n, ..., 1/n]
        if not uniform_sampling:
            self.adv_weights = self.group_counts / self.group_counts.sum()
        print ('\nself.adv_weights = ', self.adv_weights.cpu().numpy())
        print ('self.group_frac = ', self.group_frac.cpu().numpy())
        
        self.exp_avg_loss = torch.zeros(self.n_groups).cuda() # [0, ..., 0] # ema loss for each group
        self.exp_src_loss = torch.zeros(self.n_groups).cuda() # [0, ..., 0]
        self.steps = torch.zeros(self.n_groups).cuda() # [0, ..., 0] updated steps for each group
        self.exp_avg_initialized = torch.zeros(self.n_groups).byte().cuda() # [0, ..., 0]
        self.bias_correction = bias_correction
        self.reset_stats()

    def CE_loss(self, yhat, y):
        batch_size, vocab_size = yhat.shape[0], yhat.shape[-1]
        shift_logits = yhat[..., :-1, :].contiguous()
        shift_labels = y[..., 1:].contiguous()

        per_sample_losses = torch.nn.functional.cross_entropy(
            shift_logits.view(-1, vocab_size), 
            shift_labels.view(-1), 
            ignore_index=-100,  
            reduction='none'
        )

        # print ('per_sample_losses.shape = ', per_sample_losses.shape)
        losses = per_sample_losses.view(batch_size, -1)  
        assert losses.shape == shift_logits.shape[:-1], (losses.shape, shift_logits.shape)

        mask = (shift_labels.view(batch_size, -1) != -100) # [bz, T]
        mask_sum = mask.sum(dim=1) # [bz]
        mask_denom = mask_sum + (mask_sum == 0).float()  # avoid zero tokens
        per_sample_losses = (losses * mask).sum(dim=1) / mask_denom
        assert per_sample_losses.shape == (batch_size, ), (per_sample_losses.shape, batch_size)
        source_loss = (losses * mask).sum() / mask.sum() # [], for debug
        assert source_loss.shape == (), source_loss.shape
        return per_sample_losses, source_loss 
        
    def loss(self, yhat, y, group_idx=None, global_step=-1, is_eval=False):
        '''
        compute per-sample and per-group losses
        yhat: [bz, T, D]
        y: [bz, T]
        group_idx: [bz]
        '''
        # compute per-sample loss
        per_sample_losses, source_loss = self.CE_loss(yhat, y) # [batch_size], []

        if group_idx is None: # ERM
            return source_loss

        # compute per-group losses and sizes
        group_loss, group_count = self.compute_group_avg(per_sample_losses, group_idx) # [n_groups], [n_groups]

        if not is_eval:
            # update historical losses of each group by EMA
            self.update_exp_avg_loss(group_loss, group_count, bias_correction=self.bias_correction)

        # compute reweighted loss
        if self.is_robust: # VAA
            weighted_loss, weights = self.compute_robust_loss(group_loss, group_count) # [], [n_groups]
        else: # ERM
            weighted_loss = source_loss
            weights = None

        if not is_eval: 
            # update stats
            self.update_stats(source_loss, weighted_loss, group_loss, weights, group_count)

            cur_group_loss = group_loss
            cur_weighted_group_loss = group_loss * self.adv_weights
            cur_weights = self.adv_weights

            if global_step <= 0 or global_step % 5 == 0:
                print ()
                self.log_stats(global_step, cur_group_loss, cur_weighted_group_loss, cur_weights)

        return weighted_loss

    def compute_robust_loss(self, group_loss, group_count):
        '''
        VAA: update q and compute weighted loss
        group_loss: [n_groups]
        group_count: [n_groups]
        '''
        adjusted_loss = group_loss
        if self.use_ema_loss: # False by default
            adjusted_loss = self.exp_avg_loss

        if torch.all(self.adj>0): 
            print ('self.adj > 0 !!!')
            print ('self.adj = ', self.adj.cpu().numpy())
            adjusted_loss += self.adj/torch.sqrt(self.group_counts) 

        if self.normalize_loss: # False by default
            adjusted_loss = adjusted_loss/(adjusted_loss.sum())

        self.adv_weights = self.adv_weights * torch.exp(self.step_size * adjusted_loss.data) # exp3 update
        self.adv_weights = self.adv_weights / (self.adv_weights.sum()) # normalize 
        self.adv_weights = self.group_frac * self.min_var_weight + self.adv_weights * (1 - self.min_var_weight)

        # 2. use q to weight loss
        # [group_loss1, group_loss2, group_loss3] @ [q1, q2, q3]
        robust_loss = group_loss @ self.adv_weights

        return robust_loss, self.adv_weights # [], [n_groups]

    def compute_group_avg(self, losses, group_idx):
        '''
        compute observed counts and mean loss for each group
        losses: [batch_size]
        group_idx: [batch_size]
        '''
        # [bz] @ [n_groups, 1] -> [bz] @ [n_groups, bz] -> [n_groups, bz]
        # group_map：[n_groups, bz]
        # Each row corresponds to a group, and each column corresponds to a sample. A value of 1 indicates that the sample belongs to the group.
        group_map = (group_idx == torch.arange(self.n_groups).unsqueeze(1).long().cuda()).float() # [n_groups, batch_size]

        # calculate the sample size of each group.
        group_count = group_map.sum(1) # [n_groups]
        group_denom = group_count + (group_count==0).float() # avoid nans

        # calculate the average loss for each group.
        # [n_groups, batch_size] @ [batch_size] -> [n_groups]
        group_loss = (group_map @ losses.view(-1)) / group_denom
        assert group_loss.shape == (self.n_groups, ), group_loss.shape
        return group_loss, group_count # [n_groups], [n_groups]

    def update_exp_avg_loss(self, group_loss, group_count, bias_correction=False):
        '''
        (1) For groups that are initialized and currently have samples:
        EMA_new = (1-γ) × EMA_old + γ × current_loss

        (2) For groups that are not initialized:
        EMA_new = current_loss

        (3) For groups that are initialized but currently have no samples:
        EMA_new = EMA_old
        '''
        prev_weights = (1 - self.gamma*(group_count > 0).float()) * (self.exp_avg_initialized > 0).float()
        curr_weights = 1 - prev_weights
        
        self.exp_avg_loss = self.exp_avg_loss * prev_weights + group_loss * curr_weights
        self.steps = self.steps + (group_count > 0).float()
    
        if bias_correction:
            bias_correction_mask = ((self.exp_avg_initialized > 0) & (self.steps > 0)).float()
            bias_correction_term = 1 - (1 - self.gamma) ** self.steps
            bias_correction_term = torch.clamp(bias_correction_term, min=1e-8)
            self.exp_avg_loss = self.exp_avg_loss / bias_correction_term * bias_correction_mask + \
                            self.exp_avg_loss * (1 - bias_correction_mask)
        self.exp_avg_initialized = (self.exp_avg_initialized > 0) + (group_count > 0)

    def reset_stats(self):
        self.processed_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_data_counts = torch.zeros(self.n_groups).cuda()
        self.update_batch_counts = torch.zeros(self.n_groups).cuda()
        self.avg_group_loss = torch.zeros(self.n_groups).cuda()
        self.avg_per_sample_loss = 0.
        self.avg_weighted_loss = 0.
        self.avg_source_loss = 0.
        self.batch_count = 0.

    def update_stats(self, source_loss, weighted_loss, group_loss, weights, group_count):
        '''
        Running average is equivalent to a standard average but is suitable for online scenarios.
        The metrics calculated here are not EMA
        '''
        # update avg group loss (running average)
        denom = self.processed_data_counts + group_count # [n_groups]
        denom += (denom==0).float()
        prev_weight = self.processed_data_counts / denom
        curr_weight = group_count / denom
        self.avg_group_loss = prev_weight * self.avg_group_loss + curr_weight * group_loss

        # update avg weighted loss (runing average)
        denom = self.batch_count + 1
        self.avg_weighted_loss = (self.batch_count/denom) * self.avg_weighted_loss + (1/denom) * weighted_loss
        self.avg_source_loss = (self.batch_count/denom) * self.avg_source_loss + (1/denom) * source_loss

        # update group counts
        self.processed_data_counts += group_count

        if self.is_robust: # dro
            self.update_data_counts += group_count * ((weights > 0).float())
            self.update_batch_counts += ((group_count * weights) > 0).float()
        else: # erm
            self.update_data_counts += group_count
            self.update_batch_counts += (group_count>0).float()
        self.batch_count+=1

        # avg per-sample quantities
        group_frac = self.processed_data_counts / (self.processed_data_counts.sum())
        self.avg_per_sample_loss = group_frac @ self.avg_group_loss

    def get_model_stats(self, model, args, stats_dict):
        model_norm_sq = 0.
        for param in model.parameters():
            model_norm_sq += torch.norm(param) ** 2
        stats_dict['model_norm_sq'] = model_norm_sq.item() 
        stats_dict['reg_loss'] = args.weight_decay / 2 * model_norm_sq.item() 
        return stats_dict

    def get_stats(self, model=None, args=None):
        stats_dict = {}
        for idx in range(self.n_groups):
            stats_dict[f'avg_loss_group:{idx}'] = self.avg_group_loss[idx].item()
            stats_dict[f'exp_avg_loss_group:{idx}'] = self.exp_avg_loss[idx].item()
            stats_dict[f'processed_data_count_group:{idx}'] = self.processed_data_counts[idx].item()
            stats_dict[f'update_data_count_group:{idx}'] = self.update_data_counts[idx].item()
            stats_dict[f'update_batch_count_group:{idx}'] = self.update_batch_counts[idx].item()

        stats_dict['avg_weighted_loss'] = self.avg_weighted_loss.item()
        stats_dict['avg_per_sample_loss'] = self.avg_per_sample_loss.item()
        stats_dict['avg_acc'] = self.avg_acc.item()

        # Model stats
        if model is not None:
            assert args is not None
            stats_dict = self.get_model_stats(model, args, stats_dict)

        return stats_dict

    def log_stats(self, global_step, cur_group_loss, cur_weighted_group_loss, cur_weights):
        # 1. global metrics
        print ('-'*20 + ' Training: global_step = ' + str(global_step) + ' ' + '-'*20)
        print(f'Average source loss: {self.avg_source_loss.item():.3f}')
        print(f'Average weighted loss: {self.avg_weighted_loss.item():.3f}  \n')
        
        # 2. group-level metrics
        for group_idx in range(self.n_groups): 
            print (
                f'  {self.group_str[group_idx]}  ' # group name
                f'group = {group_idx}  ' # group idx
                f'[n = {int(self.processed_data_counts[group_idx])}]:\t' # group size
                f'source loss = {cur_group_loss[group_idx]:.3f}  ' # current group loss
                f'ema loss = {self.exp_avg_loss[group_idx]:.3f}  ' # group ema loss
                f'weighted loss = {cur_weighted_group_loss[group_idx]:.3f}  ' # current group weight loss
                f'weight = {self.adv_weights[group_idx]:3f}   ' # group weight
                )
        