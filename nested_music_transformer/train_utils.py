import math

import torch
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import Optimizer
from collections import defaultdict

def add_conti_for_single_feature(tensor):
  new_target = tensor.clone()
  # Assuming tensor shape is [batch, sequence, features]
  # Create a shifted version of the tensor
  shifted_tensor = torch.roll(new_target, shifts=1, dims=1)
  # The first element of each sequence cannot be a duplicate by definition
  shifted_tensor[:, 0] = new_target[:, 0] + 1
  
  # Identify where the original and shifted tensors are the same (duplicates)
  duplicates = new_target == shifted_tensor
  # Replace duplicates with 9999
  new_target[duplicates] = 9999
  return new_target

########################### Loss function ################################

class NLLLoss4REMI():
  def __init__(
      self, 
      focal_alpha:float,
      focal_gamma:float,
  ):
    self.alpha = focal_alpha
    self.gamma = focal_gamma
  
  def get_nll_loss(self, logits, target, mask):
    probs = logits.softmax(dim=-1)
    if probs.ndim == 3:
      probs = probs.flatten(0, 1) # [batch_size*seq_len x vocab_size]
    if target.ndim == 2:
      target = target.flatten(0, 1) # [batch_size*seq_len]
    # clamp min value to 1e-7 to avoid log(0)
    pt = probs[torch.arange(len(target)), target].clamp(1e-7, 1-1e-7) # [batch_size*seq_len]
    loss = -self.alpha * (1-pt)**self.gamma * torch.log(pt) # [batch_size*seq_len]
    loss_seq = loss * mask.flatten(0, 1) # [batch_size*seq_len]
    loss = loss_seq.sum() / mask.sum() # calculating mean loss considering mask
    return loss, loss_seq

  def __call__(self, logits, shifted_tgt, mask, vocab):
    if vocab is not None:
      loss, loss_seq = self.get_nll_loss(logits, shifted_tgt, mask)
      loss_by_class_normal = defaultdict(float)
      shifted_tgt_with_mask = shifted_tgt * mask # [b, t]
      answers_idx = shifted_tgt_with_mask.flatten(0,1) # [b*t]
      for feature in vocab.feature_list:
        feature_mask = vocab.total_mask[feature].to(answers_idx.device) # [327,]
        mask_for_target = feature_mask[answers_idx] # [b*t]
        normal_loss_seq_by_class = loss_seq * mask_for_target
        if mask_for_target.sum().item() != 0:
          loss_by_class_normal[feature+'_normal'] += (normal_loss_seq_by_class.sum().item() / mask_for_target.sum().item())
      return loss, loss_by_class_normal
    else:
      loss, loss_seq = self.get_nll_loss(logits, shifted_tgt, mask)
      return loss, None
    
class NLLLoss4CompoundToken():
  def __init__(self, feature_list, focal_alpha:float, focal_gamma:float):
    self.feature_list = feature_list
    self.alpha = focal_alpha
    self.gamma = focal_gamma

  def get_nll_loss(self, logits, target, mask):
    probs = logits.softmax(dim=-1)
    if probs.ndim == 3:
      probs = probs.flatten(0, 1) # [batch_size*seq_len x vocab_size]
    if target.ndim == 2:
      target = target.flatten(0, 1) # [batch_size*seq_len]
    # clamp min value to 1e-7 to avoid log(0)
    pt = probs[torch.arange(len(target)), target].clamp(1e-7, 1-1e-7) # [batch_size*seq_len]
    loss = -self.alpha * (1-pt)**self.gamma * torch.log(pt) # [batch_size*seq_len]
    loss = loss * mask.flatten(0, 1) # [batch_size*seq_len]
    loss = loss.sum() / mask.sum() # calculating mean loss considering mask
    return loss
  
  def get_nll_loss_for_logging(self, logits, target, mask, ignore_token, conti_token):
    probs = logits.softmax(dim=-1)
    
    if ignore_token is not None and conti_token is not None:
      target_conti = add_conti_for_single_feature(target) # [batch_size*seq_len]
      valid_mask = (target_conti != ignore_token) & (target_conti != conti_token) # [batch_size*seq_len]
    elif ignore_token is not None and conti_token is None:
      valid_mask = (target != ignore_token)
    elif ignore_token is None and conti_token is None:
      valid_mask = torch.ones_like(target).bool()
    valid_mask = valid_mask.flatten(0, 1)
    
    if probs.ndim == 3:
      probs = probs.flatten(0, 1) # [batch_size*seq_len x vocab_size]
    if target.ndim == 2:
      target = target.flatten(0, 1) # [batch_size*seq_len]
    pt = probs[torch.arange(len(target)), target] # [batch_size*seq_len]
    total_mask = mask.flatten(0, 1) & valid_mask # [batch_size*seq_len]
    loss = -self.alpha * (1-pt)**self.gamma * torch.log(pt) # [batch_size*seq_len]
    loss = loss * total_mask # [batch_size*seq_len]
    loss = loss.sum() / total_mask.sum() # calculating mean loss considering mask
    return loss

  def __call__(self, logits_dict, shifted_tgt, mask, valid):
    train_loss_list = []
    log_loss_dict_normal = {}
    for idx, key in enumerate(self.feature_list):
      training_loss = self.get_nll_loss(logits_dict[key], shifted_tgt[..., idx], mask)
      train_loss_list.append(training_loss)
      if valid:
        if key == 'type':
          log_normal_loss = self.get_nll_loss_for_logging(logits_dict[key], shifted_tgt[..., idx], mask, ignore_token=None, conti_token=None)
        elif key == 'beat':
          log_normal_loss = self.get_nll_loss_for_logging(logits_dict[key], shifted_tgt[..., idx], mask, ignore_token=0, conti_token=9999)
        elif key == 'chord' or key == 'tempo' or key == 'instrument':
          log_normal_loss = self.get_nll_loss_for_logging(logits_dict[key], shifted_tgt[..., idx], mask, ignore_token=0, conti_token=9999)
        else:
          log_normal_loss = self.get_nll_loss_for_logging(logits_dict[key], shifted_tgt[..., idx], mask, ignore_token=0, conti_token=None)
        k_normal = key + '_normal'
        log_loss_dict_normal[k_normal] = log_normal_loss
    total_loss = sum(train_loss_list) / len(train_loss_list)
    if valid:
      return  total_loss, log_loss_dict_normal
    else:
      return total_loss, None
    
class EncodecFlattenLoss():
  def __init__(self, feature_list):
    self.feature_list = feature_list
  
  def get_nll_loss(self, logits, target, mask):
    probs = logits.softmax(dim=-1)
    if probs.ndim == 3:
      probs = probs.flatten(0, 1) # [batch_size*seq_len x vocab_size]
    if target.ndim == 2:
      target = target.flatten(0, 1) # [batch_size*seq_len]
    pt = probs[torch.arange(len(target)), target].clamp(1e-7, 1-1e-7) # [batch_size*seq_len]
    loss_seq = -torch.log(pt) # [batch_size*seq_len]
    loss_seq = loss_seq * mask.flatten(0, 1) # [batch_size*seq_len]
    loss = loss_seq.sum() / mask.sum() # calculating mean loss considering mask
    return loss

  def __call__(self, logits, shifted_tgt, mask):
    loss = self.get_nll_loss(logits, shifted_tgt, mask)
    return loss
  
class EncodecMultiClassLoss(EncodecFlattenLoss):
  def __init__(self, feature_list):
    super().__init__(feature_list)
  
  def __call__(self, logits_dict, shifted_tgt, mask):
    train_loss_list = []
    for idx, key in enumerate(self.feature_list):
      training_loss = self.get_nll_loss(logits_dict[key], shifted_tgt[..., idx], mask)
      train_loss_list.append(training_loss)
    total_loss = sum(train_loss_list) / len(train_loss_list)
    return total_loss

########################### Learning rate Scheduler ################################
'''
This scheduler is from  https://gaussian37.github.io/dl-pytorch-lr_scheduler/#custom-cosineannealingwarmrestarts-1
It's basically a cosine annealing scheduler with warm restarts including two methods, warm up start and reducing maximum lr.
'''

class CosineAnnealingWarmUpRestarts(_LRScheduler):
    def __init__(self, optimizer, T_0, T_mult=1, eta_max=0.1, T_up=0, gamma=1., last_epoch=-1, eta_min=0):
        if T_0 <= 0 or not isinstance(T_0, int):
            raise ValueError("Expected positive integer T_0, but got {}".format(T_0))
        if T_mult < 1 or not isinstance(T_mult, int):
            raise ValueError("Expected integer T_mult >= 1, but got {}".format(T_mult))
        if T_up < 0 or not isinstance(T_up, int):
            raise ValueError("Expected positive integer T_up, but got {}".format(T_up))
        self.T_0 = T_0
        self.T_mult = T_mult
        self.base_eta_max = eta_max
        self.eta_max = eta_max
        self.T_up = T_up
        self.T_i = T_0
        self.gamma = gamma
        self.cycle = 0
        self.T_cur = last_epoch
        super(CosineAnnealingWarmUpRestarts, self).__init__(optimizer, last_epoch)
    
    def get_lr(self):
        if self.T_cur == -1:
            return self.base_lrs
        elif self.T_cur < self.T_up:
            return [(self.eta_max - base_lr)*self.T_cur / self.T_up + base_lr for base_lr in self.base_lrs]
        else:
            return [base_lr + (self.eta_max - base_lr) * (1 + math.cos(math.pi * (self.T_cur-self.T_up) / (self.T_i - self.T_up))) / 2
                    for base_lr in self.base_lrs]

    def step(self, epoch=None):
        if epoch is None:
            epoch = self.last_epoch + 1
            self.T_cur = self.T_cur + 1
            if self.T_cur >= self.T_i:
                self.cycle += 1
                self.T_cur = self.T_cur - self.T_i
                self.T_i = (self.T_i - self.T_up) * self.T_mult + self.T_up
        else:
            if epoch >= self.T_0:
                if self.T_mult == 1:
                    self.T_cur = epoch % self.T_0
                    self.cycle = epoch // self.T_0
                else:
                    n = int(math.log((epoch / self.T_0 * (self.T_mult - 1) + 1), self.T_mult))
                    self.cycle = n
                    self.T_cur = epoch - self.T_0 * (self.T_mult ** n - 1) / (self.T_mult - 1)
                    self.T_i = self.T_0 * self.T_mult ** (n)
            else:
                self.T_i = self.T_0
                self.T_cur = epoch
                
        self.eta_max = self.base_eta_max * (self.gamma**self.cycle)
        self.last_epoch = math.floor(epoch)
        for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr

class CosineLRScheduler(_LRScheduler):
    """Cosine LR scheduler.
    Args:
        optimizer (Optimizer): Torch optimizer.
        warmup_steps (int): Number of warmup steps.
        total_steps (int): Total number of steps.
        lr_min_ratio (float): Minimum learning rate.
        cycle_length (float): Cycle length.
    """
    def __init__(self, optimizer: Optimizer, total_steps: int, warmup_steps: int,
                 lr_min_ratio: float = 0.0, cycle_length: float = 1.0):
        self.warmup_steps = warmup_steps
        assert self.warmup_steps >= 0
        self.total_steps = total_steps
        assert self.total_steps >= 0
        self.lr_min_ratio = lr_min_ratio
        self.cycle_length = cycle_length
        super().__init__(optimizer)

    def _get_sched_lr(self, lr: float, step: int):
        if step < self.warmup_steps:
            lr_ratio = step / self.warmup_steps
            lr = lr_ratio * lr
        elif step <= self.total_steps:
            s = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr_ratio = self.lr_min_ratio + 0.5 * (1 - self.lr_min_ratio) * \
                (1. + math.cos(math.pi * s / self.cycle_length))
            lr = lr_ratio * lr
        else:
            lr_ratio = self.lr_min_ratio
            lr = lr_ratio * lr
        return lr

    def get_lr(self):
        return [self._get_sched_lr(lr, self.last_epoch) for lr in self.base_lrs]