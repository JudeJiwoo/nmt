import time
import pickle
import os
from pathlib import Path
from typing import Union

import torch
import torchaudio
from torch.utils.data import DataLoader
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

import wandb
from collections import defaultdict
from tqdm.auto import tqdm

from .model_zoo import DoubleSequentialTransformer
from .symbolic_encoding.compile_utils import reverse_shift_and_pad_for_tensor
from .symbolic_encoding.metric_utils import *
from .symbolic_encoding import metric_utils
from .utils import add_conti_in_valid

# import sys
# sys.path.append('/home/clay/userdata/symbolic-music-encoding/audiocraft')

# from audiocraft.solvers import CompressionSolver


class LanguageModelTrainer:
  def __init__(
      self,
      model:DoubleSequentialTransformer,
      optimizer: torch.optim.Optimizer, 
      scheduler: torch.optim.lr_scheduler._LRScheduler, 
      loss_fn, 
      midi_decoder, 
      train_set, 
      valid_set, 
      save_dir: str, 
      vocab, 
      use_ddp:bool,
      use_fp16:bool,
      world_size:int,
      batch_size:int,
      infer_target_len:int,
      gpu_id:int,
      sampling_method:str,
      sampling_threshold:float,
      sampling_temperature:float,
      config
  ):
    self.model = model
    self.optimizer = optimizer
    self.scheduler = scheduler
    self.loss_fn = loss_fn
    self.train_set = train_set
    self.valid_set = valid_set
    self.vocab = vocab
    self.use_ddp = use_ddp
    self.world_size = world_size
    self.batch_size = batch_size
    self.gpu_id = gpu_id
    self.sampling_method = sampling_method
    self.sampling_threshold = sampling_threshold
    self.sampling_temperature = sampling_temperature
    self.config = config

    self.train_loader = self.generate_data_loader(train_set, shuffle=True, drop_last=False)
    self.valid_loader = self.generate_data_loader(valid_set, shuffle=False, drop_last=False)
    self.save_dir = Path(save_dir)
    self.save_dir.mkdir(exist_ok=True, parents=True)
    if use_ddp:
      self.device = torch.device(f'cuda:{self.gpu_id}')
      self.model.to(self.device) 
      self.model = DDP(self.model, device_ids=[self.gpu_id], find_unused_parameters=True)
    else:
      self.device=config.train_params.device
      self.model.to(self.device)
    if use_fp16:
      self.use_fp16 = True
      self.scaler = torch.cuda.amp.GradScaler()
    else:
      self.use_fp16 = False
    self.grad_clip=config.train_params.grad_clip
    self.num_cycles_for_inference=config.train_params.num_cycles_for_inference
    self.num_cycles_for_model_checkpoint=config.train_params.num_cycles_for_model_checkpoint
    self.iterations_per_training_cycle=config.train_params.iterations_per_training_cycle
    self.iterations_per_validation_cycle=config.train_params.iterations_per_validation_cycle
    self.make_log=config.general.make_log
    self.num_uncond_generation=config.inference_params.n_uncond_generation
    self.num_cond_generation=config.inference_params.n_cond_generation
    self.num_max_seq_len=infer_target_len
    self.infer_and_log=config.general.infer_and_log
    self.log_train_metric=config.general.log_train_metric
    if type(config.nn_params.prediction_order[0]) == str:
      assert config.nn_params.prediction_order[0] == config.data_params.first_pred_feature, "first_pred_feature should be the first element of prediction_order"

    self.best_valid_accuracy = 0
    self.best_valid_loss = 100

    self.training_loss = []
    self.validation_loss = []
    self.validation_acc = []

    self.midi_decoder = midi_decoder
    self.set_save_out()

  def set_save_out(self):
    if self.infer_and_log:
      self.valid_out_dir = self.save_dir / 'valid_out'
      os.makedirs(self.valid_out_dir, exist_ok=True)

  def save_model(self, path):
    if isinstance(self.model, DDP):
      torch.save({'model':self.model.module.state_dict(), 'optim':self.optimizer.state_dict()}, path)
    else:
      torch.save({'model':self.model.state_dict(), 'optim':self.optimizer.state_dict()}, path)


  def generate_data_loader(self, dataset, shuffle=False, drop_last=False) -> DataLoader:
    if self.use_ddp:
      sampler = DistributedSampler(dataset, 
                                  num_replicas=self.world_size, 
                                  rank=self.gpu_id,
                                  shuffle=shuffle)
      return DataLoader(dataset, 
                        batch_size=self.batch_size, 
                        shuffle=False, 
                        drop_last=drop_last, 
                        sampler=sampler)
    return DataLoader(dataset, batch_size=self.batch_size, shuffle=shuffle, drop_last=drop_last)

  def train_by_num_iter(self, num_iters):
    generator = iter(self.train_loader)
    for i in tqdm(range(num_iters)):
      try:
        batch = next(generator)
      except StopIteration:
        self.train_set._update_segments_for_trainset(random_seed=i)
        self.train_loader = self.generate_data_loader(self.train_set, shuffle=True, drop_last=False)
        generator = iter(self.train_loader)
        batch = next(generator)

      loss_value, loss_dict = self._train_by_single_batch(batch)
      loss_dict = self._rename_dict(loss_dict, 'train')
      self.training_loss.append(loss_value)
      # train loss logging 
      if (i+1) % self.iterations_per_training_cycle == 0 and self.make_log:
        wandb.log(loss_dict, step=i)
      # train acc logging
      if (i+1) % (self.iterations_per_training_cycle * 10) == 0 and self.make_log:
        validation_loss, num_nonmask_tokens, loss_dict, num_tokens_by_feature, correct_guess_by_feature = self._get_valid_loss_and_acc_from_batch(batch, train=True)
        train_metric_dict = self._get_train_accuracy(num_nonmask_tokens, num_tokens_by_feature, correct_guess_by_feature)
        train_metric_dict.update(loss_dict)
        train_metric_dict = self._rename_dict(train_metric_dict, 'train')
        wandb.log(train_metric_dict, step=i)
        # delete unnecessary variables unless it reaches the last iteration
        if i != num_iters - 1:
          del validation_loss, num_nonmask_tokens, loss_dict, num_tokens_by_feature, correct_guess_by_feature, train_metric_dict
      if (i+1) % self.iterations_per_validation_cycle == 0:
        self.model.eval()
        validation_loss, validation_acc, validation_metric_dict = self.validate()
        validation_metric_dict['acc'] = validation_acc
        validation_metric_dict = self._rename_dict(validation_metric_dict, 'valid')
        # if self.log_train_metric and (i+1) % (self.iterations_per_validation_cycle * 5) == 0:
        #   _, train_acc, train_metric_dict = self.validate(self.train_loader)
        #   train_metric_dict['acc'] = train_acc
        #   train_metric_dict = self._rename_dict(train_metric_dict, 'train')
        #   validation_metric_dict = {**validation_metric_dict, **train_metric_dict}
        if self.make_log:
          wandb.log(validation_metric_dict, step=i)
        self.validation_loss.append(validation_loss)
        self.validation_acc.append(validation_acc)
        self.best_valid_loss = min(validation_loss, self.best_valid_loss)
        if (i+1) % (self.num_cycles_for_inference * self.iterations_per_validation_cycle) == 0 and self.infer_and_log:
          self.inference_and_log(i, self.num_uncond_generation, self.num_cond_generation, self.num_max_seq_len)
        if (i+1) % (self.iterations_per_validation_cycle * self.num_cycles_for_model_checkpoint) == 0:
          self.save_model(self.save_dir / f'iter{i}_loss{validation_loss:.4f}.pt')
        self.model.train()
    # save model after training
    self.save_model(self.save_dir / f'iter{i}_loss{validation_loss:.4f}.pt')

  def _train_by_single_batch(self, batch):
    start_time = time.time()
    loss, _, loss_dict = self._get_loss_pred_from_single_batch(batch)
    self.optimizer.zero_grad()
    if self.use_fp16:
      # with torch.cuda.amp.autocast(dtype=torch.float16):
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.optimizer.step()
    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.scheduler is not None:
      self.scheduler.step()
    loss_dict['time'] = time.time() - start_time
    loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
    return loss.item(), loss_dict

  def _get_loss_pred_from_single_batch(self, batch):
    segment, mask, _ = batch
    input_seq, target = segment[:, :-1], segment[:, 1:] 
    input_seq = input_seq.to(self.device)
    target = target.to(self.device)
    mask = mask[:, :-1].to(self.device)
    if self.use_fp16:
      with torch.cuda.amp.autocast():
        logits = self.model(input_seq, target)
        loss = self.loss_fn(logits, target, mask)
    else:
      logits = self.model(input_seq, None)
      loss = self.loss_fn(logits, target, mask)
    loss_dict = {'total':loss.item()}
    return loss, logits, loss_dict

  def _get_valid_loss_and_acc_from_batch(self, batch, train=False):
    segment, mask, _ = batch
    input_seq, target = segment[:, :-1], segment[:, 1:] 
    loss, logits, loss_dict = self._get_loss_pred_from_single_batch(batch)
    prob = torch.softmax(logits, dim=-1)
    num_tokens = torch.sum(mask) # batch x seq_len(512)
    target = target.to(self.device) # batch x seq_len(512)
    mask = mask[:, :-1].to(self.device) # batch x seq_len(512)

    selected_tokens = torch.argmax(prob, dim=-1) * mask
    shifted_tgt_with_mask = target * mask
    num_correct_guess = torch.sum(selected_tokens == shifted_tgt_with_mask) - torch.sum(mask == 0)

    validation_loss = loss.item() * num_tokens
    num_correct_guess = num_correct_guess.item()
    return validation_loss, num_tokens, loss_dict, None, num_correct_guess

  def _get_train_accuracy(self, num_tokens, num_tokens_by_feature, num_correct_guess):
    return num_correct_guess / num_tokens

  def validate(self, external_loader=None):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.model.eval()
    total_validation_loss = 0
    total_num_correct_guess = 0
    total_num_tokens = 0
    validation_metric_dict = defaultdict(float)
    with torch.inference_mode():
      '''
      total_num_tokens = num_batch x batch x seq_len
      '''
      for batch in tqdm(loader, leave=False):
        validation_loss, num_tokens, loss_dict, _, num_correct_guess = self._get_valid_loss_and_acc_from_batch(batch)
        total_validation_loss += validation_loss
        total_num_tokens += num_tokens
        total_num_correct_guess += num_correct_guess
        for key, value in loss_dict.items():
          validation_metric_dict[key] += value * num_tokens
      for key in validation_metric_dict.keys():
        validation_metric_dict[key] /= total_num_tokens

    return total_validation_loss / total_num_tokens, total_num_correct_guess / total_num_tokens, validation_metric_dict

  def _make_midi_from_generated_output(self, generated_output, iter, seed, condition=None):
    if condition is not None:
      path_addition = "cond_"
    else:
      path_addition = ""
      syntax_error_fn:Union[check_syntax_errors_in_inference_for_cp,
                      check_syntax_errors_in_inference_for_nb,
                      check_syntax_errors_in_inference_for_remi] = getattr(metric_utils, f'check_syntax_errors_in_inference_for_{self.vocab.encoding_scheme}')
      second_input = self.vocab if self.vocab.encoding_scheme == 'remi' else self.vocab.feature_list
      error_metric_dict = syntax_error_fn(generated_output, second_input)
      if self.make_log:
        wandb.log(error_metric_dict, step=(iter+seed))
    with open(self.valid_out_dir / f"{path_addition}generated_output_{iter}_seed_{seed}.pkl", 'wb') as f:
      pickle.dump(generated_output, f)
    self.midi_decoder(generated_output, self.valid_out_dir / f"{path_addition}midi_decoded_{iter}_seed_{seed}.mid")
    if self.make_log:
      log_dict = {}
      log_dict[f'{path_addition}gen_score'] = wandb.Image(str(self.valid_out_dir / f'{path_addition}midi_decoded_{iter}_seed_{seed}.png')) 
      log_dict[f'{path_addition}gen_audio'] = wandb.Audio(str(self.valid_out_dir / f'{path_addition}midi_decoded_{iter}_seed_{seed}.mp3'))
      wandb.log(log_dict, step=(iter+seed))
      print(f"{path_addition}inference is logged: Iter {iter} / seed {seed}")
    return generated_output

  @torch.inference_mode()
  def inference_and_log(self, iter, num_uncond_generation=5, num_cond_generation=5, max_seq_len=10000):
    self.model.eval()
    for i in range(num_uncond_generation):
      try:
        start_time = time.time()
        uncond_generated_output = self.model.generate(manual_seed=i, max_seq_len=max_seq_len, condition=None, \
          sampling_method=self.sampling_method, threshold=self.sampling_threshold, temperature=self.sampling_temperature)
        if len(uncond_generated_output) == 0: continue
        print(f"unconditional generation time_{iter}: {time.time() - start_time:.4f}")
        print(f"unconditional length of generated_output: {uncond_generated_output.shape[1]}")
        self._make_midi_from_generated_output(uncond_generated_output, iter, i, None)
      except Exception as e:
        print(e)
    condition_list = [x[1] for x in self.valid_set.data_list[:num_cond_generation] ] 
    for i in range(num_cond_generation):
      condition = self.valid_set.get_segments_with_tune_idx(condition_list[i], 0)[0]
      try:
        start_time = time.time()
        generated_output = self.model.generate(manual_seed=i, max_seq_len=max_seq_len, condition=condition, \
          sampling_method=self.sampling_method, threshold=self.sampling_threshold, temperature=self.sampling_temperature)
        if len(generated_output) == 0: continue
        print(f"conditional generation time_{iter}: {time.time() - start_time:.4f}")
        print(f"conditional length of generated_output: {generated_output.shape[1]}")
        self._make_midi_from_generated_output(generated_output, iter+num_uncond_generation, i, condition)
      except Exception as e:
        print(e)

  def _rename_dict(self, adict, prefix='train'):
    keys = list(adict.keys())
    for key in keys:
      adict[f'{prefix}.{key}'] = adict.pop(key)
    return dict(adict)

class SingleClassLMTrainer(LanguageModelTrainer):
  def __init__(self, model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config):
    super().__init__(model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config)
    '''
    for remi encoding scheme
    '''

  def _get_loss_pred_from_single_batch(self, batch, valid=False):
    segment, mask, _ = batch
    input_seq, target = segment[:, :-1], segment[:, 1:] 
    input_seq = input_seq.to(self.device)
    target = target.to(self.device)
    mask = mask[:, :-1].to(self.device)
    if self.use_fp16:
      with torch.cuda.amp.autocast():
        logits = self.model(input_seq, target)
        if not valid:
          total_loss, loss_dict = self.loss_fn(logits, target, mask, None)
          return total_loss, logits, {'total':total_loss.item()}
        else:
          total_loss, loss_dict = self.loss_fn(logits, target, mask, self.vocab)
          loss_dict['total'] = total_loss.item()
          return total_loss, logits, loss_dict
    else:
      logits = self.model(input_seq, target)
      if not valid:
        total_loss, loss_dict = self.loss_fn(logits, target, mask, None)
        return total_loss, logits, {'total':total_loss.item()}
      else:
        total_loss, loss_dict = self.loss_fn(logits, target, mask, self.vocab)
        loss_dict['total'] = total_loss.item()
        return total_loss, logits, loss_dict

  def _get_valid_loss_and_acc_from_batch(self, batch, train=False):
    segment, mask, _ = batch
    mask = mask[:, :-1]
    _, target = segment[:, :-1], segment[:, 1:] 
    loss, logits, loss_dict = self._get_loss_pred_from_single_batch(batch, valid=True)
    prob = torch.softmax(logits, dim=-1)
    num_nonmask_tokens = torch.sum(mask) # [b, t]
    target = target.to(self.device) # [b, t]
    mask = mask.to(self.device)

    prob_with_mask = torch.argmax(prob, dim=-1) * mask # [b, t]
    shifted_tgt_with_mask = target * mask # [b, t]

    correct_guess_by_feature = defaultdict(int)
    num_tokens_by_feature = defaultdict(int)
    tokens_idx = prob_with_mask.flatten(0,1) # [b*t]
    answers_idx = shifted_tgt_with_mask.flatten(0,1) # [b*t]
    if self.vocab.encoding_scheme == 'remi':
      eos_idx = 2
    for feature in self.vocab.feature_list:
      feature_mask = self.vocab.total_mask[feature].to(self.device) # [327,]
      mask_for_target = feature_mask[answers_idx] # [b*t]
      if feature == 'type': # because Bar token is 0, we need to add 1 to calculate accuracy
        valid_pred = (tokens_idx+1) * mask_for_target
        valid_answers = (answers_idx+1) * mask_for_target
        eos_mask = valid_answers != eos_idx # because EOS is also working as a padding
        correct_guess_by_feature[feature] += torch.sum(valid_pred[eos_mask] == valid_answers[eos_mask]).item() - torch.sum(mask_for_target[eos_mask] == 0).item()
        num_tokens_by_feature[feature] += torch.sum(mask_for_target[eos_mask]).item()
      else:
        valid_pred = tokens_idx * mask_for_target # [b, t]
        valid_answers = answers_idx * mask_for_target # [b, t]
        correct_guess_by_feature[feature] += torch.sum(valid_pred == valid_answers).item() - torch.sum(mask_for_target == 0).item()
        num_tokens_by_feature[feature] += torch.sum(mask_for_target).item()
    validation_loss = loss.item() * num_nonmask_tokens.item()
    return validation_loss, num_nonmask_tokens, loss_dict, num_tokens_by_feature, correct_guess_by_feature

  def _get_train_accuracy(self, num_tokens, num_tokens_by_feature, num_correct_guess_by_feature):
    total_num_correct_guess = 0
    total_num_tokens = 0
    acc_dict = {}
    for feature, num_correct_guess in num_correct_guess_by_feature.items():
      if feature == 'type':
        continue
      total_num_correct_guess += num_correct_guess
      total_num_tokens += num_tokens_by_feature[feature]
      if num_tokens_by_feature[feature] == 0:
        continue
      acc_dict[f"{feature}_acc"] = num_correct_guess / num_tokens_by_feature[feature]
    total_accuracy = total_num_correct_guess / total_num_tokens
    acc_dict['total_acc'] = total_accuracy
    return acc_dict

  def validate(self, external_loader=None):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.model.eval()
    total_validation_loss = 0
    total_num_tokens = 0
    total_num_valid_tokens = 0
    total_num_correct_guess = 0
    validation_metric_dict = defaultdict(float)
    total_num_tokens_by_feature = defaultdict(int)
    total_num_correct_guess_dict = defaultdict(int)
    with torch.inference_mode():
      '''
      total_num_tokens: for calculating loss, nonmask tokens
      total_num_valid_tokens: for calculating accuracy, valid tokens
      '''
      for num_iter, batch in enumerate(tqdm(loader, leave=False)):
        if num_iter == len(self.valid_loader):
          if loader is not self.valid_loader: # when validate with train_loader
            break
        validation_loss, num_nonmask_tokens, loss_dict, num_tokens_by_feature, num_correct_guess_by_feature = self._get_valid_loss_and_acc_from_batch(batch)
        total_validation_loss += validation_loss
        total_num_tokens += num_nonmask_tokens.item()
        for key, num_tokens in num_tokens_by_feature.items():
          total_num_tokens_by_feature[key] += num_tokens
          if key == 'type':
            continue
          total_num_valid_tokens += num_tokens # num tokens are all the same for each musical type, torch.sum(mask)
        for key, num_correct_guess in num_correct_guess_by_feature.items():
          total_num_correct_guess_dict[key] += num_correct_guess
          if key == 'type':
            continue
          total_num_correct_guess += num_correct_guess
        for key, value in loss_dict.items():
          if key == 'total':
            validation_metric_dict[key] += value * num_nonmask_tokens
          else:
            feature_name = key.split('_')[0]
            validation_metric_dict[key] += value * num_tokens_by_feature[feature_name]

      for key in validation_metric_dict.keys():
        if key == 'total':
          validation_metric_dict[key] /= total_num_tokens
        else:
          feature_name = key.split('_')[0]
          if total_num_tokens_by_feature[feature_name] == 0:
            continue
          validation_metric_dict[key] /= total_num_tokens_by_feature[feature_name]

      for key in total_num_tokens_by_feature.keys():
        num_tokens = total_num_tokens_by_feature[key]
        num_correct = total_num_correct_guess_dict[key]
        if num_tokens == 0:
          continue
        validation_metric_dict[f'{key}_acc'] = num_correct / num_tokens
    return total_validation_loss / total_num_tokens, total_num_correct_guess / total_num_valid_tokens, validation_metric_dict

class MultiClassLMTrainer(LanguageModelTrainer):
  def __init__(self, model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config):
    super().__init__(model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config)

  def _get_num_valid_and_correct_tokens(self, prob, ground_truth, mask, ignore_token=None, conti_token=None):
    '''
    We designed the vocab in both cp and nb encoding scheme to have the same idx in ignore and conti token
    ignore token: 0
    conti token: 1
    '''
    valid_prob = torch.argmax(prob, dim=-1) * mask
    valid_ground_truth = ground_truth * mask

    if ignore_token is None and conti_token is None: 
      num_valid_tokens = torch.sum(mask)
      num_correct_tokens = torch.sum(valid_prob == valid_ground_truth) - torch.sum(mask == 0)
    elif ignore_token is not None and conti_token is None:
      ignore_mask = valid_ground_truth != ignore_token # batch x seq_len
      num_valid_tokens = torch.sum(ignore_mask)
      num_correct_tokens = torch.sum(valid_prob[ignore_mask] == valid_ground_truth[ignore_mask]) # by using mask, the tensor becomes 1d
    elif ignore_token is not None and conti_token is not None:
      ignore_conti_mask = (valid_ground_truth != ignore_token) & (valid_ground_truth != conti_token)
      num_valid_tokens = torch.sum(ignore_conti_mask)
      num_correct_tokens = torch.sum(valid_prob[ignore_conti_mask] == valid_ground_truth[ignore_conti_mask])
    return num_correct_tokens.item(), num_valid_tokens.item()
      
  def _get_loss_pred_from_single_batch(self, batch, valid=False):
    segment, mask, _ = batch
    input_seq, target = segment[:, :-1], segment[:, 1:] 
    input_seq = input_seq.to(self.device)
    target = target.to(self.device)
    mask = mask[:, :-1].to(self.device)
    if self.use_fp16:
      with torch.cuda.amp.autocast(dtype=torch.float16):
        logits_dict = self.model(input_seq, target)
        total_loss, loss_dict = self.loss_fn(logits_dict, target, mask, valid)
    else:
      logits_dict = self.model(input_seq, target)
      total_loss, loss_dict = self.loss_fn(logits_dict, target, mask, valid)
    if valid:
      loss_dict['total'] = total_loss.item()
    else:
      loss_dict = {'total':total_loss.item()}
    return total_loss, logits_dict, loss_dict

  def _get_valid_loss_and_acc_from_batch(self, batch, train=False):
    '''
    in this method, valid means handled with both ignore token and mask
    when valid tokens with only mask, it is called num_nonmask_tokens

    input_seq, target: batch x seq_len x num_features
    mask: batch x seq_len, 0 for padding
    prob: batch x seq_len x total_vocab_size
    '''
    segment, mask, _ = batch
    input_seq, target = segment[:, :-1], segment[:, 1:] 
    total_loss, logits_dict, loss_dict = self._get_loss_pred_from_single_batch(batch, valid=True)
    probs_dict = {key:torch.softmax(value, dim=-1) for key, value in logits_dict.items()}
    num_nonmask_tokens = torch.sum(mask)
    input_seq = input_seq.to(self.device)
    target = add_conti_in_valid(target, self.config.data_params.encoding_scheme).to(self.device)
    mask = mask[:, :-1].to(self.device)
    
    correct_guess_by_feature = defaultdict(int)
    num_tokens_by_feature = defaultdict(int)
    for idx, key in enumerate(self.vocab.feature_list):
      if key == 'type':
        num_correct_tokens, num_valid_tokens = self._get_num_valid_and_correct_tokens(probs_dict[key], target[..., idx], mask, ignore_token=None, conti_token=None)
      elif key == 'chord' or key == 'tempo' or key == 'instrument':
        num_correct_tokens, num_valid_tokens = self._get_num_valid_and_correct_tokens(probs_dict[key], target[..., idx], mask, ignore_token=0, conti_token=9999)
      elif key == 'beat':
        # NB's beat vocab has Ignore and CONTI token
        # CP's beat vocab has Ignore and BAR token, we exclude BAR token in accuracy calculation for parity with NB
        num_correct_tokens, num_valid_tokens = self._get_num_valid_and_correct_tokens(probs_dict[key], target[..., idx], mask, ignore_token=0, conti_token=9999)
        type_idx = self.vocab.feature_list.index('type')
        beat_idx = self.vocab.feature_list.index('beat')
        # type beat error algorithm doesn't support beat 1st option
        num_valid_BT_tokens, num_unmatched_BT_tokens, num_backwards_BT_tokens = getattr(metric_utils, f'type_beat_errors_in_validation_{self.vocab.encoding_scheme}')(probs_dict[key], input_seq[..., type_idx], input_seq[..., beat_idx], mask)
        if not train:
          self.type_beat_total_case += num_valid_BT_tokens
          self.type_beat_unmatch_error_case += num_unmatched_BT_tokens
          self.type_beat_backwards_error_case += num_backwards_BT_tokens
      else:
        num_correct_tokens, num_valid_tokens = self._get_num_valid_and_correct_tokens(probs_dict[key], target[..., idx], mask, ignore_token=0, conti_token=None)
      correct_guess_by_feature[key] = num_correct_tokens
      num_tokens_by_feature[key] = num_valid_tokens
    validation_loss = total_loss.item() * num_nonmask_tokens.item()
    return validation_loss, num_nonmask_tokens, loss_dict, num_tokens_by_feature, correct_guess_by_feature

  def _get_train_accuracy(self, num_tokens, num_tokens_by_feature, num_correct_guess_by_feature):
    total_num_correct_guess = 0
    total_num_tokens = 0
    acc_dict = {}
    for feature, num_correct_guess in num_correct_guess_by_feature.items():
      if feature == 'type':
        continue
      total_num_correct_guess += num_correct_guess
      total_num_tokens += num_tokens_by_feature[feature]
      acc_dict[f"{feature}_acc"] = num_correct_guess / num_tokens_by_feature[feature]
    total_accuracy = total_num_correct_guess / total_num_tokens
    acc_dict['total_acc'] = total_accuracy
    return acc_dict

  def validate(self, external_loader=None):
    if external_loader and isinstance(external_loader, DataLoader):
      loader = external_loader
      print('An arbitrary loader is used instead of Validation loader')
    else:
      loader = self.valid_loader

    self.model.eval()
    total_validation_loss = 0
    total_num_correct_guess = 0
    total_num_tokens = 0
    total_num_valid_tokens = 0
    validation_metric_dict = defaultdict(float)
    total_num_tokens_by_feature = defaultdict(int)
    total_num_correct_guess_dict = defaultdict(int)
    self.type_beat_total_case = 0
    self.type_beat_unmatch_error_case = 0
    self.type_beat_backwards_error_case = 0
    # self.beat_difference_metric_list = []
    with torch.inference_mode():
      '''
      mask is used to calculate loss, accuracy
      validation_loss: sum of loss for valid tokens conditioned on mask
      num_nonmask_tokens: sum of tokens conditioned on mask
      num_tokens_by_feature: sum of valid tokens(handle ignore) for each musical features
      num_correct_guess_by_feature: sum of correct tokens(handle ignore) for each musical features
      '''
      for num_iter, batch in tqdm(enumerate(loader), leave=False):
        if num_iter == len(self.valid_loader):
          if loader is not self.valid_loader: # when validate with train_loader
            break
        validation_loss, num_nonmask_tokens, loss_dict, num_tokens_by_feature, num_correct_guess_by_feature = self._get_valid_loss_and_acc_from_batch(batch)
        total_validation_loss += validation_loss
        total_num_tokens += num_nonmask_tokens
        for key, num_tokens in num_tokens_by_feature.items():
          total_num_tokens_by_feature[key] += num_tokens
          if key == 'type': # because cp and nb have different number of type tokens, we don't want to calculate accuracy for type token
            continue
          total_num_valid_tokens += num_tokens # num tokens are all the same for each musical type, torch.sum(mask)
        for key, num_correct_guess in num_correct_guess_by_feature.items():
          total_num_correct_guess_dict[key] += num_correct_guess
          if key == 'type':
            continue
          total_num_correct_guess += num_correct_guess
        for key, value in loss_dict.items():
          if key == 'total':
            validation_metric_dict[key] += value * num_nonmask_tokens
          else:
            if torch.isnan(value): # in case num valid tokens is 0 because of mask
              continue
            feature_name = key.split('_')[0]
            validation_metric_dict[key] += value * num_tokens_by_feature[feature_name]
      # save attn_output
      # self.attn_output = []
      # if external_loader is None and self.config.nn_params.sub_decoder == 'CrossAttention_Strategy': # when validate with valid_loader
      #   # self_attn_output_weight = self.model.sub_decoder.sub_decoder_layers[0].self_attn_block.self_attn_output_weights
      #   cross_attn_output_weight = self.model.sub_decoder.sub_decoder_layers[0].cross_attn_block.cross_attn_output_weights
      #   self.attn_output.append({'cross_attn_output_weight':cross_attn_output_weight})
      
      for key in validation_metric_dict.keys():
        if key == 'total':
          validation_metric_dict[key] /= total_num_tokens
        else:
          feature_name = key.split('_')[0]
          if total_num_tokens_by_feature[feature_name] == 0:
            continue
          validation_metric_dict[key] /= total_num_tokens_by_feature[feature_name]
      for (key_t, num_tokens), (key_c, num_correct) in zip(total_num_tokens_by_feature.items(), total_num_correct_guess_dict.items()):
        validation_metric_dict[f'{key_c}_acc'] = num_correct / num_tokens
      validation_metric_dict['beat_type_unmatched_error_in_validation'] = self.type_beat_unmatch_error_case / self.type_beat_total_case
      validation_metric_dict['beat_backwards_error_in_validation'] = self.type_beat_backwards_error_case / self.type_beat_total_case
      # beat difference metric
      # beat_cos_sim_list = [list(dict.values())[0] for dict in self.beat_difference_metric_list]
      # validation_metric_dict['beat_upon_type_cos_sim'] = sum(beat_cos_sim_list) / len(beat_cos_sim_list)
      # beat_same_ratio_list = [list(dict.values())[1] for dict in self.beat_difference_metric_list]
      # validation_metric_dict['beat_upon_type_sim_ratio'] = sum(beat_same_ratio_list) / len(beat_same_ratio_list)
    # print(f"total_num_tokens_by_feature: {total_num_tokens_by_feature}")
    return total_validation_loss / total_num_tokens, total_num_correct_guess / total_num_valid_tokens, validation_metric_dict

  def _make_midi_from_generated_output(self, generated_output, iter, seed, condition=None):
    if self.config.data_params.first_pred_feature != 'type':
      generated_output = reverse_shift_and_pad_for_tensor(generated_output, self.config.data_params.first_pred_feature)
    if condition is not None:
      path_addition = "cond_"
    else:
      path_addition = ""
      syntax_error_fn:Union[check_syntax_errors_in_inference_for_cp,
                      check_syntax_errors_in_inference_for_nb,
                      check_syntax_errors_in_inference_for_remi] = getattr(metric_utils, f'check_syntax_errors_in_inference_for_{self.vocab.encoding_scheme}')
      second_input = self.vocab if 'remi' in self.vocab.encoding_scheme else self.vocab.feature_list
      error_metric_dict = syntax_error_fn(generated_output, second_input)
      if self.make_log:
        wandb.log(error_metric_dict, step=(iter+seed))
    # save generated_output as pickle
    with open(self.valid_out_dir / f"{path_addition}generated_output_{iter}_seed_{seed}.pkl", 'wb') as f:
      pickle.dump(generated_output, f)
    self.midi_decoder(generated_output, self.valid_out_dir / f"{path_addition}midi_decoded_{iter}_seed_{seed}.mid")
    if self.make_log and self.infer_and_log:
      log_dict = {}
      log_dict[f'{path_addition}gen_score'] = wandb.Image(str(self.valid_out_dir / f'{path_addition}midi_decoded_{iter}_seed_{seed}.png')) 
      log_dict[f'{path_addition}gen_audio'] = wandb.Audio(str(self.valid_out_dir / f'{path_addition}midi_decoded_{iter}_seed_{seed}.mp3'))
      wandb.log(log_dict, step=(iter+seed))
      print(f"{path_addition}inference is logged: Iter {iter} / seed {seed}")
  
class EncodecFlattenTrainer(LanguageModelTrainer):
  def __init__(self, model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config):
    super().__init__(model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config)
    # self.encodec_pretrained_model = CompressionSolver.model_from_checkpoint("/home/clay/userdata/symbolic-music-encoding/encodec_checkpoint/checkpoint.th", 'cuda')

  def train_by_num_iter(self, num_iters):
    generator = iter(self.train_loader)
    for i in tqdm(range(num_iters)):
      try:
        batch = next(generator)
      except StopIteration:
        self.train_loader = self.generate_data_loader(self.train_set, shuffle=True, drop_last=False)
        generator = iter(self.train_loader)
        batch = next(generator)
    
      self.model.train()
      _, loss_dict = self._train_by_single_batch(batch)
      loss_dict = self._rename_dict(loss_dict, 'train')
      if (i+1) % self.iterations_per_training_cycle == 0 and self.make_log:
        wandb.log(loss_dict, step=i)
      if (i+1) % self.iterations_per_validation_cycle == 0:
        self.model.eval()
        validation_loss, validation_acc, validation_metric_dict = self.validate()
        validation_metric_dict['acc'] = validation_acc
        validation_metric_dict = self._rename_dict(validation_metric_dict, 'valid')
        if self.make_log:
          wandb.log(validation_metric_dict, step=i)
      if (i+1) % (self.num_cycles_for_inference * self.iterations_per_validation_cycle) == 0 and self.infer_and_log:
        self.inference_and_log(i, self.num_uncond_generation, self.num_cond_generation, self.num_max_seq_len)
      if (i+1) % (self.iterations_per_validation_cycle * self.num_cycles_for_model_checkpoint) == 0:
        self.save_model(self.save_dir / f'iter{i}_loss{validation_loss:.4f}.pt')
        print(f"Model saved at {self.save_dir / f'iter{i}_loss{validation_loss:.4f}.pt'}")
    self.save_model(self.save_dir / f'iter{num_iters}_loss{validation_loss:.4f}.pt')

  def _train_by_single_batch(self, batch):
    start_time = time.time()
    loss, _, loss_dict = self._get_loss_pred_from_single_batch(batch)
    if self.use_fp16:
      self.scaler.scale(loss).backward()
      self.scaler.unscale_(self.optimizer)
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.scaler.step(self.optimizer)
      self.scaler.update()
    else:
      loss.backward()
      torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip)
      self.optimizer.step()
    self.optimizer.zero_grad()
    if not isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau) and self.scheduler is not None:
      self.scheduler.step()
    loss_dict['time'] = time.time() - start_time
    loss_dict['lr'] = self.optimizer.param_groups[0]['lr']
    return loss.item(), loss_dict
  
  def _get_loss_pred_from_single_batch(self, batch):
    input_seq, target, mask = batch
    input_seq = input_seq.to(self.device)
    target = target.to(self.device)
    mask = mask.to(self.device)
    if self.use_fp16:
      with torch.cuda.amp.autocast(dtype=torch.float16):
        logits = self.model(input_seq, target)
        loss = self.loss_fn(logits, target, mask)
    else:
      logits = self.model(input_seq, target)
      loss = self.loss_fn(logits, target, mask)
    loss_dict = {'total':loss.item()}
    return loss, logits, loss_dict
  
  def validate(self):
    loader = self.valid_loader
    total_validation_loss = 0
    total_num_correct_guess = 0
    total_num_tokens = 0
    validation_metric_dict = defaultdict(float)
    with torch.inference_mode():
      for batch in tqdm(loader, leave=False):
        validation_loss, num_tokens, num_correct_guess, loss_dict = self._get_valid_loss_and_acc_from_batch(batch)
        total_validation_loss += validation_loss
        total_num_tokens += num_tokens
        total_num_correct_guess += num_correct_guess
        for key, value in loss_dict.items():
          validation_metric_dict[key] += value * num_tokens
      for key in validation_metric_dict.keys():
        validation_metric_dict[key] /= total_num_tokens
    return total_validation_loss / total_num_tokens, total_num_correct_guess / total_num_tokens, validation_metric_dict

  def _get_valid_loss_and_acc_from_batch(self, batch):
    _, target, mask = batch
    loss, logits, loss_dict = self._get_loss_pred_from_single_batch(batch)
    num_tokens = torch.sum(mask) # batch x seq_len(512)
    target = target.to(self.device) # batch x seq_len(512)
    mask = mask.to(self.device) # batch x seq_len(512)

    probs = torch.softmax(logits, dim=-1)
    prob_with_mask = torch.argmax(probs, dim=-1) * mask
    shifted_tgt_with_mask = target * mask
    num_correct_guess = torch.sum(prob_with_mask == shifted_tgt_with_mask) - torch.sum(mask == 0)

    validation_loss = loss.item() * num_tokens
    num_correct_guess = num_correct_guess.item()
    return validation_loss, num_tokens, num_correct_guess, loss_dict
  
  def _make_audio_from_generated_outputs(self, generated_output, iter, seed, condition=None):
    if condition is not None:
      path_addition = "cond_"
    else:
      path_addition = ""
    decoded_output = self.encodec_pretrained_model.decode(generated_output)
    torchaudio.save(self.valid_out_dir / f"{path_addition}audio_generated_{iter}_seed_{seed}.mp3", decoded_output.squeeze(0), format='mp3', compression=256)
    if self.make_log:
      log_dict = {}
      log_dict[f'{path_addition}gen_audio'] = wandb.Audio(str(self.valid_out_dir / f'{path_addition}audio_generated_{iter}_seed_{seed}.mp3'))
      wandb.log(log_dict, step=(iter+seed))
      print(f"{path_addition}inference is logged: Iter {iter} / seed {seed}")
    return generated_output

  @torch.inference_mode()
  def inference_and_log(self, iter, num_uncond_generation=5, num_cond_generation=5, max_seq_len=10000):
    self.model.eval()
    for i in range(num_uncond_generation):
      try:
        start_time = time.time()
        uncond_generated_output = self.model.generate(manual_seed=i, max_seq_len=max_seq_len, condition=None, \
          sampling_method=self.sampling_method, threshold=self.sampling_threshold, temperature=self.sampling_temperature)
        if len(uncond_generated_output) == 0: continue
        print(f"unconditional generation time_{iter}: {time.time() - start_time:.4f}")
        print(f"unconditional length of generated_output: {uncond_generated_output.shape[1]}")
        self._make_audio_from_generated_outputs(uncond_generated_output, iter, i, None)
      except Exception as e:
        print(e)
    condition_list = [x[1] for x in self.valid_set.data_list[:num_cond_generation] ] 
    for i in range(num_cond_generation):
      condition = self.valid_set.get_segments_with_tune_idx(condition_list[i], 0)[0]
      try:
        start_time = time.time()
        generated_output = self.model.generate(manual_seed=i, max_seq_len=max_seq_len, condition=condition, \
          sampling_method=self.sampling_method, threshold=self.sampling_threshold, temperature=self.sampling_temperature)
        if len(generated_output) == 0: continue
        print(f"conditional generation time_{iter}: {time.time() - start_time:.4f}")
        print(f"conditional length of generated_output: {generated_output.shape[1]}")
        self._make_audio_from_generated_outputs(generated_output, iter+num_uncond_generation, i, condition)
      except Exception as e:
        print(e)

class EncodecMultiClassTrainer(EncodecFlattenTrainer):
  def __init__(self, model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config):
    super().__init__(model, optimizer, scheduler, loss_fn, midi_decoder, train_set, valid_set, save_dir, vocab, use_ddp, use_fp16, world_size, batch_size, infer_target_len, gpu_id, sampling_method, sampling_threshold, sampling_temperature, config)

  def _get_loss_pred_from_single_batch(self, batch):
    input_seq, target, mask = batch
    input_seq = input_seq.to(self.device)
    target = target.to(self.device)
    mask = mask.to(self.device)
    if self.use_fp16:
      with torch.cuda.amp.autocast(dtype=torch.float16):
        logits_dict = self.model(input_seq, target)
        loss = self.loss_fn(logits_dict, target, mask)
    else:
      logits_dict = self.model(input_seq, target)
      loss = self.loss_fn(logits_dict, target, mask)
    loss_dict = {'total':loss.item()}
    return loss, logits_dict, loss_dict

  def _get_valid_loss_and_acc_from_batch(self, batch):
    input_seq, target, mask = batch
    loss, logits_dict, loss_dict = self._get_loss_pred_from_single_batch(batch)
    probs_dict = {key:torch.softmax(value, dim=-1) for key, value in logits_dict.items()}
    num_nonmask_tokens = torch.sum(mask)
    input_seq = input_seq.to(self.device)
    target = target.to(self.device)
    mask = mask.to(self.device)

    num_correct = 0
    num_tokens = 0
    for idx, key in enumerate(self.vocab.feature_list):
      prob_with_mask = torch.argmax(probs_dict[key], dim=-1) * mask
      shifted_tgt_with_mask = target[..., idx] * mask
      num_correct += torch.sum(prob_with_mask == shifted_tgt_with_mask) - torch.sum(mask == 0)
      num_tokens += torch.sum(mask)
    validation_loss = loss.item() * num_nonmask_tokens
    return validation_loss, num_tokens.item(), num_correct.item(), loss_dict

  def validate(self):
    loader = self.valid_loader
    total_validation_loss = 0
    total_num_correct_guess = 0
    total_num_tokens = 0
    validation_metric_dict = defaultdict(float)
    with torch.inference_mode():
      for batch in tqdm(loader, leave=False):
        validation_loss, num_tokens, correct_guess, loss_dict = self._get_valid_loss_and_acc_from_batch(batch)
        total_validation_loss += validation_loss
        total_num_tokens += num_tokens
        total_num_correct_guess += correct_guess
        for key, value in loss_dict.items():
          validation_metric_dict[key] += value * num_tokens
      for key in validation_metric_dict.keys():
        validation_metric_dict[key] /= total_num_tokens
    return total_validation_loss / total_num_tokens, total_num_correct_guess / total_num_tokens, validation_metric_dict
        