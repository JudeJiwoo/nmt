import torch
import numpy as np

from collections import Counter

# TODO: refactor hard coded values
def check_syntax_errors_in_inference_for_nb(generated_output, feature_list):
  generated_output = generated_output.squeeze(0)
  type_idx = feature_list.index('type')
  beat_idx = feature_list.index('beat')
  type_beat_list = []
  for token in generated_output:
    type_beat_list.append((token[type_idx].item(), token[beat_idx].item())) # type, beat
  
  last_note = 1
  beat_type_unmatched_error_list = []
  num_unmatched_errors = 0
  beat_backwards_error_list = []
  num_backwards_errors = 0
  for type_beat in type_beat_list:
    # if type_beat[0] == 3: # same bar, same beat
    #   if type_beat[1] != 1: # conti
    #     num_unmatched_errors += 1
    #     beat_type_unmatched_error_list.append(type_beat)
    if type_beat[0] == 4: # same bar, new beat
      if type_beat[1] == 0 or type_beat[1] == 1:
        num_unmatched_errors += 1
        beat_type_unmatched_error_list.append(type_beat)
      if type_beat[1] <= last_note:
        num_backwards_errors += 1
        beat_backwards_error_list.append([last_note, type_beat])
      else:
        last_note = type_beat[1] # update last note
    elif type_beat[0] >= 5: # new bar, new beat
      if type_beat[1] == 0:
        num_unmatched_errors += 1
        beat_type_unmatched_error_list.append(type_beat)
      last_note = 1
  unmatched_error_rate = num_unmatched_errors / len(type_beat_list)
  backwards_error_rate = num_backwards_errors / len(type_beat_list)
  # print(f"error in beat_type unmatched: {beat_type_unmatched_error_list}")
  # print(f"error in beat backwards: {beat_backwards_error_list}")
  # print(f"error rate in beat_type unmatched: {unmatched_error_rate}")
  # print(f"error rate in beat backwards: {backwards_error_rate}")
  type_beat_errors_dict = {'beat_type_unmatched_error': unmatched_error_rate, 'beat_backwards_error': backwards_error_rate}
  return type_beat_errors_dict

def check_syntax_errors_in_inference_for_cp(generated_output, feature_list):
  generated_output = generated_output.squeeze(0)
  type_idx = feature_list.index('type')
  beat_idx = feature_list.index('beat')
  pitch_idx = feature_list.index('pitch')
  duration_idx = feature_list.index('duration')
  last_note = 1
  beat_type_unmatched_error_list = []
  num_unmatched_errors = 0
  beat_backwards_error_list = []
  num_backwards_errors = 0
  for token in generated_output:
    if token[type_idx].item() == 2: # Metrical
      if token[pitch_idx].item() != 0 or token[duration_idx].item() != 0:
        num_unmatched_errors += 1
        beat_type_unmatched_error_list.append(token)
      if token[beat_idx].item() == 1: # new bar
        last_note = 1 # last note will be updated in the next token
      elif token[beat_idx].item() != 0 and token[beat_idx].item() <= last_note:
        num_backwards_errors += 1
        last_note = token[beat_idx].item() # update last note
        beat_backwards_error_list.append([last_note, token])
      else:
        last_note = token[beat_idx].item() # update last note
    if token[type_idx].item() == 3: # Note
      if token[beat_idx].item() != 0:
        num_unmatched_errors += 1
        beat_type_unmatched_error_list.append(token)
  unmatched_error_rate = num_unmatched_errors / len(generated_output)
  backwards_error_rate = num_backwards_errors / len(generated_output)
  # print(f"error in beat_type unmatched: {beat_type_unmatched_error_list}")
  # print(f"error in beat backwards: {beat_backwards_error_list}")
  # print(f"error rate in beat_type unmatched: {unmatched_error_rate}")
  # print(f"error rate in beat backwards: {backwards_error_rate}")
  type_beat_errors_dict = {'beat_type_unmatched_error': unmatched_error_rate, 'beat_backwards_error': backwards_error_rate}
  return type_beat_errors_dict

def check_syntax_errors_in_inference_for_remi(generated_output, vocab):
  generated_output = generated_output.squeeze(0)
  # to check duration errors
  beat_mask = vocab.total_mask['beat'].to(generated_output.device)
  beat_mask_for_target = beat_mask[generated_output]
  beat_target = generated_output * beat_mask_for_target
  bar_mask = vocab.total_mask['type'].to(generated_output.device)
  bar_mask_for_target = bar_mask[generated_output]
  bar_target = (generated_output+1) * bar_mask_for_target # as bar token in 0 in remi vocab, we add 1 to bar token
  target = beat_target + bar_target
  target = target[target!=0]
  # collect beats in between bars(idx=1)
  num_backwards_errors = 0
  collected_beats = []
  total_beats = 0
  for token in target:
    if token == 1 or 3 <= token <= 26: # Bar_None, or Bar_time_signature
      collected_beats_tensor = torch.tensor(collected_beats)
      diff = torch.diff(collected_beats_tensor)
      num_error_beats = torch.where(diff<=0)[0].shape[0]
      num_backwards_errors += num_error_beats
      collected_beats = []
    else:
      collected_beats.append(token.item())
      total_beats += 1
  if total_beats != 0:
    backwards_error_rate = num_backwards_errors / total_beats
  else:
    backwards_error_rate = 0
  # print(f"error rate in beat backwards: {backwards_error_rate}")
  return {'beat_backwards_error': backwards_error_rate}

def type_beat_errors_in_validation_nb(beat_prob, answer_type, input_beat, mask):
  '''
  beat_prob: b x t x vocab_size
  answer_type: type features in shifted_target, b x t
  input_beat: beat features in tgt, b x t
  mask: b x t, value is 1 if valid, 0 if invalid
  '''
  bool_mask = mask.bool().flatten() # (b*t)
  pred_beat_idx = torch.argmax(beat_prob, dim=-1).flatten() # (b*t)
  valid_pred_beat_idx = pred_beat_idx[bool_mask] # valid beat_idx
  answer_type = answer_type.flatten() # (b*t)
  valid_type_input = answer_type[bool_mask] # valid answer_type
  type_beat_list = []
  for i in range(len(valid_pred_beat_idx)):
    type_beat_list.append((valid_type_input[i].item(), valid_pred_beat_idx[i].item())) # type, beat
  input_beat = input_beat.flatten()
  valid_input_beat = input_beat[bool_mask]
  
  last_note = 1
  num_unmatched_errors = 0
  num_backwards_errors = 0
  for type_beat, input_beat_idx in zip(type_beat_list, valid_input_beat):
    # update last note
    if input_beat_idx.item() >= 1: # beat
      last_note = input_beat_idx.item()
    # check errors
    # if type_beat[0] == 3: # same bar, same beat
    #   if type_beat[1] != 1:
    #     num_unmatched_errors += 1
    if type_beat[0] == 4: # same bar, new beat
      if type_beat[1] == 0 or type_beat[1] == 1:
        num_unmatched_errors += 1
      if type_beat[1] <= last_note:
        num_backwards_errors += 1
    elif type_beat[0] >= 5: # new bar, new beat
      if type_beat[1] == 0:
        num_unmatched_errors += 1
  return len(type_beat_list), num_unmatched_errors, num_backwards_errors

def type_beat_errors_in_validation_cp(beat_prob, answer_type, input_beat, mask):
  bool_mask = mask.bool().flatten() # (b*t)
  beat_idx = torch.argmax(beat_prob, dim=-1).flatten() # (b*t)
  valid_beat_idx = beat_idx[bool_mask] # valid beat_idx
  answer_type = answer_type.flatten() # (b*t)
  valid_type_input = answer_type[bool_mask] # valid answer_type
  type_beat_list = []
  for i in range(len(valid_beat_idx)):
    type_beat_list.append((valid_type_input[i].item(), valid_beat_idx[i].item())) # type, beat
  input_beat = input_beat.flatten()
  valid_input_beat = input_beat[bool_mask]
  
  last_note = 1
  num_unmatched_errors = 0
  num_backwards_errors = 0
  for type_beat, input_beat_idx in zip(type_beat_list, valid_input_beat):
    # update last note
    if input_beat_idx.item() == 1: # bar
      last_note = 1
    elif input_beat_idx.item() >= 2: # new beat
      last_note = input_beat_idx.item()
    # check errors
    if type_beat[0] == 2: # Metrical
      if type_beat[1] == 0: # ignore
        num_unmatched_errors += 1
      elif type_beat[1] >= 2: # new beat
        if type_beat[1] <= last_note:
          num_backwards_errors += 1
    elif type_beat[0] == 3: # Note
      if type_beat[1] != 0:
        num_unmatched_errors += 1
  return len(type_beat_list), num_unmatched_errors, num_backwards_errors

  
def get_beat_difference_metric(prob_dict, arranged_prob_dict, mask):
  orign_beat_prob = prob_dict['beat'] # b x t x vocab_size
  arranged_beat_prob = arranged_prob_dict['beat'] # b x t x vocab_size

  # calculate similarity between original beat prob and arranged beat prob
  origin_beat_token = torch.argmax(orign_beat_prob, dim=-1) * mask # b x t
  arranged_beat_token = torch.argmax(arranged_beat_prob, dim=-1) * mask # b x t
  num_same_beat = torch.sum(origin_beat_token == arranged_beat_token) - torch.sum(mask==0)
  num_beat = torch.sum(mask==1)
  beat_sim = (num_same_beat / num_beat).item() # scalar

  # apply mask, shape of mask: b x t
  orign_beat_prob = orign_beat_prob * mask.unsqueeze(-1) # b x t x vocab_size
  arranged_beat_prob = arranged_beat_prob * mask.unsqueeze(-1)

  # calculate cosine similarity between original beat prob and arranged beat prob
  orign_beat_prob = orign_beat_prob.flatten(0,1) # (b*t) x vocab_size
  arranged_beat_prob = arranged_beat_prob.flatten(0,1) # (b*t) x vocab_size
  cos = torch.nn.CosineSimilarity(dim=-1, eps=1e-6)
  beat_cos_sim = cos(orign_beat_prob, arranged_beat_prob) # (b*t)
  # exclude invalid tokens, zero padding tokens
  beat_cos_sim = beat_cos_sim[mask.flatten().bool()] # num_valid_tokens
  beat_cos_sim = torch.mean(beat_cos_sim).item() # scalar
  return {'beat_cos_sim': beat_cos_sim, 'beat_sim': beat_sim}

def get_gini_coefficient(generated_output):
  if len(generated_output.shape) == 3:
    generated_output = generated_output.squeeze(0).tolist()
    gen_list = [tuple(x) for x in generated_output]
  else:
    gen_list = generated_output.squeeze(0).tolist()
  counts = Counter(gen_list).values()
  sorted_counts = sorted(counts)
  n = len(sorted_counts)
  cumulative_counts = np.cumsum(sorted_counts)
  cumulative_proportion = cumulative_counts / cumulative_counts[-1]

  lorenz_area = sum(cumulative_proportion[:-1]) / n  # Exclude the last element
  equality_area = 0.5  # The area under line of perfect equality

  gini = (equality_area - lorenz_area) / equality_area
  return gini