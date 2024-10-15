import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.auto import tqdm

from . import transformer_utils
from . import sub_decoder_zoo
from x_transformers.x_transformers import LayerIntermediates, AbsolutePositionalEmbedding

class DoubleSequentialTransformerWrapper(nn.Module):
  def __init__(
      self, 
      *, 
      vocab, 
      input_length,
      prediction_order,
      input_embedder_name,
      main_decoder_name,
      sub_decoder_name,
      sub_decoder_depth,
      sub_decoder_enricher_use,
      dim,  
      heads, 
      depth, 
      dropout,
  ):
    super().__init__()
    self.vocab = vocab
    self.vocab_size = vocab.get_vocab_size()
    self.start_token = vocab.sos_token if hasattr(vocab, 'sos_token') else None
    self.end_token = vocab.eos_token if hasattr(vocab, 'eos_token') else None
    self.input_length = input_length
    self.prediction_order = prediction_order
    self._get_input_embedder(input_embedder_name, vocab, dropout, dim)
    self._get_main_decoder(main_decoder_name, input_length, dim, heads, depth, dropout)
    self._get_sub_decoder(sub_decoder_name, prediction_order, vocab, sub_decoder_depth, sub_decoder_enricher_use, dim, heads, dropout)

  def _get_input_embedder(self, input_embedder_name, vocab, dropout, dim):
    self.emb_dropout = nn.Dropout(dropout)
    self.input_embedder = getattr(transformer_utils, input_embedder_name)(
      vocab=vocab,
      dim_model=dim
    )

  def _get_main_decoder(self, main_decoder_name, input_length, dim, heads, depth, dropout):
    self.pos_enc = AbsolutePositionalEmbedding(dim, input_length)
    self.main_norm = nn.LayerNorm(dim)
    self.main_decoder = getattr(transformer_utils, main_decoder_name)(
      dim=dim,
      depth=depth,
      heads=heads,
      dropout=dropout
    )
  
  def _get_sub_decoder(self, sub_decoder_name, prediction_order, vocab, sub_decoder_depth, sub_decoder_enricher_use, dim, heads, dropout):
    self.sub_decoder = getattr(sub_decoder_zoo, sub_decoder_name)(
      prediction_order=prediction_order,
      vocab=vocab,
      dim=dim,
      sub_decoder_depth=sub_decoder_depth,
      heads=heads,
      dropout=dropout,
      sub_decoder_enricher_use=sub_decoder_enricher_use
    )

  @property
  def device(self):
    return next(self.parameters()).device

  def forward(self, input_seq:torch.Tensor, target:torch.Tensor):
    embedding = self.input_embedder(input_seq) + self.pos_enc(input_seq)
    embedding = self.emb_dropout(embedding)
    hidden_vec = self.main_decoder(embedding)
    hidden_vec = self.main_norm(hidden_vec)
    input_dict = {'hidden_vec':hidden_vec, 'input_seq': input_seq, 'target': target}
    logits = self.sub_decoder(input_dict)
    return logits

class DoubleSequentialAutoregressiveWrapper(nn.Module):
  def __init__(self, net:DoubleSequentialTransformerWrapper):
    super().__init__()
    self.net = net

  def forward(self, input_seq:torch.Tensor, target:torch.Tensor):
    return self.net(input_seq, target)
  
  def _prepare_inference(self, start_token, manual_seed, condition=None):
    if manual_seed > 0:
      torch.manual_seed(manual_seed)
    total_out = []
    if condition is None:
      total_out.extend(start_token)
    else: 
      num_target_measures = 4
      if self.net.vocab.encoding_scheme == 'remi':
        type_boundaries = self.net.vocab.remi_vocab_boundaries_by_key['type']
        measure_bool = (condition == 0) | ((3 <= condition) & (condition < type_boundaries[1])) # between Bar_ts_start and Bar_ts_end 
        conditional_input_len = torch.where(measure_bool)[0][num_target_measures].item()
      elif self.net.vocab.encoding_scheme == 'cp':
        measure_bool = (condition[:,1] == 1) # Bar tokens
        conditional_input_len = torch.where(measure_bool)[0][num_target_measures].item()
      elif self.net.vocab.encoding_scheme == 'nb':
        measure_bool = (condition[:,0] == 2) | (condition[:,0] >= 5) # Emptybar or NNN_ts
        conditional_input_len = torch.where(measure_bool)[0][num_target_measures].item()
      if conditional_input_len == 0:
        conditional_input_len = 50
      selected_tokens = condition[:conditional_input_len].tolist()
      total_out.extend(selected_tokens)
    total_out = torch.LongTensor(total_out).unsqueeze(0).to(self.net.device)
    return total_out

  def _run_one_step(self, input_seq, cache=None, sampling_method=None, threshold=None, temperature=1):
    embedding = self.net.input_embedder(input_seq) + self.net.pos_enc(input_seq)
    embedding = self.net.emb_dropout(embedding)
    hidden_vec, intermidiates = self.net.main_decoder(embedding, cache) # B x T x d_model
    hidden_vec = self.net.main_norm(hidden_vec)
    hidden_vec = hidden_vec[:, -1:] # B x 1 x d_model
    input_dict = {'hidden_vec':hidden_vec, 'input_seq': input_seq, 'target': None}
    logits, sampled_token = self.net.sub_decoder(input_dict, sampling_method, threshold, temperature)
    return logits, sampled_token, intermidiates

  def _update_total_out(self, total_out, sampled_token):
    if self.net.vocab.encoding_scheme == 'remi':
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0)], dim=-1)
    else:
      sampled_token_list = []
      for key in self.net.vocab.feature_list:
        sampled_token_list.append(sampled_token[key])
      sampled_token = torch.cat(sampled_token_list, dim=-1) # B(1) x num_features
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0).unsqueeze(0)], dim=1)
    return total_out, sampled_token

  @torch.inference_mode()
  def generate(self, manual_seed, max_seq_len, condition=None, sampling_method=None, threshold=None, temperature=1):
    total_out = self._prepare_inference(self.net.start_token, manual_seed, condition)
    if condition is not None:
      _, _, cache = self._run_one_step(total_out[:, -self.net.input_length:], cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    else:
      cache = LayerIntermediates()
    while total_out.shape[1] < max_seq_len:
      input_tensor = total_out[:, -self.net.input_length:]
      _, sampled_token, cache = self._run_one_step(input_tensor, cache=cache, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      for inter in cache.attn_intermediates:
        inter.cached_kv = [t[..., -(self.net.input_length - 1):, :] for t in inter.cached_kv] # B x num_heads x T x d_head
      total_out, sampled_token = self._update_total_out(total_out, sampled_token)
      if sampled_token.tolist() == self.net.end_token[0]:
        break
    return total_out

class DoubleSequentialTransformer(nn.Module):
  def __init__(
      self, 
      vocab, 
      input_length, 
      prediction_order, 
      input_embedder_name, 
      main_decoder_name, 
      sub_decoder_name, 
      sub_decoder_depth, 
      sub_decoder_enricher_use,
      dim, 
      heads, 
      depth, 
      dropout
  ):
    super().__init__()
    decoder = DoubleSequentialTransformerWrapper(
      vocab=vocab,
      input_length=input_length,
      prediction_order=prediction_order,
      input_embedder_name=input_embedder_name,
      main_decoder_name=main_decoder_name,
      sub_decoder_name=sub_decoder_name,
      sub_decoder_depth=sub_decoder_depth,
      sub_decoder_enricher_use=sub_decoder_enricher_use,
      dim=dim,
      heads=heads,
      depth=depth,
      dropout=dropout
    )
    self.decoder = DoubleSequentialAutoregressiveWrapper(
      net=decoder
    )
  
  def forward(self, input_seq:torch.Tensor, target:torch.Tensor):
    return self.decoder(input_seq, target)
  
  @torch.inference_mode()
  def generate(self, manual_seed, max_seq_len, condition=None, sampling_method=None, threshold=None, temperature=1):
    return self.decoder.generate(manual_seed, max_seq_len, condition, sampling_method, threshold, temperature)

class EncodecDoubleSeqTransformer(DoubleSequentialTransformer):
  def __init__(
      self, 
      vocab, 
      input_length, 
      prediction_order, 
      input_embedder_name, 
      main_decoder_name, 
      sub_decoder_name, 
      sub_decoder_depth, 
      sub_decoder_enricher_use,
      dim, 
      heads, 
      depth, 
      dropout
  ):
    super().__init__(
      vocab=vocab, 
      input_length=input_length, 
      prediction_order=prediction_order, 
      input_embedder_name=input_embedder_name, 
      main_decoder_name=main_decoder_name, 
      sub_decoder_name=sub_decoder_name, 
      sub_decoder_depth=sub_decoder_depth, 
      sub_decoder_enricher_use=sub_decoder_enricher_use,
      dim=dim, 
      heads=heads, 
      depth=depth, 
      dropout=dropout
    )

  def _prepare_inference(self, start_token, manual_seed, condition=None):
    if manual_seed > 0:
      torch.manual_seed(manual_seed)
    total_out = []
    if condition is None:
      total_out.extend(start_token)
    else:
      if self.decoder.net.vocab.encoding_scheme == 'remi':
        selected_tokens = condition[:1500].tolist()
      else:
        selected_tokens = condition[:500].tolist()
      total_out.extend(selected_tokens)
    total_out = torch.LongTensor(total_out).unsqueeze(0).to(self.decoder.net.device)
    return total_out

  def _update_total_out(self, total_out, sampled_token):
    if self.decoder.net.vocab.encoding_scheme == 'remi':
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0)], dim=-1)
    else:
      sampled_token_list = []
      for key in self.decoder.net.vocab.feature_list:
        sampled_token_list.append(sampled_token[key])
      sampled_token = torch.cat(sampled_token_list, dim=-1) # B(1) x num_features
      total_out = torch.cat([total_out, sampled_token.unsqueeze(0).unsqueeze(0)], dim=1)
    return total_out, sampled_token

  def _run_one_step(self, input_seq, cache=None, sampling_method=None, threshold=None, temperature=1):
    embedding = self.decoder.net.input_embedder(input_seq) + self.decoder.net.pos_enc(input_seq)
    embedding = self.decoder.net.emb_dropout(embedding)
    hidden_vec, intermidiates = self.decoder.net.main_decoder(embedding, cache) # B x T x d_model
    hidden_vec = self.decoder.net.main_norm(hidden_vec)
    hidden_vec = hidden_vec[:, -1:] # B x 1 x d_model
    input_dict = {'hidden_vec':hidden_vec, 'input_seq': input_seq, 'target': None}
    if self.decoder.net.vocab.encoding_scheme == 'remi':
      feature_class_idx = (input_seq.shape[1] - 1) % 4
      feature_type = self.decoder.net.vocab.feature_list[feature_class_idx]
      logits, sampled_token = self.decoder.net.sub_decoder.run_one_step(input_dict, sampling_method, threshold, temperature, feature_type)
    else:
      logits, sampled_token = self.decoder.net.sub_decoder(input_dict, sampling_method, threshold, temperature)
    return logits, sampled_token, intermidiates

  @torch.inference_mode()
  def generate(self, manual_seed, max_seq_len, condition=None, sampling_method=None, threshold=None, temperature=1):
    total_out = self._prepare_inference(self.decoder.net.start_token, manual_seed, condition)
    if condition is not None:
      _, _, cache = self._run_one_step(total_out[:, -self.decoder.net.input_length:], cache=LayerIntermediates(), sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    else:
      cache = LayerIntermediates()
    while total_out.shape[1] < max_seq_len:
      input_tensor = total_out[:, -self.decoder.net.input_length:]
      _, sampled_token, cache = self._run_one_step(input_tensor, cache=cache, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      for inter in cache.attn_intermediates:
        inter.cached_kv = [t[..., -(self.decoder.net.input_length - 1):, :] for t in inter.cached_kv] # B x num_heads x T x d_head
      total_out, sampled_token = self._update_total_out(total_out, sampled_token)
      if sampled_token.tolist() == self.decoder.net.end_token[0]:
        break
    return total_out