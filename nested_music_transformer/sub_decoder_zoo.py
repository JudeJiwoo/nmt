import torch
import torch.nn as nn

from x_transformers import Decoder

from .transformer_utils import MultiEmbedding, RVQMultiEmbedding
from .sub_decoder_utils import *
from .sampling_utils import sample

from data_representation.vocab_utils import LangTokenVocab

class SingleProjection(nn.Module):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    '''
    This sub-decoder is used for REMI based models
    '''
    super().__init__()
    vocab_size = vocab.get_vocab_size()
    self.proj = nn.Linear(dim, vocab_size)
    
  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=1):
    hidden_vec = input_dict['hidden_vec']
    target = input_dict['target']
    # ---- Generate(Inference) ---- #
    if target is None:
      logits = self.proj(hidden_vec[:, -1:])
      sampled_token = sample(logits, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
      return logits, sampled_token
    # ---- Training ---- #
    logits = self.proj(hidden_vec)
    return logits

class SubDecoderClass(nn.Module):
  def __init__(
    self, 
    prediction_order:list, 
    vocab:LangTokenVocab, 
    sub_decoder_depth:int, 
    dim:int, 
    heads:int, 
    dropout:float, 
    sub_decoder_enricher_use:bool
  ):
    super().__init__()
    '''
    This is the base class for all sub-decoders
    '''
    self.prediction_order = prediction_order
    self.vocab = vocab
    self.vocab_size = vocab.get_vocab_size()
    # make layers
    self._make_emb_layer(vocab, dim)
    self._make_projection_layer(vocab, dim)
    self._make_nonlinear_layer()

  @property
  def device(self):
    return next(self.parameters()).device

  def _make_emb_layer(self, vocab, dim):
    self.emb_layer = MultiEmbedding(
      vocab=vocab,
      dim_model=dim
    )

  def _make_projection_layer(self, vocab, dim):
    vocab_sizes = vocab.get_vocab_size()
    self.hidden2logit = nn.ModuleDict({
      f"layer_{key}": nn.Linear(dim, size) for key, size in vocab_sizes.items()
      })

  def _make_nonlinear_layer(self):
    pass
  
class FeedForward(SubDecoderClass):
  def __init__(
      self, 
      prediction_order:list, 
      vocab:LangTokenVocab, 
      sub_decoder_depth:int, 
      dim:int, 
      heads:int, 
      dropout:float, 
      sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def _make_projection_layer(self, vocab, dim):
    vocab_sizes = vocab.get_vocab_size()
    self.hidden2logit = nn.ModuleDict({
      f"layer_{key}": nn.Linear(dim, size) for key, size in vocab_sizes.items()
      })
    self.catvec2hidden = nn.ModuleDict({
      f"layer_{key}": nn.Linear(dim+dim, dim) for key, _ in vocab_sizes.items()
      })

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec']
    target = input_dict['target']

    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      for feature in self.prediction_order:
        if isinstance(feature, str):
          logit = self.hidden2logit[f"layer_{feature}"](hidden_vec)
          logits_dict[feature] = logit
          sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
          sampled_token_dict[feature] = sampled_token
          feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token) # B x T x emb_size
          catvec = torch.cat([hidden_vec, feature_emb.unsqueeze(0)], dim=-1)
          hidden_vec = self.catvec2hidden[f"layer_{feature}"](catvec)
        else:
          assert feature == self.prediction_order[-1], "Parallel prediction should be the last feature"
          for par_feature in feature:
            logit = self.hidden2logit[f"layer_{par_feature}"](hidden_vec)
            logits_dict[par_feature] = logit
            sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
            sampled_token_dict[par_feature] = sampled_token
      return logits_dict, sampled_token_dict

    # ---- Training ---- #
    for feature in self.prediction_order:
      if isinstance(feature, str):
        logit = self.hidden2logit[f"layer_{feature}"](hidden_vec)
        logits_dict[feature] = logit
        feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., self.vocab.feature_list.index(feature)]) # B x T x emb_size
        catvec = torch.cat([hidden_vec, feature_emb], dim=-1)
        hidden_vec = self.catvec2hidden[f"layer_{feature}"](catvec)
      else:
        assert feature == self.prediction_order[-1], "Parallel prediction should be the last feature"
        for par_feature in feature:
          logit = self.hidden2logit[f"layer_{par_feature}"](hidden_vec)
          logits_dict[par_feature] = logit
    return logits_dict

class Parallel(SubDecoderClass):
  def __init__(
      self, 
      prediction_order:list, 
      vocab:LangTokenVocab, 
      sub_decoder_depth:int, 
      dim:int, 
      heads:int, 
      dropout:float, 
      sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec']
    target = input_dict['target']

    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      for feature in self.prediction_order:
        logit = self.hidden2logit[f"layer_{feature}"](hidden_vec) # B x T x vocab_size
        logits_dict[feature] = logit
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    for feature in self.prediction_order:
      logit = self.hidden2logit[f"layer_{feature}"](hidden_vec)
      logits_dict[feature] = logit
    return logits_dict

class RNN(SubDecoderClass):
  def __init__(
      self, 
      prediction_order:list, 
      vocab:LangTokenVocab, 
      sub_decoder_depth:int, 
      dim:int, 
      heads:int, 
      dropout:float, 
      sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}

    self.pos_enc = nn.Embedding(len(prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)

    self.decoding_rnn = nn.GRU(
                      input_size=dim,
                      hidden_size=dim,
                      num_layers=sub_decoder_depth,
                      dropout=dropout,
                      batch_first=True)

  def _apply_pos_enc(self, tgt, apply_type='last'):
    if apply_type == 'all':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      tgt_pos = tgt + self.pos_enc(pos.long())
    elif apply_type == 'last':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      pos_emb = self.pos_enc(pos.long())
      # zero out the pos_emb except for the last token
      pos_emb[:, :-1, :] = 0
      tgt_pos = tgt + pos_emb
    return tgt_pos

  def _prepare_token_embedding_for_teacher_forcing(self, input_seq, target):
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq = torch.cat([input_seq, feature_emb_reshape], dim=1) 
    return input_seq

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x 7
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], -1)).unsqueeze(1) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape # (B*T) x 1 x d_model
    
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      h_0 = input_seq[:, 0, :].unsqueeze(0) # 1 x (B*T) x d_model
      input_seq = self._apply_pos_enc(input_seq, apply_type='all') # (B*T) x 1 x d_model
      for idx, feature in enumerate(self.prediction_order):
        input_seq, _ = self.decoding_rnn(input_seq, h_0) # input_seq: (B*T) x (idx+1) x hidden_size, h_n: num_layers x (B*T) x hidden_size
        logit = self.hidden2logit[f"layer_{feature}"](input_seq[:, -1, :]) # (B*T) x vocab_size
        logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
        logits_dict[feature] = logit
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token) # B x T x emb_size
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        input_seq = torch.cat([input_seq, feature_emb_reshape], dim=1) # (B*T) x (idx+2) x d_model
        input_seq = self._apply_pos_enc(input_seq, apply_type='last') # (B*T) x (idx+2) x d_model
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    input_seq = self._prepare_token_embedding_for_teacher_forcing(input_seq, target) # (B*T) x len(prediction_order) x d_model
    # initial hidden state has no positional encoding
    h0 = input_seq[:, 0, :].unsqueeze(0) # 1 x (B*T) x d_model 
    h0 = h0.contiguous()
    # apply positional encoding
    input_seq = self._apply_pos_enc(input_seq, apply_type='all') # (B*T) x len(prediction_order) x d_model
    # get output using rnn
    output, _ = self.decoding_rnn(input_seq, h0) # (B*T) x len(prediction_order) x d_model
    output = output.reshape((hidden_vec.shape[0], hidden_vec.shape[1], len(self.prediction_order), -1)) # B x T x len(prediction_order) x d_model
    for idx, feature in enumerate(self.prediction_order):
      logit = self.hidden2logit[f"layer_{feature}"](output[:, :, idx, :]) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class SelfAttention(SubDecoderClass):
  def __init__(
      self, 
      prediction_order:list, 
      vocab:LangTokenVocab, 
      sub_decoder_depth:int, 
      dim:int, 
      heads:int, 
      dropout:float, 
      sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(1 + len(prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)
    
    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    
    window_size = 1 # number of previous hidden vector of tokens from the main decoder
    causal_mask = generate_causality_mask_on_window(size=window_size + len(prediction_order), window_size=window_size)
    self.register_buffer('causal_mask', causal_mask)

    self.transformer_decoder = Decoder(
                                    dim = dim,
                                    depth = sub_decoder_depth,
                                    heads = heads,
                                    attn_dropout = dropout,
                                    ff_dropout = dropout,
                                    attn_flash = True)
    # add final dropout
    print('Applying Xavier Uniform Init to x-transformer following torch.Transformer')
    self._apply_xavier_init()
    print('Adding dropout after feedforward layer in x-transformer')
    self._add_dropout_after_ff(dropout)
    print('Adding dropout after attention layer in x-transformer')
    self._add_dropout_after_attn(dropout)

  def _add_dropout_after_attn(self, dropout):
    for layer in self.transformer_decoder.layers:
      if 'Attention' in str(type(layer[1])): 
        if isinstance(layer[1].to_out, nn.Sequential): # if GLU
          layer[1].to_out.append(nn.Dropout(dropout))
        elif isinstance(layer[1].to_out, nn.Linear): # if simple linear
          layer[1].to_out = nn.Sequential(layer[1].to_out, nn.Dropout(dropout))
        else:
          raise ValueError('to_out should be either nn.Sequential or nn.Linear')

  def _add_dropout_after_ff(self, dropout):
    for layer in self.transformer_decoder.layers:
      if 'FeedForward' in str(type(layer[1])):
        layer[1].ff.append(nn.Dropout(dropout))

  def _apply_xavier_init(self):
    for name, param in self.transformer_decoder.named_parameters():
      if 'to_q' in name or 'to_k' in name or 'to_v' in name:
          torch.nn.init.xavier_uniform_(param, gain=0.5**0.5)

  def _apply_pos_enc(self, tgt, apply_type='last'):
    if apply_type == 'all':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      tgt_pos = tgt + self.pos_enc(pos.long())
    elif apply_type == 'last':
      pos = torch.arange(tgt.shape[1]).to(tgt.device)
      pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1)
      pos_emb = self.pos_enc(pos.long()) # (B*T) x (window_size + BOS + num_features-1) x dim
      # zero out the pos_emb except for the last token
      pos_emb[:, :-1, :] = 0
      tgt_pos = tgt + pos_emb
    return tgt_pos

  def _prepare_input_seq_list(self, hidden_vec_reshape, target=None):
    input_seq_list = []
    input_seq_list.append(hidden_vec_reshape)
    BOS_emb = self.sub_decoder_BOS_emb.unsqueeze(0).repeat(hidden_vec_reshape.shape[0], 1, 1) # (B*T) x 1 x d_model
    if target is None:
      input_seq_list.append(BOS_emb[-1:, :, :])
    else: # training
      input_seq_list.append(BOS_emb)
    return input_seq_list

  def _prepare_token_embedding_for_teacher_forcing(self, input_seq_list, target):
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS + 7) x d_model
    return memory_tensor

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x 8
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq_list = self._prepare_input_seq_list(hidden_vec_reshape, target)
    
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      input_seq_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS) x d_model
      pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS) x d_model
      for idx, feature in enumerate(self.prediction_order):
        output = self.transformer_decoder(pos_target_tensor)
        logit = self.hidden2logit[f"layer_{feature}"](output[:, -1:])
        logits_dict[feature] = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        input_seq_list.append(feature_emb_reshape)
        input_seq_tensor = torch.cat(input_seq_list, dim=1)
        pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='last')
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    # preparing for training
    input_seq_tensor = self._prepare_token_embedding_for_teacher_forcing(input_seq_list, target) # (B*T) x (window_size + BOS + num_features-1) x d_model
    pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS + num_features-1) x d_model
    # get output using self-attention
    output = self.transformer_decoder(pos_target_tensor)
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class SelfAttentionUniAudio(SelfAttention):
  def __init__(
      self, 
      prediction_order, 
      vocab, 
      sub_decoder_depth, 
      dim, 
      heads, 
      dropout,
      sub_decoder_enricher_use
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    '''
    Uniaudio version of self-attention sub-decoder
    '''

  def _prepare_token_embedding_for_teacher_forcing(self, hidden_vec_reshape, target):
    input_seq_list = []
    # append zero vector
    input_seq_list.append(torch.zeros(hidden_vec_reshape.shape[0], 1, hidden_vec_reshape.shape[2]).to(self.device))
    for feature in self.prediction_order[:-1]:
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      input_seq_list.append(feature_emb_reshape)

    feature_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x num_sub-tokens x d_model
    # Ensure hidden_vec_reshape and feature_tensor have the same shape
    assert hidden_vec_reshape.shape == feature_tensor.shape, f"Shapes of hidden_vec_reshape and feature_tensor do not match: {hidden_vec_reshape.shape} vs {feature_tensor.shape}"
    # Sum hidden_vec_reshape and feature_tensor in the last dimension
    memory_tensor = hidden_vec_reshape + feature_tensor
    return memory_tensor
  
  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target'] # B x T x num_sub-tokens
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    hidden_vec_reshape = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x num_sub-tokens x d_model
    
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      # input_seq_tensor = torch.cat(input_seq_list, dim=1) # (B*T) x (window_size + BOS) x d_model
      pos_target_tensor = self._apply_pos_enc(hidden_vec_reshape, apply_type='all') # (B*T) x (window_size + BOS) x d_model
      # pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS) x d_model
      for idx, feature in enumerate(self.prediction_order):
        output = self.transformer_decoder(pos_target_tensor)
        logit = self.hidden2logit[f"layer_{feature}"](output[:, -1:])
        logits_dict[feature] = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        pos_target_tensor = torch.cat([pos_target_tensor[:, :idx+1, :], feature_emb_reshape + pos_target_tensor[:, idx+1:idx+2, :], pos_target_tensor[:, idx+2:, :]], dim=1)

      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    # preparing for training
    input_seq_tensor = self._prepare_token_embedding_for_teacher_forcing(hidden_vec_reshape, target) # (B*T) x (window_size + BOS + num_features-1) x d_model
    pos_target_tensor = self._apply_pos_enc(input_seq_tensor, apply_type='all') # (B*T) x (window_size + BOS + num_features-1) x d_model
    # get output using self-attention
    output = self.transformer_decoder(pos_target_tensor)
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class CrossAttention(SubDecoderClass):
  def __init__(
      self, 
      prediction_order:list, 
      vocab:LangTokenVocab, 
      sub_decoder_depth:int, 
      dim:int, 
      heads:int, 
      dropout:float, 
      sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)
    self.sub_decoder_enricher_use = sub_decoder_enricher_use
    self.feature_order_in_output = {key: (idx-len(prediction_order)) for idx, key in enumerate(prediction_order)}
    
    self.pos_enc = nn.Embedding(len(self.prediction_order), dim)
    nn.init.zeros_(self.pos_enc.weight)

    self.sub_decoder_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    if sub_decoder_enricher_use:
      self.enricher_BOS_emb = nn.Parameter(torch.zeros(dim), requires_grad=True)
    causal_mask = generate_SA_mask(len(prediction_order))
    causl_ca_mask = generate_CA_mask(len(prediction_order), len(prediction_order)).to(self.device)
    self.register_buffer('causal_mask', causal_mask)
    self.register_buffer('causal_ca_mask', causl_ca_mask)

    self.sub_decoder_layers = nn.Sequential(DecoderLayer(dim=dim, num_heads=heads, dropout=dropout))
    if sub_decoder_enricher_use:
      self.feature_enricher_layers = nn.Sequential(FeatureEnricher(dim=dim, num_heads=heads, dropout=dropout))

  def _apply_window_on_hidden_vec(self, hidden_vec):
    BOS_emb = self.enricher_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    # window_size = self.net_param.decoding_attention.decout_window_size
    window_size = 1
    zero_vec = torch.zeros((hidden_vec.shape[0], window_size-1, hidden_vec.shape[2])).to(self.device) # B x (window_size-1) x d_model
    cat_hidden_vec = torch.cat([zero_vec, hidden_vec], dim=1) # B x (window_size-1+T) x d_model
    new_hidden_vec = cat_hidden_vec.unfold(1, window_size, 1).transpose(2, 3) # B x T x window_size x d_model
    new_hidden_vec = new_hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], window_size, -1)) # (B*T) x window_size x d_model
    new_hidden_vec = torch.cat([BOS_emb, new_hidden_vec], dim=1) # (B*T) x (window_size+1) x d_model
    return new_hidden_vec

  def _apply_pos_enc(self, tgt):
    pos = torch.arange(tgt.shape[1]).to(tgt.device) # 8
    pos = pos.unsqueeze(0).repeat(tgt.shape[0], 1) # (B*T) x 8
    tgt_pos = tgt + self.pos_enc(pos.long()) # (B*T) x 8 x d_model
    return tgt_pos

  def _prepare_token_embedding_for_teacher_forcing(self, memory_list, target):
    for _, feature in enumerate(self.prediction_order[:-1]):
      feature_idx = self.vocab.feature_list.index(feature)
      feature_emb = self.emb_layer.get_emb_by_key(feature, target[..., feature_idx]) # B x T x emb_size
      feature_emb_reshape = feature_emb.reshape((feature_emb.shape[0]*feature_emb.shape[1], 1, -1)) # (B*T) x 1 x emb_size
      memory_list.append(feature_emb_reshape)
    memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + num_features-1) x d_model
    return memory_tensor

  def _prepare_memory_list(self, hidden_vec, target=None):
    memory_list = [] # used for key and value in cross attention
    BOS_emb = self.sub_decoder_BOS_emb.reshape(1,1,-1).repeat(hidden_vec.shape[0]*hidden_vec.shape[1], 1, 1) # (B*T) x 1 x d_model
    if target is not None: # training
      memory_list.append(BOS_emb)
    else: # inference
      memory_list.append(BOS_emb[-1:, :, :])
    return memory_list

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    logits_dict = {}
    hidden_vec = input_dict['hidden_vec'] # B x T x d_model
    target = input_dict['target']

    # apply window on hidden_vec for enricher
    if self.sub_decoder_enricher_use:
      window_applied_hidden_vec = self._apply_window_on_hidden_vec(hidden_vec) # (B*T) x window_size x d_model
    hidden_vec_reshape = hidden_vec.reshape((hidden_vec.shape[0]*hidden_vec.shape[1], 1, -1)) # (B*T) x 1 x d_model
    input_seq = hidden_vec_reshape.repeat(1, len(self.prediction_order), 1) # (B*T) x 8 x d_model
    input_seq_pos = self._apply_pos_enc(input_seq)
    # prepare memory
    memory_list = self._prepare_memory_list(hidden_vec=hidden_vec, target=target)
    # ---- Generate(Inference) ---- #
    if target is None:
      sampled_token_dict = {}
      memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x 1 x d_model
      for idx, feature in enumerate(self.prediction_order):
        feature_pos = self.feature_order_in_output[feature]
        if self.sub_decoder_enricher_use:
          input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec[-1:]}
          input_dict = self.feature_enricher_layers(input_dict)
          memory_tensor = input_dict['input_seq']
        CA_attn_mask = generate_CA_mask(input_seq_pos.shape[1], memory_tensor.shape[1]).to(self.device)
        input_dict = {'input_seq': input_seq_pos[-1:], 'memory': memory_tensor, 'memory_mask': CA_attn_mask}
        input_dict = self.sub_decoder_layers(input_dict)
        attn_output = input_dict['input_seq']
        logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
        logit = logit.reshape((1, 1, -1)) # 1 x 1 x vocab_size
        logits_dict[feature] = logit
        sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
        sampled_token_dict[feature] = sampled_token
        if idx == len(self.prediction_order)-1:
          return logits_dict, sampled_token_dict
        feature_emb = self.emb_layer.get_emb_by_key(feature, sampled_token)
        feature_emb_reshape = feature_emb.reshape((1, 1, -1)) # (B*T) x 1 x emb_size
        memory_list.append(feature_emb_reshape)
        memory_tensor = torch.cat(memory_list, dim=1) # (B*T) x (BOS + idx+1) x d_model
      return logits_dict, sampled_token_dict
    
    # ---- Training ---- #
    memory_tensor = self._prepare_token_embedding_for_teacher_forcing(memory_list, target) # (B*T) x (BOS + num_features-1) x d_model
    # apply feature enricher to memory
    if self.sub_decoder_enricher_use:
      input_dict = {'input_seq': memory_tensor, 'memory': window_applied_hidden_vec}
      input_dict = self.feature_enricher_layers(input_dict)
      memory_tensor = input_dict['input_seq'] # (B*T) x num_features x d_model
    # implement sub decoder cross attention
    input_dict = {'input_seq': input_seq_pos, 'memory': memory_tensor, 'memory_mask': self.causal_ca_mask}
    input_dict = self.sub_decoder_layers(input_dict)
    attn_output = input_dict['input_seq'] # (B*T) x num_features x d_model
    # get prob
    for idx, feature in enumerate(self.prediction_order):
      feature_pos = self.feature_order_in_output[feature]
      logit = self.hidden2logit[f"layer_{feature}"](attn_output[:, feature_pos, :])
      logit = logit.reshape((hidden_vec.shape[0], hidden_vec.shape[1], -1)) # B x T x vocab_size
      logits_dict[feature] = logit
    return logits_dict

class Flatten4Encodec(SubDecoderClass):
  def __init__(
      self, 
      prediction_order:list, 
      vocab:LangTokenVocab, 
      sub_decoder_depth:int, 
      dim:int, 
      heads:int, 
      dropout:float, 
      sub_decoder_enricher_use:bool
  ):
    super().__init__(prediction_order, vocab, sub_decoder_depth, dim, heads, dropout, sub_decoder_enricher_use)

  def forward(self, input_dict, sampling_method=None, threshold=None, temperature=None):
    hidden_vec = input_dict['hidden_vec']

    # ---- Training ---- #
    logits_tensor = torch.zeros(hidden_vec.shape[0], hidden_vec.shape[1], 2049).to(self.device)
    for idx, feature_type in enumerate(self.prediction_order):
      # ::4 means that we only use the first token in each 4 tokens
      # so the chosen tokens will be: 0, 4, 8, 12, ...
      # 1::4 means that we only use the second token in each 4 tokens
      # so the chosen tokens will be: 1, 5, 9, 13, ...
      separated_hidden_vec = hidden_vec[:, idx::4, :]
      logit = self.hidden2logit[f"layer_{feature_type}"](separated_hidden_vec)
      logits_tensor[:, idx::4, :] = logit
      # prob_dict[feature_type] = prob
    return logits_tensor
  
  def run_one_step(self, input_dict, sampling_method=None, threshold=None, temperature=None, feature_type=None):
    # ---- Generate(Inference) ---- #
    hidden_vec = input_dict['hidden_vec']
    logit = self.hidden2logit[f"layer_{feature_type}"](hidden_vec[:, -1:])
    sampled_token = sample(logit, sampling_method=sampling_method, threshold=threshold, temperature=temperature)
    return logit, sampled_token


