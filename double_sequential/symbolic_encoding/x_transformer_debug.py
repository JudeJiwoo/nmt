import torch
import torch.nn as nn
from x_transformers import TransformerWrapper, Decoder

if __name__ == '__main__':
  model = TransformerWrapper(
      num_tokens = 20000,
      max_seq_len = 2048,
      attn_layers = Decoder(
          dim = 512,
          depth = 6,
          heads = 8,
          attn_flash = False
      )
  ).cuda()

  # x = torch.randint(0, 256, (1, 2048)).cuda()
  x = torch.randn(1, 2048, 512).cuda()
  # mask = torch.randn(1, 2048)
  # mask = torch.ones_like(mask).bool().cuda()

  # model(x, return_embeddings=True).shape # (1, 1024, 20000)

  model_xl = TransformerWrapper(
      num_tokens = 20000,
      max_seq_len = 512,
      max_mem_len = 2048,
      attn_layers = Decoder(
          dim = 512,
          depth = 6,
          heads = 8,
          rel_pos_bias = True
      )
  )
  
  seg1 = torch.randint(0, 20000, (1, 512))
  seg2 = torch.randint(0, 20000, (1, 512))
  seg3 = torch.randint(0, 20000, (1, 512))

  logits1, mems1  = model_xl(seg1, return_mems = True, return_embeddings = True)
  logits2, mems2  = model_xl(seg2, mems = mems1, return_mems = True, return_embeddings = True)
  logits3, mems3  = model_xl(seg3, mems = mems2, return_mems = True)
