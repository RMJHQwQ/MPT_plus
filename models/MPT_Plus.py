import torch

import torch.nn as nn
from models.qformer import QFormer
from models.utils import CrossAttention, SKNetandLSTM_Model, AxModel, SKNet_LSTM_Attention_Model


# torch.cuda.manual_seed(42)
class SwiGLU(nn.Module):
    def __init__(self):
        super().__init__()
        self.gate = nn.Sigmoid()  # Swish uses sigmoid

    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return x1 * self.gate(x2)




import torch
import torch.nn as nn
from einops.layers.torch import EinMix as E



class SwiGLU(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=-1)
        return nn.SiLU()(x1) * x2


class CrossAttention(nn.Module):
    def __init__(self, dim, heads=8, dropout=0.1):
        super().__init__()
        self.heads = heads
        self.scale = dim ** -0.5

        # Query, Key, Value Projections for Cross-Attention
        self.to_q = nn.Linear(dim, dim)
        self.to_k = nn.Linear(dim, dim)
        self.to_v = nn.Linear(dim, dim)

        self.to_out = nn.Sequential(
            nn.Linear(dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, q, k, v):
        # q: (batch, q_len, dim) | k, v: (batch, kv_len, dim)

        # Project and reshape for multi-head
        # (batch, len, dim) -> (batch, len, heads, head_dim) -> (batch, heads, len, head_dim)
        B, QL, D = q.shape
        H = self.heads
        D_head = D // H

        queries = self.to_q(q).reshape(B, QL, H, D_head).permute(0, 2, 1, 3)
        keys = self.to_k(k).reshape(B, -1, H, D_head).permute(0, 2, 3,
                                                              1)  # Key needs to be transposed for multiplication
        values = self.to_v(v).reshape(B, -1, H, D_head).permute(0, 2, 1, 3)

        # Attention calculation: (batch, heads, q_len, head_dim) @ (batch, heads, head_dim, kv_len)
        # -> (batch, heads, q_len, kv_len)
        attn = torch.matmul(queries, keys) * self.scale
        attn = nn.Softmax(dim=-1)(attn)

        # Apply attention: (batch, heads, q_len, kv_len) @ (batch, heads, kv_len, head_dim)
        # -> (batch, heads, q_len, head_dim)
        out = torch.matmul(attn, values)

        # Reshape back: (batch, heads, q_len, head_dim) -> (batch, q_len, heads, head_dim) -> (batch, q_len, dim)
        out = out.permute(0, 2, 1, 3).reshape(B, QL, D)

        return self.to_out(out)

class MPT_Plus(nn.Module):
    def __init__(self, batch_size, basemodel, visual_model):
        super().__init__()
        self.batch_size = batch_size
        self.base_model = basemodel
        self.visual_model = visual_model
        dim = 1024
        self.dim = dim

        # --- NEW Cross-Attention Modules ---
        # All inputs (txt_f, img_f, llm) are expected to be (batch, length, dim)
        # LLM output (Q) is (batch, 320, 1024)
        # txt_f/img_f (K/V) are (batch, 1, 1024) after unsqueeze

        # 1. LLM (Q) vs TXT_F (K/V)
        self.llm_to_txt_att = CrossAttention(dim=dim, heads=1, dropout=0.1)
        # 2. LLM (Q) vs IMG_F (K/V)
        self.llm_to_img_att = CrossAttention(dim=dim, heads=1, dropout=0.1)
        # 3. TXT_F (Q) vs IMG_F (K/V) - for interaction between modalities
        self.txt_to_img_att = CrossAttention(dim=dim, heads=1, dropout=0.1)
        # 4. IMG_F (Q) vs TXT_F (K/V) - for interaction between modalities
        self.img_to_txt_att = CrossAttention(dim=dim, heads=1, dropout=0.1)
        # --- END NEW Cross-Attention Modules ---

        self.dropout = nn.Dropout(0.2)
        self.softmax = nn.Softmax(dim=-1)
        self.act = nn.Tanh()
        self.SwiGlu = SwiGLU()

        # New Projection Layer:
        # Input: Concat of 4 attention results (llm_len=320) + LLM_CLS (1) + TXT_F (1) + IMG_F (1)
        # (4 * 320 + 3) * dim. This is too large.
        # Strategy: Use Mean-Pooling on attention results (320 -> 1) for a fixed size output.
        # Total input dim for final fusion: (1 + 1 + 1 + 1) * dim = 4 * 1024 = 4096
        # If we use LLM_CLS + 4 mean-pooled attention features + TXT_F + IMG_F: (1+4+1+1) * 1024 = 7168

        # Let's simplify and use the 4 attention outputs after mean-pooling.
        fusion_input_dim = 4 * dim  # Four vectors of size 1024 concatenated

        self.fusion_projection = nn.Sequential(
            nn.Linear(fusion_input_dim, dim),  # Downsample to the target dim (1024)
            nn.GELU(),
            nn.LayerNorm(dim),
            nn.Dropout(0.2)
        )

        # Adjust downsample input dimension if using the output of fusion_projection
        self.downsample = nn.Sequential(
            nn.LayerNorm(dim),  # Input dim is 1024
            nn.Linear(dim, dim * 4),
            self.SwiGlu,
            nn.Linear(dim * 2, dim),
            nn.Dropout(0.2),
        )

        for param in basemodel.parameters():
            param.requires_grad = (True)
        self.axmodel = AxModel(basemodel)  # Assuming AxModel is defined elsewhere
        self.pemnet_txt = SKNetandLSTM_Model(self.base_model, 3,
                                             dim)  # Assuming SKNetandLSTM_Model is defined elsewhere
        self.classf = nn.Linear(dim, 3)
        self.GELU = nn.GELU()


    def forward(self, input, img, pos, neu, neg):
        # txt_f, img_f, and llm are the three features.
        txt_f = self.pemnet_txt(input)  # (batch, 1024)
        img_f = self.visual_model(img)  # (batch, 1024)
        llm = self.axmodel(pos, neu, neg)  # (batch, 320, 1024)

        # Check and prepare dimensions for Cross-Attention (Q, K, V must be (B, L, D))
        # txt_f/img_f need sequence dimension L=1
        txt_f_seq = txt_f.unsqueeze(1)  # (batch, 1, 1024)
        img_f_seq = img_f.unsqueeze(1)  # (batch, 1, 1024)
        # llm is already (batch, 320, 1024)

        # (A) LLM as Query, TXT_F as K/V
        # Output: (batch, 320, 1024) -> LLM representations attended over text
        llm_over_txt = self.llm_to_txt_att(llm, txt_f_seq, txt_f_seq)

        # (B) LLM as Query, IMG_F as K/V
        # Output: (batch, 320, 1024) -> LLM representations attended over image
        llm_over_img = self.llm_to_img_att(llm, img_f_seq, img_f_seq)

        # (C) TXT_F as Query, IMG_F as K/V
        # Output: (batch, 1, 1024) -> Text representation attended over image
        txt_over_img = self.txt_to_img_att(txt_f_seq, img_f_seq, img_f_seq)

        # (D) IMG_F as Query, TXT_F as K/V
        # Output: (batch, 1, 1024) -> Image representation attended over text
        img_over_txt = self.img_to_txt_att(img_f_seq, txt_f_seq, txt_f_seq)

        # Mean-pool the two LLM-based attention results to get a fixed size (batch, 1024) vector
        # (batch, 320, 1024) -> (batch, 1024)
        llm_txt_pooled = torch.mean(llm_over_txt, dim=1)
        llm_img_pooled = torch.mean(llm_over_img, dim=1)

        # Squeeze the two fixed-size attention results to (batch, 1024)
        # (batch, 1, 1024) -> (batch, 1024)
        txt_img_squeezed = txt_over_img.squeeze(1)
        img_txt_squeezed = img_over_txt.squeeze(1)

        # Concatenate the four fixed-size vectors
        # (batch, 4 * 1024)
        fused_att_features = torch.cat([
            llm_txt_pooled,
            llm_img_pooled,
            txt_img_squeezed,
            img_txt_squeezed
        ], dim=-1)

        # Apply final projection to get the fused vector 'fused'
        fused = self.fusion_projection(fused_att_features)  # (batch, 1024)

        # Add residual connection and downsample (as per your original code pattern)
        # downsample sequence is now applied to the single fused vector
        pre = self.downsample(fused) + fused  # (batch, 1024)

        # Final classification
        res = self.classf(pre)  # (batch, 3)

        return res, fused