import torch
import torch.nn as nn
import torch.nn.functional as F

class ScaledDotProductAttention(nn.Module):
    def __init__(self, temperature, dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        # batch_size, num_heads, len_q, dim_k
        attention = torch.matmul(q / self.temperature, k.transpose(2, 3))   # len_k <-> dim_k
        if mask is not None:
            attention = attention.masked_fill(mask == 0, -1e9)
        attention = self.dropout(F.softmax(attention, dim=-1))
        # batch_size, num_heads, len_q, dim_v
        output = torch.matmul(attention, v)

        return output, attention

class MultiheadAttention(nn.Module):
    def __init__(self, dim_model, dim_k, dim_v, num_heads, dropout=0.1):    # dim_model = dim_k * num_heads
        super().__init__()

        self.num_heads = num_heads
        self.embed_dim = dim_model
        self.dim_q = dim_k
        self.dim_k = dim_k
        self.dim_v = dim_v

        self.W_qs = nn.Linear(dim_model, num_heads * dim_k, bias=False)
        self.W_ks = nn.Linear(dim_model, num_heads * dim_k, bias=False)
        self.W_vs = nn.Linear(num_heads * dim_v, dim_model, bias=False)
        self.fc = nn.Linear(num_heads * dim_v, dim_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=dim_k ** 0.5)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(dim_model, eps=1e-6)

    def forward(self, q, k, v, mask=None):
        # batch_size, len_q, dim_model
        dim_k, dim_v, num_heads = self.dim_k, self.dim_v, self.num_heads
        batch_size, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # batch_size, len_q, num_heads, dim_k
        q = self.W_qs(q).view(batch_size, len_q, num_heads, dim_k)
        k = self.W_ks(k).view(batch_size, len_k, num_heads, dim_k)
        v = self.W_vs(v).view(batch_size, len_v, num_heads, dim_v)

        # batch_size, num_heads, len_q, dim_k
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        # batch_size, 1, len_q, dim_k
        if mask is not None:
            mask = mask.unsqueeze(1)

        q, attention = self.attention(q, k, v, mask=mask)

        # batch_size, len_q, num_heads * dim_k
        q = q.transpose(1, 2).contiguous().view(batch_size, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q, attention
