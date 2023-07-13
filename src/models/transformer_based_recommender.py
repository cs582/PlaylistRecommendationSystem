import torch
import torch.nn as nn

class RecommendationSystem(nn.Module):
    def __init__(self, vocab_size, max_length, nhead, model_dim, ff_dim, layers, device, out_size, eos_id):
        super(RecommendationSystem, self).__init__()

        self.device = device
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.nhead = nhead
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.layers = layers
        self.out_size = out_size

        self.eos_id = eos_id

        self.poss_embedding = nn.Parameter(torch.rand(1, max_length, model_dim))
        self.token_embedding = nn.Embedding(num_embeddings=vocab_size, embedding_dim=model_dim)
        self.encoder = TransformerEncoder(model_dim, ff_dim, vocab_size, nhead, max_length, layers)
        self.to_latent = nn.Identity()

        self.linear_out = nn.Linear(model_dim, out_size)

    def forward(self, x):
        b, l = x.shape

        # create [EOS] mask
        eos_mask = torch.zeros(b, l, device=self.device, dtype=torch.bool)
        eos_mask[:, :] = (x == self.eos_id)

        # Encode tokens
        x = self.token_embedding(x) + self.poss_embedding

        # Pass through transformer encoder
        x = self.encoder(x)

        # Get eos mask
        x = x[eos_mask]
        x = self.to_latent(x)

        # Output layer
        x = self.linear_out(x)
        x = nn.functional.softmax(x, dim=-1)

        return x



class TransformerEncoder(nn.Module):
    def __init__(self, model_dim, ff_dim, vocab_size, nhead, max_length, layers):
        super(TransformerEncoder, self).__init__()
        self.model_dim = model_dim
        self.ff_dim = ff_dim
        self.vocab_size = vocab_size
        self.nhead = nhead
        self.max_length = max_length

        self.transformers = nn.ModuleList([
            TransformerEncoderLayer(model_dim=model_dim, nhead=nhead, ff_dim=ff_dim) for _ in range(layers)
        ])

    def forward(self, x):
        for transformer in self.transformers:
            x = transformer(x)
        return x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, model_dim, nhead, ff_dim):
        super(TransformerEncoderLayer, self).__init__()

        self.nhead = nhead
        self.ff_dim = ff_dim

        self.msa = MSA(model_dim=model_dim, nhead=nhead)
        self.ln1 = nn.LayerNorm(model_dim)
        self.mlp = MLP(dim_in=model_dim, dim_hid=ff_dim, dim_out=model_dim)
        self.ln2 = nn.LayerNorm(model_dim)

    def forward(self, x):
        x_ = self.ln1(x)
        x = torch.add(self.msa(x_), x)
        x_ = self.ln2(x)
        x = torch.add(self.mlp(x_), x)
        return x

class MSA(nn.Module):
    def __init__(self, model_dim, nhead):
        super(MSA, self).__init__()
        self.model_dim = model_dim
        self.att_dim = self.model_dim // nhead
        self.n = nhead

        self.heads = nn.ModuleList([
            Attention(model_dim=self.model_dim, att_dim=self.att_dim) for _ in range(self.n)
        ])
        self.linear = nn.Linear(self.n*self.att_dim, self.model_dim)

    def forward(self, x):

        out = [None] * self.n
        for i, attention in enumerate(self.heads):
            out[i] = attention(x)

        x = torch.cat(out, dim=-1)
        x = self.linear(x)
        return x


class MLP(nn.Module):
    def __init__(self, dim_in, dim_hid, dim_out):
        super(MLP, self).__init__()

        self.dim_in = dim_in
        self.dim_hid = dim_hid
        self.dim_out = dim_out

        self.net = nn.Sequential(
            nn.Linear(self.dim_in, self.dim_hid, bias=True),
            nn.GELU(),
            nn.Linear(self.dim_hid, self.dim_out, bias=True)
        )

    def forward(self, x):
        x = self.net(x)
        return x

class Attention(nn.Module):
    def __init__(self, model_dim, att_dim):
        super(Attention, self).__init__()
        self.model_dim = model_dim

        self.k_dim = att_dim
        self.v_dim = att_dim
        self.q_dim = att_dim

        self.k = nn.Parameter(torch.rand(1, self.model_dim, self.k_dim))
        self.q = nn.Parameter(torch.rand(1, self.model_dim, self.q_dim))
        self.v = nn.Parameter(torch.rand(1, self.model_dim, self.v_dim))

    def forward(self, x, mask=None):
        k = torch.einsum('blx,odk->blk', x, self.k) # b x length x k_dim
        q = torch.einsum('blx,odq->blq', x, self.q)  # b x length x q_dim
        v = torch.einsum('blx,odv->blv', x, self.v) # b x length x v_dim

        s = torch.einsum('blk,bmq->blm', k, q) # b x length x length
        if mask is not None:
            s = s.masked_fill(~mask, -1000.0)
        s = s / (self.k_dim ** 0.5)
        s = nn.functional.softmax(s, dim=-1)
        out = torch.einsum('blm,blv->blv', s, v)  # b x length x key_dim
        return out