import torch
import torch.nn as nn


class GPTMusicRecommender(nn.Module):
    def __init__(self, track_library_size, max_size, model_dim, ff_dim, layers, n_heads, special_tokens):
        super(GPTMusicRecommender).__init__()

        self.special_tokens = special_tokens

        self.position_embedding = nn.Parameter(torch.rand(max_size, model_dim))
        self.track_embedding = nn.Embedding(num_embeddings=track_library_size + len(self.special_tokens), embedding_dim=model_dim)

        self.transformer_e_layer = nn.TransformerEncoderLayer(d_model=model_dim, dim_feedforward=ff_dim, nhead=n_heads, batch_first=True, norm_first=True)
        self.transformer_d_layer = nn.TransformerDecoderLayer(d_model=model_dim, dim_feedforward=ff_dim, nhead=n_heads, batch_first=True, norm_first=True)

        self.transformer_tracks = nn.TransformerEncoder(encoder_layer=self.transformer_e_layer, num_layers=layers)
        self.transformer_options = nn.TransformerDecoder(decoder_layer=self.transformer_d_layer, num_layers=layers)

        self.layer_norm = nn.LayerNorm(model_dim)
        self.to_latent = nn.Identity()

        self.fc_options = nn.Sequential(
            nn.LayerNorm(model_dim),
            nn.Linear(model_dim, ff_dim),
            nn.Linear(ff_dim, model_dim)
        )

    def forward(self, tracks, options):
        class_mask = (tracks == self.special_tokens["[class]"])

        # Track + Position Embedding
        x = self.track_embedding(tracks) + self.position_embedding
        x = self.transformer_tracks(x, mask_tracks)

        # Options track embeddings
        z = self.track_embedding(options)
        z = self.fc_options(z)
        z = self.transformer_options(z, x, mask_options)

        # Compute the distribution over all options
        out = self.to_latent(z[class_mask])
        out = self.layer_norm(out)
        return out