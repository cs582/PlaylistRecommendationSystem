import time
import torch
import unittest
import numpy as np

from src.models.transformer_based_recommender import RecommendationSystem, TransformerEncoder, TransformerEncoderLayer, Attention


out_size = 1000

max_length = 16
vocab_size = 1000
device = 'cpu'
eos_id = 1

b = 32

layers = 2
nheads = 8
model_dim = 256
att_dim = model_dim // nheads
ff_dim = 128


def test_model(model, x, name=None):
    start = time.time()
    out = model(x)
    end = time.time()
    print(f"{name} Test Finished in {end-start} seconds")
    return out.shape, end-start


class ClassifierUnitTest(unittest.TestCase):
    def test_attention(self):
        x = torch.rand(b, max_length, att_dim)
        model = Attention(model_dim=model_dim, att_dim=att_dim)

        out_shape, timing = test_model(model, x, "Attention")
        desired_shape = (b, max_length, att_dim)
        self.assertEqual(
            out_shape,
            desired_shape,
            f"Output shape is not right, obtained {out_shape}, should be {desired_shape}"
        )

    def test_encoder_layer(self):
        x = torch.rand(b, max_length, model_dim)
        model = TransformerEncoderLayer(model_dim=model_dim, ff_dim=ff_dim, nhead=nheads)
        out_shape, timing = test_model(model, x, "Encoder Layer")
        desired_shape = (b, max_length, model_dim)
        self.assertEqual(
            out_shape,
            desired_shape,
            f"Output shape is not right, obtained {out_shape}, should be {desired_shape}"
        )

    def test_encoder(self):
        x = torch.rand(b, max_length, model_dim)
        model = TransformerEncoder(model_dim=model_dim, ff_dim=ff_dim, nhead=nheads, max_length=max_length, layers=layers, vocab_size=vocab_size)
        out_shape, timing = test_model(model, x, "Encoder")
        desired_shape = (b, max_length, model_dim)
        self.assertEqual(
            out_shape,
            desired_shape,
            f"Output shape is not right, obtained {out_shape}, should be {desired_shape}"
        )

    def test_recommender(self):
        x = torch.zeros(b, max_length, dtype=torch.int32, device=device)

        for i in range(len(x)):
            sequence_length = np.random.randint(low=1, high=max_length-1)
            x[i, :sequence_length] = torch.randint(low=2, high=vocab_size, size=(1, sequence_length))
            x[i, sequence_length] = eos_id

        model = RecommendationSystem(vocab_size=vocab_size, max_length=max_length, nhead=nheads, model_dim=model_dim, ff_dim=ff_dim, layers=layers, device=device, eos_id=eos_id, out_size=out_size)
        out_shape, timing = test_model(model, x, "Recommender")
        desired_shape = (b, out_size)
        self.assertEqual(
            out_shape,
            desired_shape,
            f"Output shape is not right, obtained {out_shape}, should be {desired_shape}"
        )