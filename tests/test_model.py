import unittest

import torch

from book_llm.model import GPTModel


class ModelTests(unittest.TestCase):
    def test_gpt_model_forward_shape(self):
        cfg = {
            "vocab_size": 32,
            "context_length": 8,
            "emb_dim": 16,
            "n_heads": 4,
            "n_layers": 1,
            "drop_rate": 0.0,
            "qkv_bias": False,
        }
        model = GPTModel(cfg)

        logits = model(torch.tensor([[1, 2, 3], [4, 5, 6]]))

        self.assertEqual(tuple(logits.shape), (2, 3, 32))


if __name__ == "__main__":
    unittest.main()
