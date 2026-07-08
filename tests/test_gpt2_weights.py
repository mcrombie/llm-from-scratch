import unittest

import torch

from book_llm.gpt2_weights import assign, build_gpt2_config, model_name_to_size


class Gpt2WeightHelperTests(unittest.TestCase):
    def test_model_name_to_size(self):
        self.assertEqual(model_name_to_size("gpt2-small (124M)"), "124M")
        self.assertEqual(model_name_to_size("gpt2-medium (355M)"), "355M")

    def test_build_gpt2_config_sets_context_and_bias(self):
        cfg = build_gpt2_config("gpt2-small (124M)")

        self.assertEqual(cfg["context_length"], 1024)
        self.assertTrue(cfg["qkv_bias"])
        self.assertEqual(cfg["emb_dim"], 768)

    def test_assign_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            assign(torch.zeros(2, 3), torch.zeros(3, 2))


if __name__ == "__main__":
    unittest.main()
