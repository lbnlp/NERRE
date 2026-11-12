"""Exercise actual Transformers checkpoint operations on a tiny FLAN-style T5."""
import tempfile
import unittest
from pathlib import Path

import torch
from transformers import T5Config, T5ForConditionalGeneration

from app.backend.extractor import load_seq2seq_model


class TransformersCompatibilityTests(unittest.TestCase):
    def test_untied_flan_weights_survive_resize_and_reload(self):
        config = T5Config(vocab_size=16, d_model=8, d_kv=4, d_ff=16,
                          num_layers=1, num_heads=2, tie_word_embeddings=False,
                          decoder_start_token_id=0)
        config.tie_word_embeddings = False
        model = T5ForConditionalGeneration(config)
        with torch.no_grad():
            model.lm_head.weight.fill_(0.125)
            model.shared.weight.fill_(0.25)
        with tempfile.TemporaryDirectory() as tmp:
            source, saved = Path(tmp) / 'source', Path(tmp) / 'saved'
            model.save_pretrained(source)
            loaded = load_seq2seq_model(source, local_files_only=True)
            self.assertFalse(loaded.config.scale_decoder_outputs)
            loaded.resize_token_embeddings(17)
            self.assertTrue(torch.equal(loaded.lm_head.weight[:16], model.lm_head.weight))
            self.assertTrue(torch.equal(loaded.shared.weight[:16], model.shared.weight))
            loaded.save_pretrained(saved)
            reloaded = load_seq2seq_model(saved, local_files_only=True)
            self.assertIs(reloaded.encoder.embed_tokens.weight, reloaded.shared.weight)
            self.assertIs(reloaded.decoder.embed_tokens.weight, reloaded.shared.weight)
            self.assertFalse(reloaded.config.scale_decoder_outputs)
            self.assertTrue(torch.equal(reloaded.lm_head.weight, loaded.lm_head.weight))
            self.assertTrue(torch.equal(reloaded.shared.weight, loaded.shared.weight))
