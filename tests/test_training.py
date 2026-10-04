import copy
import random
import unittest
from types import SimpleNamespace
import numpy as np
import torch
from src.config import load_params
from src.data import accumulation_groups, pad_batch
from src.runtime import set_seed
from src.train import evaluate, lora_config


class TrainingTests(unittest.TestCase):
    def test_checkpoint_restores_optimizer_and_rng(self):
        import tempfile
        from pathlib import Path
        from src.checkpoint import save_checkpoint, restore_checkpoint
        set_seed(42)
        model = torch.nn.Sequential(torch.nn.Linear(3, 2), torch.nn.Dropout(.2))
        optimizer = torch.optim.AdamW(model.parameters(), lr=.01)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, .9)
        def update():
            optimizer.zero_grad()
            model(torch.rand(2, 3)).square().mean().backward()
            optimizer.step(); scheduler.step()
        update()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'state.pt'
            save_checkpoint(path, model, optimizer, scheduler, dict(fingerprint='test', step=1), torch.device('cpu'))
            update()
            expected = {n: p.detach().clone() for n, p in model.named_parameters()}
            expected_random = (random.random(), float(np.random.random()))
            state = restore_checkpoint(path, model, optimizer, scheduler, dict(fingerprint='test'), torch.device('cpu'))
            self.assertEqual(state['step'], 1)
            update()
            for n, parameter in model.named_parameters():
                self.assertTrue(torch.equal(parameter, expected[n]))
            self.assertEqual((random.random(), float(np.random.random())), expected_random)
            with self.assertRaises(ValueError):
                restore_checkpoint(path, model, optimizer, scheduler, dict(fingerprint='wrong'), torch.device('cpu'))

    def test_partial_accumulation_group_is_kept(self):
        self.assertEqual(list(accumulation_groups(iter(range(10)), 8)), [list(range(8)), [8, 9]])
        with self.assertRaises(ValueError):
            list(accumulation_groups(iter([1]), 0))

    def test_lora_covers_only_requested_layers(self):
        p = load_params()
        all_cfg, half = lora_config(p, 28, 0), lora_config(p, 28, 14)
        self.assertEqual(all_cfg.layers_to_transform, list(range(28)))
        self.assertEqual(half.layers_to_transform, list(range(14, 28)))
        self.assertIsNone(half.modules_to_save)
        with self.assertRaises(ValueError):
            lora_config(p, 28, 28)

    def test_bucket_padding_preserves_every_token(self):
        example = dict(input_ids=[1,2,3], labels=[-100,2,3], attention_mask=[1,1,1])
        batch = pad_batch([example], 0, pad_multiple=128)
        self.assertEqual(batch['input_ids'].shape, (1,128))
        self.assertEqual(batch['input_ids'][0,-3:].tolist(), example['input_ids'])
        self.assertEqual(int(batch['attention_mask'].sum()), 3)
        self.assertEqual(int((batch['labels'] != -100).sum()), 2)

    def test_left_padding_never_contributes_to_loss(self):
        out = pad_batch([dict(input_ids=[1,2,3], labels=[-100,2,3], attention_mask=[1,1,1]),
                         dict(input_ids=[1,3], labels=[-100,3], attention_mask=[1,1])], 0)
        self.assertEqual(out['attention_mask'][1].tolist(), [0,1,1])
        self.assertEqual(out['labels'][1].tolist(), [-100,-100,3])

    def test_evaluation_weights_shifted_answer_tokens(self):
        class Fake(torch.nn.Module):
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(loss=torch.tensor(float(input_ids[0,-1])))
        model = Fake().eval()
        examples = [dict(input_ids=[0,2],labels=[-100,2],attention_mask=[1,1]),
                    dict(input_ids=[0,1,1,4],labels=[-100,1,1,4],attention_mask=[1,1,1,1])]
        value = evaluate(model, examples, 0, torch.device('cpu'), 1)
        self.assertEqual(value, 3.5)
        self.assertFalse(model.training)

    def test_empty_validation_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate(torch.nn.Linear(1,1), [], 0, torch.device('cpu'), 1)

    def test_seed_covers_python_numpy_torch(self):
        def values():
            return random.random(), float(np.random.random()), float(torch.rand(()))
        set_seed(42); a = values()
        set_seed(42); self.assertEqual(a, values())


if __name__ == '__main__':
    unittest.main()
