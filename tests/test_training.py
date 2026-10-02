"""Small CPU checks for the training loop; no dataset, weights or network needed."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch
import torch
from torch import nn
from src.train import run_two_phase_training, train_one_epoch


class TrainingTests(unittest.TestCase):
    def test_weighted_loss_and_phase_one_checkpoint_survive(self):
        model = nn.Linear(2, 2)
        config = SimpleNamespace(DEVICE="cpu", SAMPLE_RATE=16000, CLIP_DURATION=10)
        weights = torch.tensor([1., 3.])
        with tempfile.TemporaryDirectory() as directory, patch.dict("sys.modules", {"wandb": MagicMock()}), patch("src.train.train_one_epoch", return_value=(.1, .5, .5)) as train, patch("src.train.validate_one_epoch", return_value=(.1, .5, .5, [.5, .5])):
            result, score = run_two_phase_training(model, "CRNN", [1], [1], 1, 1, .001, .001, str(Path(directory)/"best.pt"), None, MagicMock(), class_weights=weights, config=config)
            self.assertTrue(torch.equal(train.call_args.args[4].weight, weights))
            self.assertEqual(score, .5)
            self.assertIs(result, model)
            self.assertTrue((Path(directory)/"best.pt").exists())

    def test_partial_accumulation_window_is_not_underweighted(self):
        model = nn.Linear(1, 2, bias=False)
        with torch.no_grad():
            model.weight.zero_()
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        loader = [(torch.ones(1, 1), torch.zeros(1, dtype=torch.long))]
        train_one_epoch(model, loader, optimizer, None, nn.CrossEntropyLoss(), torch.cuda.amp.GradScaler(enabled=False), "cpu", "CRNN", 1, accum_steps=4)
        self.assertAlmostEqual(model.weight[0, 0].item(), .05, places=5)
