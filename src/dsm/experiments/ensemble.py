from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch.optim import AdamW

from ..metrics import predictive_entropy
from ..models.unet import SimpleUNet
from ..utils.decorators import ensure_directory, logged_call, timed
from ..utils.reproducibility import seed_everything


@dataclass(slots=True)
class EnsemblePrediction:
    mean_prob: torch.Tensor
    variance: torch.Tensor
    entropy: torch.Tensor
    member_probs: torch.Tensor


class UNetEnsembleBaseline:
    def __init__(self, in_channels: int = 1, num_classes: int = 1, num_members: int = 10) -> None:
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_members = num_members

    @logged_call()
    @ensure_directory(lambda self, train_loader, output_dir, **kwargs: output_dir)
    @timed("ensemble_train_seconds")
    def train_members(
        self,
        train_loader,
        output_dir: str,
        epochs: int = 1,
        lr: float = 1e-3,
        device: torch.device | None = None,
    ) -> None:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        criterion = torch.nn.BCEWithLogitsLoss()
        output_path = Path(output_dir)
        for member_id in range(self.num_members):
            seed_everything(1000 + member_id)
            model = SimpleUNet(self.in_channels, self.num_classes).to(device)
            optimizer = AdamW(model.parameters(), lr=lr)
            model.train()
            for _ in range(epochs):
                for batch in train_loader:
                    image = batch["image"].to(device)
                    mask = batch["mask"].to(device)
                    logits = model(image)
                    loss = criterion(logits, mask)
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    optimizer.step()
            torch.save(model.state_dict(), output_path / f"model_seed_{member_id}.pth")

    @logged_call()
    @timed("ensemble_inference_seconds")
    def predict(self, x: torch.Tensor, checkpoint_dir: str, device: torch.device | None = None) -> EnsemblePrediction:
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        probs = []
        checkpoint_dir = Path(checkpoint_dir)
        for member_id in range(self.num_members):
            model = SimpleUNet(self.in_channels, self.num_classes).to(device)
            state_dict = torch.load(checkpoint_dir / f"model_seed_{member_id}.pth", map_location=device)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                probs.append(torch.sigmoid(model(x.to(device))))
        member_probs = torch.stack(probs, dim=0)
        mean_prob = member_probs.mean(dim=0)
        variance = member_probs.var(dim=0, unbiased=False)
        entropy = predictive_entropy(mean_prob)
        return EnsemblePrediction(
            mean_prob=mean_prob,
            variance=variance,
            entropy=entropy,
            member_probs=member_probs,
        )
