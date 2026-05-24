from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn.functional as F
from torch import nn

from ..prompts.auto_prompt import PromptBundle
from ..utils.decorators import logged_call, validate_tensor_output
from .backbones import ConvBlock


def apply_film(x: torch.Tensor, gamma: torch.Tensor, beta: torch.Tensor) -> torch.Tensor:
    if gamma.dim() != 1 or beta.dim() != 1:
        raise ValueError("FiLM parameters for a single expert must be 1D tensors [C].")
    return gamma.view(1, -1, 1, 1) * x + beta.view(1, -1, 1, 1)


class FiLMExpertDecoder(nn.Module):
    """Shared decoder weights with prototype-specific FiLM modulation."""

    def __init__(
        self,
        in_channels: int,
        pyramid_channels: Sequence[int] | None = None,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        num_classes: int = 1,
        prompt_channels: int = 1,
    ) -> None:
        super().__init__()
        self.decoder_channels = list(decoder_channels)
        self.num_classes = num_classes
        c4, c3, c2, c1 = self.decoder_channels
        self.pyramid_channels = list(pyramid_channels) if pyramid_channels is not None else [in_channels] * 4
        if len(self.pyramid_channels) != 4:
            raise ValueError("FiLMExpertDecoder expects four pyramid channel definitions.")

        self.input_projection = nn.Conv2d(in_channels, c4, kernel_size=1)
        self.prompt_projection = nn.Conv2d(prompt_channels, c4, kernel_size=1)
        self.skip3_projection = nn.Conv2d(self.pyramid_channels[2], c3, kernel_size=1)
        self.skip2_projection = nn.Conv2d(self.pyramid_channels[1], c2, kernel_size=1)
        self.skip1_projection = nn.Conv2d(self.pyramid_channels[0], c1, kernel_size=1)

        self.stage4 = ConvBlock(c4, c4)
        self.stage3 = ConvBlock(c4 + c3, c3)
        self.stage2 = ConvBlock(c3 + c2, c2)
        self.stage1 = ConvBlock(c2 + c1, c1)
        self.output_head = nn.Conv2d(c1, num_classes, kernel_size=1)

    @logged_call()
    @validate_tensor_output
    def forward(
        self,
        features: torch.Tensor | None = None,
        pyramid: list[torch.Tensor] | None = None,
        film_params: dict[int, dict[str, torch.Tensor]] | None = None,
        output_size: tuple[int, int] | None = None,
        prompts: PromptBundle | None = None,
        prompt_map: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if film_params is None:
            raise ValueError("film_params must be provided.")
        if output_size is None:
            raise ValueError("output_size must be provided.")
        if len(film_params) != len(self.decoder_channels):
            raise ValueError(f"Expected {len(self.decoder_channels)} FiLM stages, received {len(film_params)}.")

        if pyramid is not None:
            if len(pyramid) != 4:
                raise ValueError(f"Expected a 4-level pyramid, received {len(pyramid)} features.")
            p1, p2, p3, p4 = pyramid
            features = p4
        elif features is None:
            raise ValueError("Either features or pyramid must be provided to the decoder.")
        else:
            p1 = p2 = p3 = None

        x = self.input_projection(features)
        if prompt_map is not None:
            resized_prompt = F.interpolate(prompt_map, size=features.shape[-2:], mode="bilinear", align_corners=False)
            x = x + self.prompt_projection(resized_prompt)

        x = self.stage4(x)
        x = apply_film(x, film_params[0]["gamma"], film_params[0]["beta"])

        if p3 is not None:
            x = F.interpolate(x, size=p3.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, self.skip3_projection(p3)], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage3(x)
        x = apply_film(x, film_params[1]["gamma"], film_params[1]["beta"])

        if p2 is not None:
            x = F.interpolate(x, size=p2.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, self.skip2_projection(p2)], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage2(x)
        x = apply_film(x, film_params[2]["gamma"], film_params[2]["beta"])

        if p1 is not None:
            x = F.interpolate(x, size=p1.shape[-2:], mode="bilinear", align_corners=False)
            x = torch.cat([x, self.skip1_projection(p1)], dim=1)
        else:
            x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)
        x = self.stage1(x)
        x = apply_film(x, film_params[3]["gamma"], film_params[3]["beta"])
        x = F.interpolate(x, size=output_size, mode="bilinear", align_corners=False)
        return self.output_head(x)
