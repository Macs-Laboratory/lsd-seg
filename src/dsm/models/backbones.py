from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import warnings

import torch
import torch.nn.functional as F
from torch import nn
from torchvision import models

from ..prompts.auto_prompt import PromptBundle, scale_prompts_to_sam_input
from ..utils.decorators import logged_call, validate_tensor_output


class ConvBlock(nn.Module):
    """Two-layer convolutional block used by lightweight decoder components."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


@dataclass(slots=True)
class EncoderOutput:
    features: torch.Tensor
    pyramid: list[torch.Tensor]
    image_embeddings: torch.Tensor | None
    high_res_features: list[torch.Tensor] | None
    sam_outputs: Any | None


class BaseFrozenEncoder(nn.Module):
    def __init__(self, freeze: bool = True) -> None:
        super().__init__()
        self.freeze = freeze
        self.output_channels: list[int] = []
        self.native_image_size: int | None = None

    def train(self, mode: bool = True) -> "BaseFrozenEncoder":
        super().train(mode)
        if self.freeze:
            self._set_frozen_eval_mode()
        return self

    def _set_frozen_eval_mode(self) -> None:
        raise NotImplementedError

    def has_native_mask_decoder(self) -> bool:
        return False

    def predict_mask_prior(
        self,
        image_embeddings: torch.Tensor | None,
        high_res_features: list[torch.Tensor] | None,
        prompts: PromptBundle,
        original_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        raise RuntimeError("Native SAM/SAM2 mask decoding is not available for this encoder.")


class FrozenResNetEncoder(BaseFrozenEncoder):
    _BACKBONES: dict[str, tuple[Any, Any, list[int]]] = {
        "resnet18": (models.resnet18, models.ResNet18_Weights.DEFAULT, [64, 128, 256, 512]),
        "resnet34": (models.resnet34, models.ResNet34_Weights.DEFAULT, [64, 128, 256, 512]),
        "resnet50": (models.resnet50, models.ResNet50_Weights.DEFAULT, [256, 512, 1024, 2048]),
    }

    def __init__(
        self,
        encoder_name: str = "resnet18",
        in_channels: int = 1,
        pretrained: bool = True,
        freeze: bool = True,
        output_channels: list[int] | tuple[int, ...] | None = None,
        checkpoint_path: str | None = None,
    ) -> None:
        super().__init__(freeze=freeze)
        if encoder_name not in self._BACKBONES:
            available = ", ".join(sorted(self._BACKBONES))
            raise ValueError(f"Unknown encoder_name '{encoder_name}'. Available backbones: {available}")

        builder, default_weights, raw_channels = self._BACKBONES[encoder_name]
        weights = default_weights if pretrained else None
        try:
            backbone = builder(weights=weights)
        except Exception:
            backbone = builder(weights=None)
        if checkpoint_path:
            state_dict = torch.load(Path(checkpoint_path), map_location="cpu")
            missing, unexpected = backbone.load_state_dict(state_dict, strict=False)
            if unexpected:
                raise RuntimeError(f"Unexpected keys while loading encoder checkpoint: {unexpected}")
            if missing and pretrained:
                raise RuntimeError(f"Missing keys while loading encoder checkpoint: {missing}")

        self.backbone = backbone
        self.raw_channels = raw_channels
        self.target_channels = list(output_channels) if output_channels is not None else list(raw_channels)
        if len(self.target_channels) != 4:
            raise ValueError("FrozenResNetEncoder expects 4 output channels for the feature pyramid.")

        self._adapt_input_conv(in_channels)
        self.projections = nn.ModuleList(
            [
                nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(self.raw_channels, self.target_channels)
            ]
        )
        self.output_channels = list(self.target_channels)

        if self.freeze:
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False
            self._set_frozen_eval_mode()

    def _adapt_input_conv(self, in_channels: int) -> None:
        old_conv = self.backbone.conv1
        if old_conv.in_channels == in_channels:
            return
        new_conv = nn.Conv2d(
            in_channels,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=old_conv.bias is not None,
        )
        with torch.no_grad():
            if old_conv.weight.shape[1] == 3 and in_channels == 1:
                new_conv.weight.copy_(old_conv.weight.mean(dim=1, keepdim=True))
            else:
                repeated = old_conv.weight.mean(dim=1, keepdim=True).repeat(1, in_channels, 1, 1)
                new_conv.weight.copy_(repeated / max(in_channels, 1))
        self.backbone.conv1 = new_conv

    def _set_frozen_eval_mode(self) -> None:
        self.backbone.eval()

    def _extract_backbone(self, x: torch.Tensor) -> list[torch.Tensor]:
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        c2 = self.backbone.layer1(x)
        c3 = self.backbone.layer2(c2)
        c4 = self.backbone.layer3(c3)
        c5 = self.backbone.layer4(c4)
        return [c2, c3, c4, c5]

    @logged_call()
    @validate_tensor_output
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        if x.dim() != 4:
            raise ValueError(f"Expected input [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] != self.backbone.conv1.in_channels:
            raise ValueError(
                f"FrozenResNetEncoder expected {self.backbone.conv1.in_channels} input channels, got {x.shape[1]}"
            )

        if self.freeze:
            with torch.no_grad():
                pyramid = self._extract_backbone(x)
        else:
            pyramid = self._extract_backbone(x)
        projected = [projection(feature) for projection, feature in zip(self.projections, pyramid)]
        if len(projected) != 4:
            raise RuntimeError("ResNet encoder must return a 4-level pyramid.")
        return {
            "features": projected[-1],
            "pyramid": projected,
            "image_embeddings": projected[-1],
            "high_res_features": None,
            "sam_outputs": None,
        }


class _SAMNativeMixin:
    prompt_encoder: nn.Module | None
    mask_decoder: nn.Module | None
    image_encoder: nn.Module
    mask_input_size: int
    allow_prompt_mask_retry: bool

    def _freeze_optional_module(self, module: nn.Module | None, should_freeze: bool) -> None:
        if module is None or not should_freeze:
            return
        for parameter in module.parameters():
            parameter.requires_grad = False
        module.eval()

    def has_native_mask_decoder(self) -> bool:
        return self.prompt_encoder is not None and self.mask_decoder is not None

    def _call_prompt_encoder(
        self,
        prompts: PromptBundle,
        original_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.prompt_encoder is None:
            raise RuntimeError("Prompt encoder is not available.")
        if self.native_image_size is None:
            raise RuntimeError("Native image size must be set before prompt encoding.")
        scaled = scale_prompts_to_sam_input(
            prompts,
            original_size=original_size,
            sam_image_size=self.native_image_size,
            mask_input_size=self._infer_mask_input_size(),
        )
        points = None
        if scaled.points is not None and scaled.point_labels is not None:
            points = (scaled.points, scaled.point_labels)
        kwargs = {
            "points": points,
            "boxes": scaled.boxes,
            "masks": scaled.mask_input,
        }
        kwargs = {key: value for key, value in kwargs.items() if value is not None}
        try:
            return self.prompt_encoder(**kwargs)
        except Exception as exc:
            if kwargs.get("masks") is None or not self.allow_prompt_mask_retry:
                raise
            warnings.warn(
                "Native SAM/SAM2 prompt encoder rejected mask_input or failed with mask input "
                f"({type(exc).__name__}: {exc}); retrying without mask input.",
                stacklevel=2,
            )
            retry_kwargs = dict(kwargs)
            retry_kwargs["masks"] = None
            retry_kwargs = {key: value for key, value in retry_kwargs.items() if value is not None}
            try:
                return self.prompt_encoder(**retry_kwargs)
            except Exception as retry_exc:
                raise RuntimeError(
                    "Native SAM/SAM2 prompt encoder failed both with mask_input and without mask_input. "
                    f"With-mask error: {type(exc).__name__}: {exc}. "
                    f"Without-mask error: {type(retry_exc).__name__}: {retry_exc}."
                ) from retry_exc

    def _infer_mask_input_size(self) -> int:
        if self.prompt_encoder is None:
            return int(self.mask_input_size)
        direct = getattr(self.prompt_encoder, "mask_input_size", None)
        if isinstance(direct, int):
            return int(direct)
        if isinstance(direct, (tuple, list)) and direct:
            return int(direct[0])

        image_embedding_size = getattr(self.prompt_encoder, "image_embedding_size", None)
        if isinstance(image_embedding_size, int):
            return int(image_embedding_size * 4)
        if isinstance(image_embedding_size, (tuple, list)) and image_embedding_size:
            return int(image_embedding_size[0] * 4)

        input_image_size = getattr(self.prompt_encoder, "input_image_size", None)
        if isinstance(input_image_size, int):
            return int(input_image_size)
        if isinstance(input_image_size, (tuple, list)) and input_image_size:
            return int(input_image_size[0])

        kernel_size = getattr(getattr(self.prompt_encoder, "mask_downscaling", None), "kernel_size", None)
        if isinstance(kernel_size, int) and kernel_size > 0:
            return int(self.native_image_size // kernel_size)
        if isinstance(kernel_size, (tuple, list)) and kernel_size:
            return int(self.native_image_size // kernel_size[0])
        return int(self.mask_input_size)

    def _extract_mask_logits(self, mask_output: Any) -> torch.Tensor:
        if isinstance(mask_output, torch.Tensor):
            return mask_output
        if isinstance(mask_output, dict):
            for key in ["low_res_masks", "masks", "mask_logits", "pred_masks"]:
                if key in mask_output and isinstance(mask_output[key], torch.Tensor):
                    return mask_output[key]
        if isinstance(mask_output, (tuple, list)):
            for item in mask_output:
                if isinstance(item, torch.Tensor) and item.dim() >= 4:
                    return item
                if isinstance(item, dict):
                    try:
                        return self._extract_mask_logits(item)
                    except (TypeError, ValueError):
                        continue
        raise TypeError("Could not parse mask logits from native SAM/SAM2 decoder output.")

    def predict_mask_prior(
        self,
        image_embeddings: torch.Tensor | None,
        high_res_features: list[torch.Tensor] | None,
        prompts: PromptBundle,
        original_size: tuple[int, int],
        output_size: tuple[int, int],
    ) -> torch.Tensor:
        if image_embeddings is None:
            raise RuntimeError("Native SAM/SAM2 mask decoding requires image embeddings.")
        if self.prompt_encoder is None or self.mask_decoder is None:
            raise RuntimeError("Prompt encoder and mask decoder are required for native mask decoding.")

        sparse_embeddings, dense_embeddings = self._call_prompt_encoder(prompts, original_size=original_size)
        dense_pe = self.prompt_encoder.get_dense_pe() if hasattr(self.prompt_encoder, "get_dense_pe") else None

        kwargs = {
            "image_embeddings": image_embeddings,
            "image_pe": dense_pe,
            "sparse_prompt_embeddings": sparse_embeddings,
            "dense_prompt_embeddings": dense_embeddings,
            "multimask_output": False,
        }
        if high_res_features is not None:
            kwargs["high_res_features"] = high_res_features
        try:
            output = self.mask_decoder(**kwargs)
        except TypeError:
            kwargs.pop("high_res_features", None)
            output = self.mask_decoder(**kwargs)
        mask_logits = self._extract_mask_logits(output)
        if mask_logits.dim() == 3:
            mask_logits = mask_logits.unsqueeze(1)
        if mask_logits.shape[1] > 1:
            mask_logits = mask_logits[:, :1]
        return F.interpolate(mask_logits, size=output_size, mode="bilinear", align_corners=False)


class FrozenSAMEncoder(BaseFrozenEncoder, _SAMNativeMixin):
    def __init__(
        self,
        checkpoint_path: str,
        model_type: str = "vit_b",
        freeze: bool = True,
        image_size: int = 1024,
        output_channels: tuple[int, int, int, int] = (256, 256, 256, 256),
        mask_input_size: int = 256,
        allow_prompt_mask_retry: bool = True,
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = True,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
    ) -> None:
        super().__init__(freeze=freeze)
        self.native_image_size = image_size
        self.mask_input_size = mask_input_size
        self.allow_prompt_mask_retry = allow_prompt_mask_retry
        self.sam_model = self._build_sam(model_type=model_type, checkpoint_path=checkpoint_path)
        self.image_encoder = self.sam_model.image_encoder
        self.prompt_encoder = getattr(self.sam_model, "prompt_encoder", None)
        self.mask_decoder = getattr(self.sam_model, "mask_decoder", None)
        self.feature_channels = [256, 256, 256, 256]
        self.output_channels = list(output_channels)
        self.projections = nn.ModuleList(
            [
                nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(self.feature_channels, self.output_channels)
            ]
        )
        self.register_buffer("normalize_mean", torch.tensor(normalize_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("normalize_std", torch.tensor(normalize_std).view(1, 3, 1, 1), persistent=False)
        if self.freeze:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False
            self._set_frozen_eval_mode()
        self._freeze_optional_module(self.prompt_encoder, freeze_prompt_encoder)
        self._freeze_optional_module(self.mask_decoder, freeze_mask_decoder)

    def _build_sam(self, model_type: str, checkpoint_path: str) -> nn.Module:
        try:
            from segment_anything import sam_model_registry  # type: ignore
        except ImportError as exc:
            raise ImportError("Segment Anything is not installed. Install it or set model.encoder_type=resnet.") from exc
        return sam_model_registry[model_type](checkpoint=checkpoint_path)

    def _set_frozen_eval_mode(self) -> None:
        self.image_encoder.eval()

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError("FrozenSAMEncoder supports only 1-channel or 3-channel inputs.")
        x = F.interpolate(x, size=(self.native_image_size, self.native_image_size), mode="bilinear", align_corners=False)
        return (x - self.normalize_mean) / self.normalize_std

    def _build_pseudo_pyramid(self, feature: torch.Tensor) -> list[torch.Tensor]:
        p4 = feature
        p3 = F.interpolate(feature, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = F.interpolate(feature, scale_factor=4, mode="bilinear", align_corners=False)
        p1 = F.interpolate(feature, scale_factor=8, mode="bilinear", align_corners=False)
        return [p1, p2, p3, p4]

    @logged_call()
    @validate_tensor_output
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        x = self._prepare_input(x)
        if self.freeze:
            with torch.no_grad():
                image_embeddings = self.image_encoder(x)
        else:
            image_embeddings = self.image_encoder(x)
        if isinstance(image_embeddings, dict):
            image_embeddings = image_embeddings.get("image_embeddings", image_embeddings.get("features"))
        if not isinstance(image_embeddings, torch.Tensor):
            raise TypeError("SAM image encoder did not return a tensor image embedding.")
        pyramid = [projection(feature) for projection, feature in zip(self.projections, self._build_pseudo_pyramid(image_embeddings))]
        return {
            "features": pyramid[-1],
            "pyramid": pyramid,
            "image_embeddings": image_embeddings,
            "high_res_features": None,
            "sam_outputs": image_embeddings,
        }


class FrozenSAM2Encoder(BaseFrozenEncoder, _SAMNativeMixin):
    def __init__(
        self,
        model_cfg: str,
        checkpoint_path: str,
        freeze: bool = True,
        image_size: int = 1024,
        feature_key: str = "backbone_fpn",
        feature_format: str = "auto",
        out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
        feature_channels: tuple[int, int, int, int] = (256, 256, 256, 256),
        output_channels: tuple[int, int, int, int] = (256, 256, 256, 256),
        projection_trainable: bool = True,
        mask_input_size: int = 256,
        allow_prompt_mask_retry: bool = True,
        normalize_mean: tuple[float, float, float] = (0.485, 0.456, 0.406),
        normalize_std: tuple[float, float, float] = (0.229, 0.224, 0.225),
        freeze_prompt_encoder: bool = True,
        freeze_mask_decoder: bool = True,
    ) -> None:
        super().__init__(freeze=freeze)
        if not model_cfg or not checkpoint_path:
            raise ValueError("SAM2 encoder requires both model_cfg and checkpoint_path.")
        self.native_image_size = image_size
        self.mask_input_size = mask_input_size
        self.allow_prompt_mask_retry = allow_prompt_mask_retry
        self.model_cfg = model_cfg
        self.checkpoint_path = checkpoint_path
        self.feature_key = feature_key
        self.feature_format = feature_format
        self.out_indices = tuple(out_indices)
        self.feature_channels = list(feature_channels)
        self.output_channels = list(output_channels)
        self.projection_trainable = projection_trainable
        self._warned_format = False

        if len(self.feature_channels) != 4 or len(self.output_channels) != 4:
            raise ValueError("FrozenSAM2Encoder expects exactly four feature_channels and output_channels values.")

        self.sam2_model = self._build_sam2(model_cfg, checkpoint_path)
        self.image_encoder = self._resolve_required_component(
            self.sam2_model,
            ["image_encoder", "model.image_encoder", "backbone"],
            error="Could not find SAM2 image encoder. Expected sam2_model.image_encoder, sam2_model.model.image_encoder, or sam2_model.backbone.",
        )
        self.prompt_encoder = self._resolve_optional_component(
            self.sam2_model,
            ["sam_prompt_encoder", "prompt_encoder", "model.prompt_encoder"],
        )
        self.mask_decoder = self._resolve_optional_component(
            self.sam2_model,
            ["sam_mask_decoder", "mask_decoder", "model.mask_decoder"],
        )
        self.projections = nn.ModuleList(
            [
                nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, kernel_size=1)
                for in_ch, out_ch in zip(self.feature_channels, self.output_channels)
            ]
        )
        self.register_buffer("normalize_mean", torch.tensor(normalize_mean).view(1, 3, 1, 1), persistent=False)
        self.register_buffer("normalize_std", torch.tensor(normalize_std).view(1, 3, 1, 1), persistent=False)

        if self.freeze:
            for parameter in self.image_encoder.parameters():
                parameter.requires_grad = False
            self._set_frozen_eval_mode()
        if not projection_trainable:
            for parameter in self.projections.parameters():
                parameter.requires_grad = False
            self.projections.eval()
        self._freeze_optional_module(self.prompt_encoder, freeze_prompt_encoder)
        self._freeze_optional_module(self.mask_decoder, freeze_mask_decoder)

    def _build_sam2(self, model_cfg: str, checkpoint_path: str) -> nn.Module:
        try:
            from sam2.build_sam import build_sam2  # type: ignore
        except ImportError:
            try:
                from sam2.build_sam2 import build_sam2  # type: ignore
            except ImportError as exc:
                raise ImportError("SAM2 is not installed. Install SAM2 or set model.encoder_type=resnet.") from exc
        return build_sam2(model_cfg, checkpoint_path)

    def _resolve_optional_component(self, root: Any, paths: list[str]) -> nn.Module | None:
        for path in paths:
            value = root
            for attr in path.split("."):
                value = getattr(value, attr, None)
                if value is None:
                    break
            if value is not None:
                return value
        return None

    def _resolve_required_component(self, root: Any, paths: list[str], error: str) -> nn.Module:
        value = self._resolve_optional_component(root, paths)
        if value is None:
            raise RuntimeError(error)
        return value

    def _set_frozen_eval_mode(self) -> None:
        self.image_encoder.eval()

    def train(self, mode: bool = True) -> "FrozenSAM2Encoder":
        super().train(mode)
        if not self.projection_trainable:
            self.projections.eval()
        if self.prompt_encoder is not None and all(not parameter.requires_grad for parameter in self.prompt_encoder.parameters()):
            self.prompt_encoder.eval()
        if self.mask_decoder is not None and all(not parameter.requires_grad for parameter in self.mask_decoder.parameters()):
            self.mask_decoder.eval()
        return self

    def _prepare_input(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 4:
            raise ValueError(f"Expected input [B,C,H,W], got {tuple(x.shape)}")
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        elif x.shape[1] != 3:
            raise ValueError("FrozenSAM2Encoder supports only 1-channel or 3-channel inputs.")
        x = F.interpolate(x, size=(self.native_image_size, self.native_image_size), mode="bilinear", align_corners=False)
        return (x - self.normalize_mean) / self.normalize_std

    def _ensure_nchw(self, tensor: torch.Tensor) -> torch.Tensor:
        if tensor.dim() != 4:
            raise ValueError(f"Expected SAM2 feature map to be 4D, got {tuple(tensor.shape)}")
        if self.feature_format == "nchw":
            return tensor
        if self.feature_format == "nhwc":
            return tensor.permute(0, 3, 1, 2).contiguous()

        known_channels = set(self.feature_channels + self.output_channels + [32, 64, 128, 192, 256, 512, 768, 1024])
        if tensor.shape[1] in known_channels and tensor.shape[-1] not in known_channels:
            return tensor
        if tensor.shape[-1] in known_channels and tensor.shape[1] not in known_channels:
            return tensor.permute(0, 3, 1, 2).contiguous()
        if tensor.shape[1] <= tensor.shape[-1] and tensor.shape[-1] in known_channels and tensor.shape[1] < 16:
            return tensor.permute(0, 3, 1, 2).contiguous()
        if not self._warned_format:
            warnings.warn("Could not confidently infer SAM2 feature format. Assuming NCHW.", stacklevel=2)
            self._warned_format = True
        return tensor

    def _parse_feature_container(self, value: Any) -> list[torch.Tensor]:
        if isinstance(value, torch.Tensor):
            return [self._ensure_nchw(value)]
        if isinstance(value, (list, tuple)):
            return [self._ensure_nchw(item) for item in value if isinstance(item, torch.Tensor)]
        if isinstance(value, dict):
            return [self._ensure_nchw(value[key]) for key in sorted(value) if isinstance(value[key], torch.Tensor)]
        return []

    def _parse_sam2_output(self, raw_output: Any) -> dict[str, Any]:
        candidate_features: list[torch.Tensor] = []
        image_embeddings: torch.Tensor | None = None
        high_res_features: list[torch.Tensor] | None = None

        if isinstance(raw_output, torch.Tensor):
            image_embeddings = self._ensure_nchw(raw_output)
        elif isinstance(raw_output, dict):
            priority_keys = [self.feature_key, "backbone_fpn", "fpn_features", "high_res_features", "vision_features", "image_embeddings", "features"]
            for key in priority_keys:
                if key and key in raw_output:
                    parsed = self._parse_feature_container(raw_output[key])
                    if parsed:
                        candidate_features = parsed
                        break
            if "high_res_features" in raw_output:
                parsed_high = self._parse_feature_container(raw_output["high_res_features"])
                high_res_features = parsed_high or None
            if "vision_features" in raw_output and isinstance(raw_output["vision_features"], torch.Tensor):
                image_embeddings = self._ensure_nchw(raw_output["vision_features"])
            elif "image_embeddings" in raw_output and isinstance(raw_output["image_embeddings"], torch.Tensor):
                image_embeddings = self._ensure_nchw(raw_output["image_embeddings"])
            elif candidate_features:
                image_embeddings = candidate_features[-1]
        elif isinstance(raw_output, (list, tuple)):
            candidate_features = [self._ensure_nchw(item) for item in raw_output if isinstance(item, torch.Tensor)]
            if candidate_features:
                image_embeddings = candidate_features[-1]
        else:
            raise TypeError("Unable to parse SAM2 encoder output into usable features.")

        if not candidate_features and image_embeddings is not None:
            candidate_features = [image_embeddings]
        return {
            "candidate_features": candidate_features,
            "image_embeddings": image_embeddings,
            "high_res_features": high_res_features,
            "raw": raw_output,
        }

    def _spatial_area(self, feature: torch.Tensor) -> int:
        return int(feature.shape[-2] * feature.shape[-1])

    def _build_pseudo_pyramid(self, feature: torch.Tensor) -> list[torch.Tensor]:
        p4 = feature
        p3 = F.interpolate(feature, scale_factor=2, mode="bilinear", align_corners=False)
        p2 = F.interpolate(feature, scale_factor=4, mode="bilinear", align_corners=False)
        p1 = F.interpolate(feature, scale_factor=8, mode="bilinear", align_corners=False)
        return [p1, p2, p3, p4]

    def _select_four_level_pyramid(self, features: list[torch.Tensor]) -> list[torch.Tensor]:
        if not features:
            raise RuntimeError("SAM2 feature parser returned no features.")
        if len(features) == 1:
            return self._build_pseudo_pyramid(features[0])

        selected = features
        if len(features) >= 4:
            if any(index < 0 or index >= len(features) for index in self.out_indices):
                raise ValueError(f"Invalid SAM2 out_indices {self.out_indices} for {len(features)} available features.")
            selected = [features[index] for index in self.out_indices]

        selected = sorted(selected, key=self._spatial_area, reverse=True)
        if len(selected) == 2:
            p1, p3 = selected
            p2 = F.interpolate(p3, size=(max((p1.shape[-2] + p3.shape[-2]) // 2, 1), max((p1.shape[-1] + p3.shape[-1]) // 2, 1)), mode="bilinear", align_corners=False)
            p4 = F.interpolate(p3, size=(max(p3.shape[-2] // 2, 1), max(p3.shape[-1] // 2, 1)), mode="bilinear", align_corners=False)
            return [p1, p2, p3, p4]
        if len(selected) == 3:
            p1, p2, p3 = selected
            p4 = F.interpolate(p3, size=(max(p3.shape[-2] // 2, 1), max(p3.shape[-1] // 2, 1)), mode="bilinear", align_corners=False)
            return [p1, p2, p3, p4]
        if len(selected) > 4:
            selected = selected[:4]
        return selected

    def _validate_channels(self, pyramid: list[torch.Tensor]) -> None:
        actual = [int(feature.shape[1]) for feature in pyramid]
        expected = list(self.feature_channels)
        if actual != expected:
            raise RuntimeError(
                "SAM2 feature channel mismatch. "
                f"Expected {expected}, got {actual}. Run scripts/inspect_sam2_features.py and update model.sam2_feature_channels."
            )

    def _extract_features(self, x: torch.Tensor) -> dict[str, Any]:
        if self.freeze:
            with torch.no_grad():
                raw_output = self.image_encoder(x)
        else:
            raw_output = self.image_encoder(x)
        parsed = self._parse_sam2_output(raw_output)
        pyramid = self._select_four_level_pyramid(parsed["candidate_features"])
        self._validate_channels(pyramid)
        projected = [projection(feature) for projection, feature in zip(self.projections, pyramid)]
        return {
            "features": projected[-1],
            "pyramid": projected,
            "image_embeddings": parsed["image_embeddings"],
            "high_res_features": parsed["high_res_features"],
            "sam_outputs": parsed["raw"],
        }

    @logged_call()
    @validate_tensor_output
    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor | list[torch.Tensor] | None]:
        x = self._prepare_input(x)
        return self._extract_features(x)


FrozenEncoder = FrozenResNetEncoder
