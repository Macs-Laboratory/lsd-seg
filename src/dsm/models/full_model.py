from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any
import warnings

import torch
import torch.nn.functional as F
from torch import nn

from ..prompts.auto_prompt import AutomaticPromptGenerator
from ..utils.decorators import capture_peak_memory, collect_metrics, logged_call, reset_runtime_metrics, timed, validate_tensor_output
from .adapters import PrototypeHyperNetwork, StaticPrototypeConditioner
from .backbones import FrozenResNetEncoder, FrozenSAM2Encoder, FrozenSAMEncoder
from .decoder import FiLMExpertDecoder
from .descriptor import SubdomainDescriptorHead
from .prototype import PrototypeMemoryBank
from .routing import PrototypeRouter, UncertaintyTemperedRouter


@dataclass(slots=True)
class MaskPriorResolution:
    """Structured prior-selection payload returned by _resolve_mask_prior."""

    prompt_prior_logits: torch.Tensor
    sam_native_prior_logits: torch.Tensor | None
    decoder_prior_logits: torch.Tensor | None
    sam_native_prior_used: bool
    sam_decoder_fallback_used: bool
    sam_decode_attempted: bool

    def __getitem__(self, key: str) -> Any:
        return getattr(self, key)

    def items(self):
        return self.to_dict().items()

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_prior_logits": self.prompt_prior_logits,
            "sam_native_prior_logits": self.sam_native_prior_logits,
            "decoder_prior_logits": self.decoder_prior_logits,
            "sam_native_prior_used": self.sam_native_prior_used,
            "sam_decoder_fallback_used": self.sam_decoder_fallback_used,
            "sam_decode_attempted": self.sam_decode_attempted,
        }


class LSDSeg(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        num_classes: int = 1,
        encoder_dim: int = 256,
        descriptor_dim: int = 256,
        decoder_channels: Sequence[int] = (256, 128, 64, 32),
        max_prototypes: int = 16,
        init_threshold: float = 0.65,
        merge_threshold: float = 0.95,
        ema_momentum: float = 0.95,
        routing_tau: float = 0.07,
        uncertainty_tempering_enabled: bool = True,
        uncertainty_tempering_mode: str = "sample",
        uncertainty_alpha: float = 4.0,
        freeze_encoder: bool = True,
        dynamic_prototypes: bool = True,
        fixed_k: int | None = None,
        merge_enabled: bool = True,
        hypernetwork_enabled: bool = True,
        learnable_tau: bool = False,
        min_support: int = 3,
        merge_interval: int = 100,
        warmup_steps: int = 0,
        use_roi_descriptor: bool = True,
        descriptor_source: str = "deepest",
        encoder_type: str = "resnet",
        decoder_type: str = "film",
        use_sam_mask_prior: bool = False,
        allow_sam_decoder_fallback: bool = True,
        encoder_name: str = "resnet18",
        encoder_pretrained: bool = True,
        encoder_checkpoint_path: str | None = None,
        sam_checkpoint_path: str | None = None,
        sam_model_type: str = "vit_b",
        sam_image_size: int = 1024,
        sam_mask_input_size: int = 256,
        sam2_model_cfg: str | None = None,
        sam2_checkpoint_path: str | None = None,
        sam2_image_size: int = 1024,
        sam2_feature_key: str = "backbone_fpn",
        sam2_feature_format: str = "auto",
        sam2_out_indices: Sequence[int] = (0, 1, 2, 3),
        sam2_feature_channels: Sequence[int] = (256, 256, 256, 256),
        sam2_output_channels: Sequence[int] = (256, 256, 256, 256),
        sam2_projection_trainable: bool = True,
        sam2_mask_input_size: int = 256,
        sam2_normalize_mean: Sequence[float] = (0.485, 0.456, 0.406),
        sam2_normalize_std: Sequence[float] = (0.229, 0.224, 0.225),
        sam2_freeze_prompt_encoder: bool = True,
        sam2_freeze_mask_decoder: bool = True,
        prompt_channels: int = 1,
        prompt_num_points: int = 4,
    ) -> None:
        super().__init__()
        self.runtime_metrics: dict[str, float] = {}
        self.num_classes = num_classes
        self.hypernetwork_enabled = hypernetwork_enabled
        self.uncertainty_tempering_mode = uncertainty_tempering_mode
        self.use_uncertainty_tempering = uncertainty_tempering_enabled
        self.encoder_type = encoder_type
        self.decoder_type = decoder_type
        self.use_sam_mask_prior = use_sam_mask_prior
        self.allow_sam_decoder_fallback = allow_sam_decoder_fallback
        if decoder_type == "sam_mask" and hypernetwork_enabled:
            warnings.warn(
                "decoder_type='sam_mask' bypasses the prototype-conditioned FiLM decoder. "
                "This mode is intended only for native SAM/SAM2 baseline or sanity checks.",
                stacklevel=2,
            )

        if encoder_type == "resnet":
            self.encoder = FrozenResNetEncoder(
                encoder_name=encoder_name,
                in_channels=in_channels,
                pretrained=encoder_pretrained,
                freeze=freeze_encoder,
                output_channels=[64, 128, 256, encoder_dim],
                checkpoint_path=encoder_checkpoint_path,
            )
        elif encoder_type == "sam":
            if not sam_checkpoint_path:
                raise ValueError("encoder_type='sam' requires sam_checkpoint_path.")
            self.encoder = FrozenSAMEncoder(
                checkpoint_path=sam_checkpoint_path,
                model_type=sam_model_type,
                freeze=freeze_encoder,
                image_size=sam_image_size,
                output_channels=tuple(int(channel) for channel in sam2_output_channels),
                mask_input_size=sam_mask_input_size,
                allow_prompt_mask_retry=allow_sam_decoder_fallback,
                freeze_prompt_encoder=sam2_freeze_prompt_encoder,
                freeze_mask_decoder=sam2_freeze_mask_decoder,
            )
        elif encoder_type == "sam2":
            if not sam2_model_cfg or not sam2_checkpoint_path:
                raise ValueError("encoder_type='sam2' requires sam2_model_cfg and sam2_checkpoint_path.")
            self.encoder = FrozenSAM2Encoder(
                model_cfg=sam2_model_cfg,
                checkpoint_path=sam2_checkpoint_path,
                freeze=freeze_encoder,
                image_size=sam2_image_size,
                feature_key=sam2_feature_key,
                feature_format=sam2_feature_format,
                out_indices=tuple(int(index) for index in sam2_out_indices),
                feature_channels=tuple(int(channel) for channel in sam2_feature_channels),
                output_channels=tuple(int(channel) for channel in sam2_output_channels),
                projection_trainable=sam2_projection_trainable,
                mask_input_size=sam2_mask_input_size,
                allow_prompt_mask_retry=allow_sam_decoder_fallback,
                normalize_mean=tuple(float(value) for value in sam2_normalize_mean),
                normalize_std=tuple(float(value) for value in sam2_normalize_std),
                freeze_prompt_encoder=sam2_freeze_prompt_encoder,
                freeze_mask_decoder=sam2_freeze_mask_decoder,
            )
        else:
            raise ValueError(f"Unsupported encoder_type '{encoder_type}'. Expected 'resnet', 'sam', or 'sam2'.")

        pyramid_channels = list(self.encoder.output_channels)
        deepest_channels = int(pyramid_channels[-1])
        self.prompt_generator = AutomaticPromptGenerator(
            in_channels=deepest_channels,
            prompt_channels=max(prompt_channels, num_classes),
            num_points=prompt_num_points,
        )
        self.descriptor_head = SubdomainDescriptorHead(
            feature_dim=deepest_channels,
            descriptor_dim=descriptor_dim,
            use_roi=use_roi_descriptor,
            normalize=True,
            descriptor_source=descriptor_source,
            pyramid_channels=pyramid_channels,
        )
        self.prototype_bank = PrototypeMemoryBank(
            descriptor_dim=descriptor_dim,
            max_prototypes=max_prototypes,
            novelty_threshold=init_threshold,
            merge_threshold=merge_threshold,
            ema_momentum=ema_momentum,
            min_support=min_support,
            merge_interval=merge_interval,
            warmup_steps=warmup_steps,
            dynamic_prototypes=dynamic_prototypes,
            fixed_k=fixed_k,
            merge_enabled=merge_enabled,
        )
        self.prototype_memory = self.prototype_bank
        self.prototype_router = PrototypeRouter(tau=routing_tau, learnable_tau=learnable_tau)
        self.router = UncertaintyTemperedRouter(
            base_temperature=routing_tau,
            uncertainty_scale=uncertainty_alpha,
            use_uncertainty_tempering=uncertainty_tempering_enabled,
        )
        if hypernetwork_enabled:
            self.adapter_generator = PrototypeHyperNetwork(
                prototype_dim=descriptor_dim,
                decoder_channels=decoder_channels,
            )
        else:
            self.adapter_generator = StaticPrototypeConditioner(
                max_prototypes=max_prototypes,
                decoder_channels=decoder_channels,
            )
        self.hypernetwork = self.adapter_generator
        self.decoder = FiLMExpertDecoder(
            in_channels=deepest_channels,
            pyramid_channels=pyramid_channels,
            decoder_channels=decoder_channels,
            num_classes=num_classes,
            prompt_channels=1,
        )

    def export_artifacts(self) -> dict[str, Any]:
        return {"model_state_dict": self.state_dict()}

    def load_artifacts(self, payload: dict[str, Any]) -> None:
        state_dict = payload["model_state_dict"] if "model_state_dict" in payload else payload
        self.load_state_dict(state_dict)

    def prototype_summary(self) -> dict[str, Any]:
        return self.prototype_bank.summary()

    def _resolve_mask_prior(
        self,
        encoder_output: dict[str, Any],
        prompts,
        original_size: tuple[int, int],
    ) -> MaskPriorResolution:
        prompt_prior_logits = F.interpolate(prompts.coarse_logits, size=original_size, mode="bilinear", align_corners=False)

        decoder_accepts_prior = self.decoder_type in {"hybrid", "sam_mask"}

        def build(
            *,
            sam_native_prior_logits: torch.Tensor | None,
            decoder_prior_logits: torch.Tensor | None,
            sam_native_prior_used: bool,
            sam_decoder_fallback_used: bool,
            sam_decode_attempted: bool,
        ) -> MaskPriorResolution:
            return MaskPriorResolution(
                prompt_prior_logits=prompt_prior_logits,
                sam_native_prior_logits=sam_native_prior_logits,
                decoder_prior_logits=decoder_prior_logits,
                sam_native_prior_used=sam_native_prior_used,
                sam_decoder_fallback_used=sam_decoder_fallback_used,
                sam_decode_attempted=sam_decode_attempted,
            )

        if self.encoder_type not in {"sam", "sam2"}:
            return build(
                sam_native_prior_logits=None,
                decoder_prior_logits=prompt_prior_logits if decoder_accepts_prior else None,
                sam_native_prior_used=False,
                sam_decoder_fallback_used=False,
                sam_decode_attempted=False,
            )

        if not self.use_sam_mask_prior:
            return build(
                sam_native_prior_logits=None,
                decoder_prior_logits=prompt_prior_logits if decoder_accepts_prior else None,
                sam_native_prior_used=False,
                sam_decoder_fallback_used=False,
                sam_decode_attempted=False,
            )

        if not self.encoder.has_native_mask_decoder():
            if self.allow_sam_decoder_fallback:
                warnings.warn("Native SAM/SAM2 decoder components are unavailable. Falling back to coarse prompt prior.", stacklevel=2)
                return build(
                    sam_native_prior_logits=None,
                    decoder_prior_logits=prompt_prior_logits if decoder_accepts_prior else None,
                    sam_native_prior_used=False,
                    sam_decoder_fallback_used=True,
                    sam_decode_attempted=True,
                )
            raise RuntimeError("Native SAM/SAM2 decoder components are unavailable and fallback is disabled.")
        try:
            native_prior = self.encoder.predict_mask_prior(
                image_embeddings=encoder_output.get("image_embeddings"),
                high_res_features=encoder_output.get("high_res_features"),
                prompts=prompts,
                original_size=original_size,
                output_size=original_size,
            )
            return build(
                sam_native_prior_logits=native_prior,
                decoder_prior_logits=native_prior,
                sam_native_prior_used=True,
                sam_decoder_fallback_used=False,
                sam_decode_attempted=True,
            )
        except Exception as exc:
            if not self.allow_sam_decoder_fallback:
                raise
            warnings.warn(f"Native SAM/SAM2 decoder call failed ({exc}). Falling back to coarse prompt prior.", stacklevel=2)
            return build(
                sam_native_prior_logits=None,
                decoder_prior_logits=prompt_prior_logits if decoder_accepts_prior else None,
                sam_native_prior_used=False,
                sam_decoder_fallback_used=True,
                sam_decode_attempted=True,
            )

    @reset_runtime_metrics
    @collect_metrics("forward_calls")
    @capture_peak_memory
    @timed("forward_seconds")
    @logged_call()
    @validate_tensor_output
    def forward(
        self,
        x: torch.Tensor,
        targets: torch.Tensor | None = None,
        update_prototypes: bool = True,
        training: bool | None = None,
    ) -> dict[str, torch.Tensor | int | None]:
        if x.dim() != 4:
            raise ValueError(f"Expected x [B,C,H,W], got {tuple(x.shape)}")
        original_size = tuple(int(v) for v in x.shape[-2:])
        is_training = self.training if training is None else bool(training)
        should_update = bool(update_prototypes and is_training)

        encoder_output = self.encoder(x)
        features = encoder_output["features"]
        pyramid = encoder_output["pyramid"]
        image_embeddings = encoder_output.get("image_embeddings")
        high_res_features = encoder_output.get("high_res_features")

        prompts = self.prompt_generator(features, original_size=original_size)
        descriptors = self.descriptor_head(
            features=features,
            prompts=prompts,
            pyramid=pyramid,
            original_size=original_size,
        )

        prototype_out = self.prototype_bank(descriptors.detach(), update=should_update)
        prototype_embeddings = prototype_out["embeddings"]
        assignments = prototype_out["assignments"]
        if int(prototype_out["num_active"]) < 1:
            raise RuntimeError("No active prototypes found. Run training or load prototype memory first.")

        route_base = self.prototype_router(descriptors, prototype_embeddings)
        similarities = route_base["similarities"]
        base_routing_weights = route_base["weights"]
        adapter_parameters = self.adapter_generator(prototype_embeddings)

        prior_resolution = self._resolve_mask_prior(
            encoder_output=encoder_output,
            prompts=prompts,
            original_size=original_size,
        )
        prompt_prior_logits = prior_resolution.prompt_prior_logits
        decoder_prior_logits = prior_resolution.decoder_prior_logits
        sam_native_prior_logits = prior_resolution.sam_native_prior_logits

        if self.decoder_type not in {"film", "hybrid", "sam_mask"}:
            raise ValueError(f"Unsupported decoder_type '{self.decoder_type}'. Expected 'film', 'hybrid', or 'sam_mask'.")

        expert_logits = []
        if self.decoder_type == "sam_mask":
            # Baseline-only mode: every routed expert receives the same native/fallback prior.
            # This does not create prototype-conditioned expert diversity and should not be used as the main method.
            if decoder_prior_logits is None:
                raise RuntimeError("decoder_type='sam_mask' requires a native or fallback prompt prior.")
            for _ in range(int(prototype_out["num_active"])):
                expert_logits.append(decoder_prior_logits)
        else:
            for expert_index in range(int(prototype_out["num_active"])):
                film_params = adapter_parameters.for_expert(expert_index)
                expert_logits.append(
                    self.decoder(
                        features=features,
                        pyramid=pyramid,
                        film_params=film_params,
                        output_size=original_size,
                        prompts=prompts,
                        prompt_map=decoder_prior_logits,
                    )
                )
        expert_logits = torch.stack(expert_logits, dim=1)
        expert_probs = torch.sigmoid(expert_logits)

        router_out = self.router(
            similarities=similarities,
            expert_probs=expert_probs,
            mode=self.uncertainty_tempering_mode,
        )
        routing_weights = router_out["weights"]
        if routing_weights.dim() == 2:
            final_prob = (routing_weights[:, :, None, None, None] * expert_probs).sum(dim=1)
        else:
            final_prob = (routing_weights[:, :, None, :, :] * expert_probs).sum(dim=1)
        final_prob = final_prob.clamp(1e-6, 1.0 - 1e-6)
        final_logits = torch.logit(final_prob)

        weight_sum = routing_weights.sum(dim=1)
        if not torch.allclose(weight_sum, torch.ones_like(weight_sum), atol=1e-4):
            raise RuntimeError("Routing weights must sum to 1 over active prototypes.")

        return {
            "logits": final_logits,
            "prob": final_prob,
            "probabilities": final_prob,
            "expert_logits": expert_logits,
            "expert_probs": expert_probs,
            "expert_probabilities": expert_probs,
            "descriptor": descriptors,
            "descriptors": descriptors,
            "similarities": similarities,
            "base_similarity": similarities,
            "assignments": assignments,
            "assigned_ids": assignments,
            "routing_weights": routing_weights,
            "base_routing_weights": base_routing_weights,
            "temperatures": router_out["temperature"],
            "routing_entropy": router_out["routing_entropy"],
            "uncertainty": router_out["uncertainty"],
            "prototype_embeddings": prototype_embeddings,
            "prototypes": prototype_embeddings,
            "prototype_count": int(prototype_out["num_active"]),
            "num_active_prototypes": int(prototype_out["num_active"]),
            "max_similarity": prototype_out["max_similarity"],
            "prompt_boxes": prompts.boxes,
            "prompt_points": prompts.points,
            "prompt_labels": prompts.point_labels,
            "prompt_logits": prompts.coarse_logits,
            "prompt_coarse_logits": prompts.coarse_logits,
            "prompt_prior_logits": prompt_prior_logits,
            "decoder_prior_logits": decoder_prior_logits,
            "prompts_boxes": prompts.boxes,
            "sam_image_embeddings": image_embeddings,
            "sam_native_prior_logits": sam_native_prior_logits,
            "sam_mask_prior_logits": sam_native_prior_logits,
            "sam_high_res_features": high_res_features,
            "sam_native_prior_used": torch.tensor(float(prior_resolution.sam_native_prior_used), device=x.device),
            "sam_decoder_fallback_used": torch.tensor(float(prior_resolution.sam_decoder_fallback_used), device=x.device),
            "sam_mask_prior_available": torch.tensor(float(sam_native_prior_logits is not None), device=x.device),
            "sam_decode_attempted": torch.tensor(float(prior_resolution.sam_decode_attempted), device=x.device),
        }


class DynamicSubdomainModel(LSDSeg):
    def __init__(self, config: dict[str, Any] | None = None, **kwargs: Any) -> None:
        if config is not None:
            model_config = dict(config.get("model", config))
            model_config.update(kwargs)
        else:
            model_config = kwargs
        super().__init__(**model_config)
