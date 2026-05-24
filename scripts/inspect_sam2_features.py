from __future__ import annotations

import argparse

import torch

from dsm.models.backbones import FrozenSAM2Encoder
from dsm.prompts.auto_prompt import PromptBundle


def parse_int_tuple(value: str, expected_len: int | None = None) -> tuple[int, ...]:
    parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    if expected_len is not None and len(parsed) != expected_len:
        raise ValueError(f"Expected {expected_len} comma-separated integers, got {len(parsed)} from '{value}'.")
    return parsed


def inspect_encoder(
    model_cfg: str,
    checkpoint: str,
    image_size: int,
    device: str = "cpu",
    feature_key: str = "backbone_fpn",
    feature_format: str = "auto",
    out_indices: tuple[int, int, int, int] = (0, 1, 2, 3),
    feature_channels: tuple[int, int, int, int] = (256, 256, 256, 256),
    output_channels: tuple[int, int, int, int] = (256, 256, 256, 256),
    test_mask_decoder: bool = False,
    mask_input_size: int = 256,
    num_points: int = 4,
    strict: bool = False,
) -> dict[str, object]:
    torch_device = torch.device(device)
    encoder = FrozenSAM2Encoder(
        model_cfg=model_cfg,
        checkpoint_path=checkpoint,
        freeze=True,
        image_size=image_size,
        feature_key=feature_key,
        feature_format=feature_format,
        out_indices=out_indices,
        feature_channels=feature_channels,
        output_channels=output_channels,
        projection_trainable=False,
        mask_input_size=mask_input_size,
        allow_prompt_mask_retry=True,
    ).to(torch_device)

    dummy = torch.randn(1, 3, image_size, image_size, device=torch_device)
    with torch.no_grad():
        raw_output = encoder.image_encoder(dummy)
    print(f"raw output type: {type(raw_output).__name__}")
    if isinstance(raw_output, dict):
        print(f"dict keys: {list(raw_output.keys())}")
    parsed = encoder._parse_sam2_output(raw_output)
    for key in ["backbone_fpn", "fpn_features", "high_res_features", "vision_features", "image_embeddings", "features"]:
        if isinstance(raw_output, dict) and key in raw_output:
            value = raw_output[key]
            if isinstance(value, torch.Tensor):
                print(f"{key}: {tuple(value.shape)}")
            elif isinstance(value, (list, tuple)):
                print(f"{key}: {[tuple(item.shape) for item in value if isinstance(item, torch.Tensor)]}")
            elif isinstance(value, dict):
                print(f"{key}: { {sub_key: tuple(sub_value.shape) for sub_key, sub_value in value.items() if isinstance(sub_value, torch.Tensor)} }")

    candidate = parsed["candidate_features"]
    if candidate:
        selected = encoder._select_four_level_pyramid(candidate)
        recommended_channels = [int(feature.shape[1]) for feature in selected]
        print(f"candidate feature shapes: {[tuple(feature.shape) for feature in candidate]}")
    else:
        selected = []
        recommended_channels = [256, 256, 256, 256]

    result: dict[str, object] = {
        "has_prompt_encoder": encoder.prompt_encoder is not None,
        "has_mask_decoder": encoder.mask_decoder is not None,
        "sam_native_prior_succeeded": False,
        "sam_native_prior_shape": None,
        "sam_decoder_fallback_required": False,
        "mask_prior_succeeded": False,
        "mask_prior_shape": None,
        "fallback_required": False,
        "recommended_channels": recommended_channels,
    }

    if tuple(recommended_channels) != tuple(feature_channels):
        message = (
            "The configured --feature-channels do not match the extracted SAM2 feature channels. "
            f"Configured: {list(feature_channels)}. Reported: {recommended_channels}. "
            f"Re-run with --feature-channels {','.join(str(channel) for channel in recommended_channels)}."
        )
        print(message)
        result["sam_decoder_fallback_required"] = bool(result["sam_decoder_fallback_required"])
        if strict:
            raise ValueError(message)

    if test_mask_decoder:
        print(f"prompt_encoder exists: {encoder.prompt_encoder is not None}")
        print(f"mask_decoder exists: {encoder.mask_decoder is not None}")
        try:
            center_box = torch.tensor([[image_size * 0.25, image_size * 0.25, image_size * 0.75, image_size * 0.75]], device=torch_device)
            points = torch.tensor(
                [[[image_size * 0.5, image_size * 0.5]] * num_points],
                device=torch_device,
                dtype=torch.float32,
            )
            point_labels = torch.ones((1, num_points), device=torch_device)
            prompt = PromptBundle(
                boxes=center_box,
                points=points,
                point_labels=point_labels,
                coarse_logits=torch.zeros((1, 1, image_size // 4, image_size // 4), device=torch_device),
                mask_input=torch.zeros((1, 1, mask_input_size, mask_input_size), device=torch_device),
                original_size=(image_size, image_size),
                source_size=(image_size // 4, image_size // 4),
            )
            prior = encoder.predict_mask_prior(
                image_embeddings=parsed.get("image_embeddings"),
                high_res_features=parsed.get("high_res_features"),
                prompts=prompt,
                original_size=(image_size, image_size),
                output_size=(image_size, image_size),
            )
            result["sam_native_prior_succeeded"] = True
            result["sam_native_prior_shape"] = tuple(prior.shape)
            result["mask_prior_succeeded"] = True
            result["mask_prior_shape"] = tuple(prior.shape)
            print(f"native mask prior succeeded: {tuple(prior.shape)}")
        except Exception as exc:
            result["sam_decoder_fallback_required"] = True
            result["fallback_required"] = True
            print(f"native mask prior failed: {type(exc).__name__}: {exc}")
            if strict:
                raise

    print("recommended config:")
    print(f"  sam2_feature_key: {feature_key}")
    print(f"  sam2_feature_format: {feature_format}")
    print(f"  sam2_out_indices: {list(out_indices)}")
    print(f"  sam2_feature_channels: {recommended_channels}")
    print(f"  sam2_output_channels: {list(output_channels)}")
    print(f"  sam2_mask_input_size: {mask_input_size}")
    print("  use_sam_mask_prior: true")
    print("  allow_sam_decoder_fallback: true")
    if test_mask_decoder and not result["sam_native_prior_succeeded"]:
        print(
            "If sam_native_prior_succeeded=False and allow_sam_decoder_fallback=True, "
            "training will use prompt_prior_logits as decoder_prior_logits and evaluation "
            "will log sam_decoder_fallback_rate."
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect raw SAM2 feature outputs and optional native mask-decoder compatibility.")
    parser.add_argument("--model-cfg", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image-size", type=int, default=1024)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--feature-key", default="backbone_fpn")
    parser.add_argument("--feature-format", default="auto")
    parser.add_argument("--out-indices", default="0,1,2,3")
    parser.add_argument("--feature-channels", default="256,256,256,256")
    parser.add_argument("--output-channels", default="256,256,256,256")
    parser.add_argument("--test-mask-decoder", action="store_true")
    parser.add_argument("--mask-input-size", type=int, default=256)
    parser.add_argument("--num-points", type=int, default=4)
    parser.add_argument("--strict", action="store_true")
    args = parser.parse_args()

    inspect_encoder(
        model_cfg=args.model_cfg,
        checkpoint=args.checkpoint,
        image_size=args.image_size,
        device=args.device,
        feature_key=args.feature_key,
        feature_format=args.feature_format,
        out_indices=parse_int_tuple(args.out_indices, expected_len=4),
        feature_channels=parse_int_tuple(args.feature_channels, expected_len=4),
        output_channels=parse_int_tuple(args.output_channels, expected_len=4),
        test_mask_decoder=args.test_mask_decoder,
        mask_input_size=args.mask_input_size,
        num_points=args.num_points,
        strict=args.strict,
    )


if __name__ == "__main__":
    main()
