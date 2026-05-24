from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pandas as pd
import torch
from torch.optim import AdamW

from dsm.config import load_config
from dsm.engine.evaluator import Evaluator
from dsm.engine.trainer import Trainer
from dsm.losses import LSDSegLoss
from dsm.models.full_model import DynamicSubdomainModel
from dsm.models.prototype import PrototypeMemoryBank
from dsm.prompts.auto_prompt import PromptBundle, scale_prompts_to_sam_input
from inspect_sam2_features import inspect_encoder, parse_int_tuple


def build_criterion(config: dict) -> LSDSegLoss:
    return LSDSegLoss(
        expert_weight=config["loss"].get("expert_weight", config["loss"].get("lambda_expert", 0.2)),
        ortho_weight=config["loss"].get("ortho_weight", config["loss"].get("lambda_ortho", 0.01)),
        balance_weight=config["loss"].get("balance_weight", config["loss"].get("lambda_balance", 0.01)),
        proto_weight=config["loss"].get("proto_weight", 0.0),
        prompt_weight=config["loss"].get("prompt_weight", 0.1),
    )


def run_prompt_scaling_test() -> None:
    prompts = PromptBundle(
        boxes=torch.tensor([[32.0, 64.0, 128.0, 192.0]]),
        points=torch.tensor([[[32.0, 64.0], [128.0, 192.0]]]),
        point_labels=torch.tensor([[1.0, 1.0]]),
        coarse_logits=torch.zeros(1, 1, 64, 64),
        mask_input=torch.zeros(1, 1, 64, 64),
        original_size=(256, 256),
        source_size=(64, 64),
    )
    scaled = scale_prompts_to_sam_input(prompts, original_size=(256, 256), sam_image_size=1024, mask_input_size=256)
    assert torch.allclose(scaled.boxes, prompts.boxes * 4)
    assert torch.allclose(scaled.points, prompts.points * 4)
    assert scaled.mask_input is not None and scaled.mask_input.shape == (1, 1, 256, 256)
    print("Prompt scaling test passed.")


def run_parse_int_tuple_test() -> None:
    assert parse_int_tuple("0,1,2,3", expected_len=4) == (0, 1, 2, 3)
    assert parse_int_tuple("256, 512, 1024, 1024", expected_len=4) == (256, 512, 1024, 1024)
    print("parse_int_tuple test passed.")


def run_resnet_self_test() -> None:
    config = load_config("configs/default.yaml").raw
    config["model"]["freeze_encoder"] = False
    config["model"]["encoder_type"] = "resnet"
    config["model"]["encoder_pretrained"] = False
    config["model"]["max_prototypes"] = 4
    config["model"]["min_support"] = 1
    config["loss"]["prompt_weight"] = 0.1

    model = DynamicSubdomainModel(config).cpu()
    criterion = build_criterion(config)

    x = torch.randn(2, 1, 256, 256)
    y = torch.randint(0, 2, (2, 1, 256, 256)).float()

    model.train()
    out = model(x, targets=y, training=True, update_prototypes=True)
    assert "prompt_logits" in out or "prompt_coarse_logits" in out
    loss = criterion(out, y)
    loss["loss"].backward()

    assert out["probabilities"].shape == (2, 1, 256, 256)
    assert out["expert_logits"].dim() == 5
    assert int(out["prototype_count"]) >= 1
    assert torch.isfinite(out["probabilities"]).all()
    assert torch.isfinite(out["expert_logits"]).all()
    assert torch.isfinite(loss["loss_prompt"]).all()
    assert out["prompt_prior_logits"] is not None
    assert out["decoder_prior_logits"] is None
    assert out["sam_native_prior_logits"] is None
    assert out["sam_mask_prior_logits"] is None
    assert float(out["sam_mask_prior_available"].item()) == 0.0
    assert float(out["sam_native_prior_used"].item()) == 0.0
    assert float(out["sam_decoder_fallback_used"].item()) == 0.0
    assert torch.allclose(out["routing_weights"].sum(dim=1), torch.ones_like(out["routing_weights"].sum(dim=1)), atol=1e-4)
    assert "prototype_bank.embeddings" in model.state_dict()

    prototype_count_before = int(out["prototype_count"])
    model.eval()
    with torch.no_grad():
        out_eval = model(x, training=False, update_prototypes=False)
    assert out_eval["probabilities"].shape == (2, 1, 256, 256)
    assert int(out_eval["prototype_count"]) == prototype_count_before
    print("Dynamic sub-domain model self-test passed.")


def run_trainer_loss_prompt_test() -> None:
    config = load_config("configs/default.yaml").raw
    config["model"]["freeze_encoder"] = False
    config["model"]["encoder_type"] = "resnet"
    config["model"]["encoder_pretrained"] = False
    config["optimizer"]["epochs"] = 1
    config["loss"]["prompt_weight"] = 0.1
    model = DynamicSubdomainModel(config).cpu()
    trainer = Trainer(model=model, config=config, device=torch.device("cpu"))
    optimizer = AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config["optimizer"]["lr"],
        weight_decay=config["optimizer"]["weight_decay"],
    )
    loader = [
        {
            "image": torch.randn(2, 1, 128, 128),
            "mask": torch.randint(0, 2, (2, 1, 128, 128)).float(),
        }
    ]
    summary = trainer._train_epoch(loader, optimizer)
    assert "loss_prompt" in summary
    assert torch.isfinite(torch.tensor(summary["loss_prompt"]))
    print("Trainer prompt-loss aggregation test passed.")


def run_pixel_routing_test() -> None:
    config = load_config("configs/default.yaml").raw
    config["model"]["freeze_encoder"] = False
    config["model"]["encoder_type"] = "resnet"
    config["model"]["encoder_pretrained"] = False
    config["model"]["uncertainty_tempering_mode"] = "pixel"
    model = DynamicSubdomainModel(config).cpu()
    x = torch.randn(2, 1, 128, 128)
    model.train()
    out = model(x, training=True, update_prototypes=True)
    assert out["routing_weights"].dim() == 4
    assert torch.allclose(out["routing_weights"].sum(dim=1), torch.ones_like(out["routing_weights"].sum(dim=1)), atol=1e-4)
    print("Pixel routing test passed.")


def run_prototype_slot_test() -> None:
    bank = PrototypeMemoryBank(descriptor_dim=4, max_prototypes=4)
    with torch.no_grad():
        bank.maybe_create(torch.tensor([1.0, 0.0, 0.0, 0.0]))
        bank.maybe_create(torch.tensor([0.0, 1.0, 0.0, 0.0]))
        bank.maybe_create(torch.tensor([0.0, 0.0, 1.0, 0.0]))
        bank.active_mask[1] = False
        bank.counts[1] = 0
        original_third = bank.embeddings[2].clone()
        bank.maybe_create(torch.tensor([0.0, 0.0, 0.0, 1.0]))
        assert bank.active_mask[1]
        assert torch.allclose(bank.embeddings[2], original_third)
    print("Prototype slot reuse test passed.")


def run_hybrid_fallback_logging_test() -> None:
    config = load_config("configs/default.yaml").raw
    config["model"]["freeze_encoder"] = False
    config["model"]["encoder_type"] = "resnet"
    config["model"]["encoder_pretrained"] = False
    config["model"]["decoder_type"] = "hybrid"
    config["model"]["use_sam_mask_prior"] = True
    config["model"]["allow_sam_decoder_fallback"] = True
    config["evaluation"]["save_predictions"] = True
    model = DynamicSubdomainModel(config).cpu()
    model.encoder_type = "sam2"
    x = torch.randn(1, 1, 128, 128)
    model.train()
    with warnings.catch_warnings(record=True) as caught:
        out = model(x, training=True, update_prototypes=True)
    assert out["prompt_prior_logits"] is not None
    assert out["decoder_prior_logits"] is not None
    assert out["sam_native_prior_logits"] is None
    assert float(out["sam_decoder_fallback_used"].item()) == 1.0
    assert float(out["sam_mask_prior_available"].item()) == 0.0
    assert float(out["sam_native_prior_used"].item()) == 0.0
    assert any("Falling back" in str(item.message) for item in caught)

    loader = [
        {
            "image": x,
            "mask": torch.randint(0, 2, (1, 1, 128, 128)).float(),
            "sample_id": ["sample_0"],
            "subdomain_id": ["sub_0"],
        }
    ]
    with tempfile.TemporaryDirectory() as tmpdir:
        summary = Evaluator(model=model, device=torch.device("cpu"), config=config).evaluate(loader, tmpdir)
        df = pd.read_csv(Path(tmpdir) / "per_sample_metrics.csv")
        artifact_path = Path(tmpdir) / "sample_artifacts" / "sample_0.npz"
        if artifact_path.exists():
            import numpy as np

            with np.load(artifact_path) as artifact_payload:
                assert "prompt_prior_probability" in artifact_payload
                assert "decoder_prior_probability" in artifact_payload
        assert "sam_decoder_fallback_used" in df.columns
        assert "sam_native_prior_used" in df.columns
        assert "sam_mask_prior_available" in df.columns
        assert "sam_native_prior_mean" in df.columns
        assert "decoder_prior_mean" in df.columns
        assert "prompt_prior_mean" in df.columns
        assert "sam_mask_prior_mean" in df.columns
        assert "sam_decoder_fallback_rate" in summary
        assert "sam_native_prior_used_rate" in summary
        assert "sam_mask_prior_available_rate" in summary
    print("Hybrid fallback logging test passed.")


def run_optional_sam2_smoke() -> None:
    try:
        import sam2  # noqa: F401
    except Exception:
        print("SAM2 smoke test skipped: SAM2 is not installed.")
        return

    config = load_config("configs/default.yaml").raw
    if not config["model"].get("sam2_model_cfg") or not config["model"].get("sam2_checkpoint_path"):
        print("SAM2 smoke test skipped: sam2_model_cfg/checkpoint are not set in config.")
        return

    result = inspect_encoder(
        model_cfg=config["model"]["sam2_model_cfg"],
        checkpoint=config["model"]["sam2_checkpoint_path"],
        image_size=config["model"].get("sam2_image_size", 1024),
        device="cpu",
        feature_key=config["model"].get("sam2_feature_key", "backbone_fpn"),
        feature_format=config["model"].get("sam2_feature_format", "auto"),
        out_indices=tuple(config["model"].get("sam2_out_indices", [0, 1, 2, 3])),
        feature_channels=tuple(config["model"].get("sam2_feature_channels", [256, 256, 256, 256])),
        output_channels=tuple(config["model"].get("sam2_output_channels", [256, 256, 256, 256])),
        test_mask_decoder=True,
        mask_input_size=config["model"].get("sam2_mask_input_size", 256),
        strict=False,
    )
    assert result["sam_native_prior_succeeded"] or result["sam_decoder_fallback_required"]

    config["model"]["encoder_type"] = "sam2"
    config["model"]["decoder_type"] = "hybrid"
    config["model"]["use_sam_mask_prior"] = True
    model = DynamicSubdomainModel(config).cpu()
    x = torch.randn(2, 1, 256, 256)
    y = torch.randint(0, 2, (2, 1, 256, 256)).float()

    model.train()
    with warnings.catch_warnings(record=True) as caught:
        out_train = model(x, targets=y, training=True, update_prototypes=True)
    prototype_count_before = int(out_train["prototype_count"])
    model.eval()
    with torch.no_grad():
        out_eval = model(x, training=False, update_prototypes=False)
    assert out_eval["probabilities"].shape == (2, 1, 256, 256)
    assert int(out_eval["prototype_count"]) == prototype_count_before
    if out_train["sam_native_prior_logits"] is not None:
        assert float(out_train["sam_native_prior_used"].item()) == 1.0
        assert float(out_train["sam_mask_prior_available"].item()) == 1.0
        assert float(out_train["sam_decoder_fallback_used"].item()) == 0.0
    else:
        assert float(out_train["sam_native_prior_used"].item()) == 0.0
        assert float(out_train["sam_mask_prior_available"].item()) == 0.0
        assert float(out_train["sam_decoder_fallback_used"].item()) == 1.0
        assert out_train["decoder_prior_logits"] is not None
        assert any("Falling back" in str(item.message) for item in caught)
    print("SAM2 smoke test passed.")


def main() -> None:
    run_prompt_scaling_test()
    run_parse_int_tuple_test()
    run_resnet_self_test()
    run_trainer_loss_prompt_test()
    run_pixel_routing_test()
    run_prototype_slot_test()
    run_hybrid_fallback_logging_test()
    run_optional_sam2_smoke()


if __name__ == "__main__":
    main()
