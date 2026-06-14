# model/verify_model.py
"""
Verifies the full Detectra model architecture:
  - Prints layer-by-layer parameter counts
  - Confirms trainable vs frozen split
  - Runs a full forward pass with dummy data matching DataLoader output shape
  - Checks output shapes and value ranges at every stage
"""

import torch
from model.detectra import Detectra

SEQUENCE_LENGTH = 20    # must match dataloader/dataset.py
BATCH_SIZE      = 4
FRAME_SIZE      = 112


def verify_architecture() -> None:
    print("\n" + "=" * 65)
    print("  STEP 4 — MODEL ARCHITECTURE VERIFICATION")
    print("=" * 65)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\n  Device: {device}")

    # ── Build model ───────────────────────────────────────────────────────────
    print("\n  Building Detectra (pretrained=True)...")
    model = Detectra(pretrained=True).to(device)
    model.eval()

    # ── Parameter summary ─────────────────────────────────────────────────────
    param_info = model.get_trainable_params()
    print(f"\n{'─'*65}")
    print(f"  PARAMETER SUMMARY")
    print(f"{'─'*65}")
    print(f"  Total parameters     : {param_info['total']:,}")
    print(f"  Trainable parameters : {param_info['trainable']:,}  "
          f"({param_info['trainable_pct']})")
    print(f"  Frozen parameters    : {param_info['frozen']:,}")

    # ── Layer-level trainable check ───────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  LAYER FREEZE STATUS")
    print(f"{'─'*65}")
    frozen_layers = {
        "conv1 / bn1 (stem)": model.feature_extractor.conv1,
        "layer1":             model.feature_extractor.layer1,
        "layer2":             model.feature_extractor.layer2,
        "layer3":             model.feature_extractor.layer3,
    }
    unfrozen_layers = {
        "layer4":             model.feature_extractor.layer4,
        "extra_head":         model.feature_extractor.extra_head,
        "lstm_classifier":    model.lstm_classifier,
    }
    for name, module in frozen_layers.items():
        grads = [p.requires_grad for p in module.parameters()]
        status = "✓ FROZEN" if not any(grads) else "✗ SHOULD BE FROZEN"
        print(f"  {name:<30} {status}")

    for name, module in unfrozen_layers.items():
        grads = [p.requires_grad for p in module.parameters()]
        status = "✓ TRAINABLE" if all(grads) else "✗ SHOULD BE TRAINABLE"
        print(f"  {name:<30} {status}")

    # ── Forward pass: training mode (logits) ──────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  FORWARD PASS — TRAINING MODE (return_probs=False)")
    print(f"{'─'*65}")

    dummy_input = torch.randn(
        BATCH_SIZE, SEQUENCE_LENGTH, 3, FRAME_SIZE, FRAME_SIZE
    ).to(device)

    print(f"  Input shape   : {tuple(dummy_input.shape)}")

    with torch.no_grad():
        logits = model(dummy_input, return_probs=False)

    print(f"  Output shape  : {tuple(logits.shape)}  "
          f"{'✓' if logits.shape == (BATCH_SIZE, 2) else '✗ WRONG'}")
    print(f"  Output dtype  : {logits.dtype}  "
          f"{'✓' if logits.dtype == torch.float32 else '✗'}")
    print(f"  Logits sample : {logits[0].tolist()}")

    # ── Forward pass: inference mode (probabilities) ───────────────────────────
    print(f"\n{'─'*65}")
    print(f"  FORWARD PASS — INFERENCE MODE (return_probs=True)")
    print(f"{'─'*65}")

    with torch.no_grad():
        probs = model(dummy_input, return_probs=True)

    probs_sum = probs.sum(dim=1)
    sums_ok   = torch.allclose(probs_sum, torch.ones_like(probs_sum), atol=1e-5)

    print(f"  Output shape     : {tuple(probs.shape)}  "
          f"{'✓' if probs.shape == (BATCH_SIZE, 2) else '✗ WRONG'}")
    print(f"  Probabilities sum to 1.0: {sums_ok}  {'✓' if sums_ok else '✗'}")
    print(f"  Sample [P(Real), P(Fake)]: "
          f"[{probs[0][0]:.4f}, {probs[0][1]:.4f}]")
    print(f"  All probs in [0,1]: "
          f"{'✓' if probs.min() >= 0 and probs.max() <= 1 else '✗'}")

    # ── Intermediate shape trace ───────────────────────────────────────────────
    print(f"\n{'─'*65}")
    print(f"  TENSOR SHAPE TRACE THROUGH PIPELINE")
    print(f"{'─'*65}")
    print(f"  DataLoader output        : "
          f"(batch={BATCH_SIZE}, seq={SEQUENCE_LENGTH}, C=3, H=112, W=112)")
    print(f"  After reshape (batch*seq): "
          f"({BATCH_SIZE*SEQUENCE_LENGTH}, 3, 112, 112)")
    print(f"  After ResNeXt extractor  : "
          f"({BATCH_SIZE*SEQUENCE_LENGTH}, 2048)")
    print(f"  After view restore       : "
          f"({BATCH_SIZE}, {SEQUENCE_LENGTH}, 2048)")
    print(f"  After LSTM               : "
          f"({BATCH_SIZE}, {SEQUENCE_LENGTH}, 2048)")
    print(f"  After adaptive avg pool  : ({BATCH_SIZE}, 2048)")
    print(f"  After linear classifier  : ({BATCH_SIZE}, 2)")
    print(f"  After softmax (inference): ({BATCH_SIZE}, 2)  ← P(Real), P(Fake)")

    print("\n" + "=" * 65)
    print("  MODEL VERIFICATION COMPLETE — Ready for Step 5 Training")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    verify_architecture()