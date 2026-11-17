# Showing the 3-use-case demo

## Setup (do before going live; models download once, ~GBs for relex-large)
    uv sync            # or: pip install -e .
    uv run python examples/demo_three_use_cases/demo_3_use_cases.py --only 1 2 3   # warm HF cache, check output

Go offline afterwards (`HF_HUB_OFFLINE=1`) to prove the "local / privacy-safe" claim.

## Live run
    HF_HUB_OFFLINE=1 uv run python examples/demo_three_use_cases/demo_3_use_cases.py --pause

## Talk track
1. **Zero-shot NER** – point out the invented labels (`governance_standard`) and `anchor_verified=True`: every span maps to exact characters, so quotes can be cited verbatim.
2. **PII redaction** – nothing leaves the machine; show original vs redacted. Mention reverse-offset replacement.
3. **Knowledge graph** – one pass gives nodes + typed edges; each edge has a provenance span.

## Caveats to say out loud / check before presenting
- Output is model-dependent: run it first and adjust text/thresholds if a span is missed (esp. `CAN/DGSI 101`, health card number, relation edges).
- Load time is printed separately from inference time; quote only inference ms as "CPU latency", with your hardware named.
- `gliner-relex-large` is a large model; "lightweight" applies to use case 1 only.
- Models are reloaded per use case by design (clear narrative); load times are not representative of a served model.
