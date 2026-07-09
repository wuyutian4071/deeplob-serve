# deeplob-serve

> Deep learning on limit order books: DeepLOB-style CNN-LSTM and Transformer models predict
> short-horizon mid-price moves from LOB data, exported to ONNX and served from a hand-tuned
> C++ inference engine with microsecond-scale latency benchmarks.

[![CI](https://github.com/wuyutian4071/deeplob-serve/actions/workflows/ci.yml/badge.svg)](https://github.com/wuyutian4071/deeplob-serve/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/Python-3.12-blue)
![C++](https://img.shields.io/badge/C%2B%2B-17-blue)
![License](https://img.shields.io/badge/license-MIT-green)

**What / Why / Results (30-second version)**

- **What:** an end-to-end ML system on the FI-2010 limit-order-book benchmark — baselines
  (logistic regression, gradient boosting), a DeepLOB-style CNN-LSTM, and a lightweight
  Transformer, evaluated with temporal splits and compared against published results; the
  selected model is exported to ONNX (dynamically quantized) and served from a C++17 ONNX
  Runtime inference engine with a latency-percentile benchmark harness.
- **Why it's different:** it closes the loop from research to production — quantization,
  C++ deployment, and measured latency budgets, not just a notebook with an accuracy number
  — and closes the loop from prediction quality to economic reality with a trading-signal
  simulation reporting PnL after realistic costs, honestly, in the same spirit as
  [statlab](https://github.com/wuyutian4071/statlab)'s real-market results section.
- **Results:** _(populated as milestones land — see `BENCHMARKS.md` once M8 exists)._

## Status

Built milestone by milestone. Current: **M3 — baselines and the evaluation harness**: a
chronological (never shuffled) train/val/test split, F1-per-class and confusion-matrix
reporting shared by every model in this project, and two baselines (logistic regression,
gradient boosting) any later model is expected to beat.

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Repo skeleton, dual CI (Python + C++) | ✅ |
| M2 | Data pipeline: FI-2010 loader + synthetic LOB generator, windowed sequences, labels | ✅ |
| M3 | Baselines (logistic regression, gradient boosting) + temporal-split evaluation harness | ✅ |
| M4 | DeepLOB-style CNN-LSTM + training infra (Lightning, Hydra, MLflow) | ⬜ |
| M5 | Transformer baseline + ablation sweeps | ⬜ |
| M6 | Comparative evaluation across all models + calibration + published-results comparison | ⬜ |
| M7 | ONNX export + quantization + differential parity check | ⬜ |
| M8 | C++ inference engine + latency/throughput benchmarks | ⬜ |
| M9 | Trading-signal simulation + polished README/DESIGN.md/BENCHMARKS.md | ⬜ |

## Quickstart

```bash
# Python
uv sync --all-groups
uv run ruff check src tests
uv run mypy
uv run pytest

# C++
cmake -S cpp -B cpp/build -DCMAKE_BUILD_TYPE=Debug -DENABLE_ASAN=ON -DENABLE_UBSAN=ON
cmake --build cpp/build
ctest --test-dir cpp/build --output-on-failure
```

See `make help` for the full list of local commands (mirrors `ci.yml` exactly).

## The data pipeline (M2)

`src/deeplob/data/lob.py` defines this project's LOB feature convention: 40 raw columns
(10 price levels, each `[ask_price, ask_volume, bid_price, bid_volume]`) — the representation
DeepLOB-style CNN models consume directly, not FI-2010's full 149-column format with
handcrafted features this project doesn't use.

`synthetic.py`'s `generate_synthetic_lob()` is a deterministic, seeded generator that all
tests (and CI) actually run against — mirrors liquibook-x's own `itch/synth.hpp` philosophy:
no real-market realism claimed, just structurally valid data (positive spread, monotonically
widening levels, positive volumes) varied enough to exercise the full pipeline meaningfully.

`labeling.py`'s `compute_labels()` implements the standard FI-2010/DeepLOB smoothed
mid-price-movement scheme: for each time *t*, compares the mean mid-price over the *past*
`horizon` snapshots to the mean over the *next* `horizon` snapshots — not a naive "raw price
at *t*+horizon vs *t*" comparison, which is noisier and non-standard in this literature.
`windowing.py`'s `make_windows()` slides a fixed-length window over the features, pairing
each window with the label at its most recent position, and drops windows ending on an
invalid boundary label rather than including them with a placeholder.

`fi2010.py`'s `load_fi2010()` reads the real dataset's on-disk format (features-as-rows,
time-as-columns — transposed relative to this project's own convention) and extracts only
the 40 raw feature rows, deliberately ignoring FI-2010's own pre-computed label rows: labels
are *always* computed via `labeling.py`, the same pathway used for synthetic data, so there
is exactly one labeling implementation to trust rather than two that might disagree on
threshold/window conventions. FI-2010 itself requires a manual download (search "FI-2010
limit order book dataset Ntakaris"; see the module's own docstring) and is never fetched in
CI — this loader is validated against a small synthetic-format fixture instead, with the
module's docstring flagging its row-order assumptions for empirical validation against the
real file.

## Baselines and the evaluation harness (M3)

`evaluation/splits.py`'s `temporal_train_val_test_split()` is the anti-leakage guarantee
every model in this project is built on: chronological splitting only, never shuffled — a
random split would let a validation window sit chronologically *before* a training window
whose own horizon-labeling already peeked past it, silently leaking future information into
training.

`evaluation/metrics.py`'s `evaluate()` — accuracy, F1 per class, macro F1, and a confusion
matrix, built once and reused by every model (baselines through the Transformer) so every
model's numbers are directly comparable — uses scikit-learn's own metric implementations
rather than reimplementing F1/confusion-matrix math from scratch.

`models/baselines.py` provides logistic regression and gradient boosting, both operating on
flattened windowed features (the same windowed `(X, y)` data every later model consumes,
just adapted at this one boundary). Logistic regression is wrapped in a scaling `Pipeline` —
not a defensive habit: the first version, unscaled, failed to converge within a reasonable
iteration budget on raw LOB features that mix prices (~100) and volumes (~1-500) on very
different scales, a real finding confirmed via `sklearn`'s own `ConvergenceWarning`, not
assumed. The `Pipeline` fits the scaler on training data only and just transforms with it at
prediction time — the standard leakage-free way to do this.

## License

MIT — see [LICENSE](LICENSE).
