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

Built milestone by milestone. Current: **M5 — Transformer baseline + ablation sweeps**: a
lightweight Transformer encoder plugged into the same generic training loop the CNN-LSTM
uses, a proper Hydra config group so the model is swappable from the command line, and a
6-way ablation sweep (model × label horizon) run end to end via Hydra's `--multirun`.

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Repo skeleton, dual CI (Python + C++) | ✅ |
| M2 | Data pipeline: FI-2010 loader + synthetic LOB generator, windowed sequences, labels | ✅ |
| M3 | Baselines (logistic regression, gradient boosting) + temporal-split evaluation harness | ✅ |
| M4 | DeepLOB-style CNN-LSTM + training infra (Lightning, Hydra, MLflow) | ✅ |
| M5 | Transformer baseline + ablation sweeps | ✅ |
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

## Training infra and the CNN-LSTM (M4)

`training/` is the infrastructure every neural model in this project (this CNN-LSTM, M5's
Transformer) shares: `seeding.py` seeds Python/numpy/torch reproducibly; `device.py` picks
the accelerator and precision by actually checking `torch`'s backend at runtime, not by
assuming "this is a Mac so it's MPS"; `dataset.py` wraps windowed `(X, y)` arrays for
PyTorch DataLoaders; `lightning_module.py`'s `LOBClassifier` is a generic training loop any
classifier `nn.Module` plugs into.

Mixed precision on MPS was verified empirically, not assumed: a real
`lightning.Trainer.fit()` run with `accelerator="mps", precision="16-mixed"` completes
cleanly on this project's PyTorch 2.13 / Lightning 2.6 install, confirmed by actually running
it (twice — once in isolation, once inside the full `train.py` pipeline) rather than taken on
faith from CUDA's much more mature AMP reputation.

`models/cnn_lstm.py`'s `DeepLOBCNNLSTM` follows the DeepLOB paper's (Zhang, Zohren, Roberts,
2018) key structural ideas — a 2D conv stack that pairs each price level's (price, volume)
columns and progressively merges levels, an Inception-style multi-scale temporal block, then
an LSTM over the compressed sequence — but is explicitly **"DeepLOB-style", not a byte-exact
reproduction**: the specific channel counts and kernel sizes here weren't independently
re-verified against the paper's own tables during development, and saying so plainly is more
honest than implying a precision this implementation doesn't have.

`training/train.py` is the Hydra config-driven entry point
(`uv run python -m deeplob.training.train`, config at `configs/config.yaml`) tying together
every M2-M4 piece — synthetic data through labeling, windowing, the temporal split, training,
and the shared evaluation harness — with local MLflow tracking (`sqlite:///mlflow.db`, a
database backend: current MLflow has put the plain filesystem `./mlruns` store into
maintenance mode and raises rather than silently using it, discovered only by running this
script and reading the actual error). Running it end to end on synthetic data surfaces an
expected, honest result: training accuracy climbs well above validation/test accuracy, which
lands right around chance (~33%, three classes) — exactly what should happen, since the
synthetic generator is a pure random walk with no real predictive signal by construction.
That gap confirms the training mechanics work (the model can fit data, gradients flow, loss
decreases) without the model spuriously "cheating" on data that has nothing genuine to learn;
real FI-2010 data (a separate, documented manual step) is what would actually test whether
this architecture predicts anything.

## Transformer + ablation sweep (M5)

`models/transformer.py`'s `LOBTransformer` is M5's alternative to the CNN-LSTM: a linear
projection of the 40 raw per-timestep features into a `d_model` embedding, fixed sinusoidal
positional encoding (Vaswani et al., 2017), a stack of standard `TransformerEncoder` layers,
then mean-pooling over time into the classifier head. It plugs into the exact same
`LOBClassifier` training loop the CNN-LSTM uses — no changes to `training/` were needed,
which is the point of M4's model-agnostic design.

`configs/model/` is now a proper Hydra config group (`cnn_lstm.yaml` / `transformer.yaml`),
replacing M4's flat inline `model:` block — the model is swappable from the command line
(`model=cnn_lstm` or `model=transformer`) or swept across, without touching `config.yaml`.
`mlflow.experiment_name` now resolves from the actual model choice
(`deeplob-${hydra:runtime.choices.model}`) rather than a fixed name, so Transformer and
CNN-LSTM runs land in separate MLflow experiments instead of silently mixing together.

An ablation sweep — model choice × label horizon (5/10/20 snapshots) — was run end to end via
`uv run python -m deeplob.training.train --multirun model=cnn_lstm,transformer
data.horizon=5,10,20`, 6 combinations on synthetic data:

| Model | Horizon | Accuracy | Macro F1 | Note |
|-------|---------|----------|----------|------|
| CNN-LSTM | 5 | 0.351 | 0.324 | |
| CNN-LSTM | 10 | 0.350 | 0.347 | |
| CNN-LSTM | 20 | 0.336 | 0.300 | |
| Transformer | 5 | 0.341 | 0.276 | |
| Transformer | 10 | 0.329 | 0.196 | STATIONARY F1 = 0.0 — collapsed to never predicting it |
| Transformer | 20 | 0.367 | 0.265 | STATIONARY F1 = 0.0 — same collapse |

Every combination lands in the 33-37% range — chance, for three classes — which is the
expected, correct result on this project's pure-random-walk synthetic data (see M4's own
finding above; this isn't a new bug, it's the same honest baseline holding across a second
model and three horizons). One genuine finding *did* surface from the sweep, though: at
horizons 10 and 20, the Transformer collapsed to predicting only DOWN/UP and never
STATIONARY at all, while the CNN-LSTM showed no such collapse at any horizon tested. Reported
plainly rather than smoothed over — with only 3 epochs per sweep combination (kept short
deliberately, to make a 6-way sweep run quickly on synthetic data), this reads as an
under-trained Transformer settling into a degenerate 2-class solution rather than a deeper
architectural problem, but that's an inference, not something independently confirmed here;
a longer per-combination training budget would be the natural next check if this mattered for
a real deployment decision.

## License

MIT — see [LICENSE](LICENSE).
