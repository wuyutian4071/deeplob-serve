# deeplob-serve — Design Decisions

`README.md` covers what each milestone built and how it's verified. This document covers
*why* — organized by decision, not by milestone, for anyone who wants the reasoning without
reading the build history in order. Latency/throughput numbers backing the C++ inference
engine decisions live in [`BENCHMARKS.md`](BENCHMARKS.md).

## Table of contents

- [Synthetic data: structural validity, no real-market realism claimed](#synthetic-data-structural-validity-no-real-market-realism-claimed)
- [The smoothed labeling scheme's mechanical momentum bias](#the-smoothed-labeling-schemes-mechanical-momentum-bias)
- [Temporal splitting: the anti-leakage guarantee](#temporal-splitting-the-anti-leakage-guarantee)
- [The CNN-LSTM: structural fidelity, not byte-exact reproduction](#the-cnn-lstm-structural-fidelity-not-byte-exact-reproduction)
- [The Transformer: mean pooling, not a final hidden state](#the-transformer-mean-pooling-not-a-final-hidden-state)
- [Training infra: MPS mixed precision verified, not assumed](#training-infra-mps-mixed-precision-verified-not-assumed)
- [Model selection for deployment: architecture and robustness, not score](#model-selection-for-deployment-architecture-and-robustness-not-score)
- [ONNX export: a fixed batch size, not dynamic](#onnx-export-a-fixed-batch-size-not-dynamic)
- [Quantization: pre-processing, and a real cross-platform tolerance gap](#quantization-pre-processing-and-a-real-cross-platform-tolerance-gap)
- [The C++ inference engine: const-correctness, RPATH, sanitizers vs. benchmarks](#the-c-inference-engine-const-correctness-rpath-sanitizers-vs-benchmarks)
- [The trading-signal simulation: a naive translation, honest costs](#the-trading-signal-simulation-a-naive-translation-honest-costs)
- [Findings the process caught](#findings-the-process-caught)

## Synthetic data: structural validity, no real-market realism claimed

`data/synthetic.py`'s `generate_synthetic_lob()` is a deterministic, seeded generator every
test and CI run actually exercises — mirrors the sibling **liquibook-x** project's own
`itch/synth.hpp` philosophy exactly: no claim that this looks like a real market, just
structurally valid data (positive spread, monotonically widening levels, positive volumes)
varied enough to exercise the full pipeline meaningfully. The mid-price is a driftless
Gaussian random walk — deliberately, since a martingale's future increments are
mathematically independent of its past, giving every model in this project a hard ceiling on
genuine predictive skill that isn't a training-budget or architecture problem to solve.
FI-2010 (the real dataset this pipeline is built to also consume) requires a manual download
and is never fetched in CI; `data/fi2010.py`'s loader is validated against a synthetic-format
fixture instead, with its own docstring flagging row-order assumptions that need empirical
validation against the real file.

## The smoothed labeling scheme's mechanical momentum bias

`data/labeling.py`'s `compute_labels()` implements the standard FI-2010/DeepLOB smoothed
mid-price-movement scheme: compare a **backward**-looking mean against a **forward**-looking
mean of the mid-price, thresholded by `alpha`. On a driftless random walk, this has a real,
non-obvious property discovered during M6's comparative evaluation, not designed for or
anticipated in advance: since the walk's true expected future value equals its *current*
price, but `backward_mean` (a trailing average) necessarily *lags* the current price,
`E[forward_mean - backward_mean | history] = current_price - backward_mean` — exactly the
window's own recent momentum, visible at prediction time. A trivial momentum feature
(`current_price - backward_mean`) correlates **0.587** with the label across the full
synthetic dataset (verified directly, not just reasoned about), cleanly separating DOWN
(-0.059), STATIONARY (-0.0008), and UP (+0.058) mean momentum. This is a structural property
of *any* backward/forward-mean smoothed label applied to a non-stationary series, not a bug —
and not evidence the random walk's actual future is predictable (it mathematically isn't).
Logistic regression's linear form fits this artifact directly and cheaply; the tree ensemble
and both neural models, with more capacity and less inductive bias toward a single linear
feature, didn't converge onto the same shortcut as readily within M6's training budget — a
plausible explanation, not independently confirmed for each model. The practical
implication, stated plainly: **any labeling-scheme validation on real FI-2010 data should
check for this exact momentum artifact** before trusting an above-chance score as genuine
predictive skill.

## Temporal splitting: the anti-leakage guarantee

`evaluation/splits.py`'s `temporal_train_val_test_split()` is the anti-leakage guarantee
every model in this project is built on: chronological splitting only, never shuffled. A
random split would let a validation window sit chronologically *before* a training window
whose own horizon-labeling already peeked past it, silently leaking future information into
training. This is exactly the leakage risk the function exists to prevent, not an incidental
detail — every downstream evaluation (M3's baselines, M6's comparison, M9's simulation) relies
on it holding.

## The CNN-LSTM: structural fidelity, not byte-exact reproduction

`models/cnn_lstm.py`'s `DeepLOBCNNLSTM` follows the DeepLOB paper's (Zhang, Zohren, Roberts,
2018) key structural ideas: a 2D conv stack over the `[window, 40]` input treated as a
(time × feature) "image", a first layer that pairs each price level's (price, volume) columns
via a `(1,2)`-stride-2 kernel, further blocks that progressively merge price levels, an
Inception-style multi-scale temporal block, then an LSTM over the compressed sequence. It is
explicitly **"DeepLOB-style", not a byte-exact reproduction** — the specific channel counts
and kernel sizes weren't independently re-verified against the paper's own tables during
development, and stating that plainly is more honest than implying a precision this
implementation doesn't have. `window_size` must exceed 18 (the conv stack's fixed cumulative
time-axis reduction from six unpadded `(4,1)` convolutions) for the LSTM to have any time
steps left to process — enforced in the constructor with a clear error, not left to fail
inside the conv stack with a cryptic shape mismatch.

## The Transformer: mean pooling, not a final hidden state

`models/transformer.py`'s `LOBTransformer` is M5's alternative: a linear projection of the 40
raw per-timestep features into a `d_model` embedding, fixed sinusoidal positional encoding
(Vaswani et al., 2017), a stack of standard `TransformerEncoder` layers, then **mean pooling**
over time into the classifier head — deliberately not the CNN-LSTM's LSTM-final-hidden-state
choice, since self-attention already lets every time step attend to every other one, so a
Transformer doesn't have the same "only the final step has accumulated everything" structure
an LSTM does. It plugs into the exact same `training/lightning_module.py`'s `LOBClassifier`
training loop the CNN-LSTM uses — no changes to `training/` were needed, confirming M4's
model-agnostic design actually holds for a second, structurally different model, not just
claimed to.

## Training infra: MPS mixed precision verified, not assumed

`training/device.py` picks the accelerator and precision by actually checking `torch`'s
backend at runtime (`torch.backends.mps.is_available()`), not by assuming "this is a Mac so
it's MPS." Mixed precision on Apple Silicon was verified empirically, not assumed to behave
like CUDA's much more mature AMP support just because it's "GPU-like": a real
`lightning.Trainer.fit()` run with `accelerator="mps", precision="16-mixed"` completes cleanly
on this project's PyTorch 2.13 / Lightning 2.6 install, confirmed by actually running it —
once in isolation, once inside the full training pipeline, and again inside M9's simulation
pipeline. `training/seeding.py`'s `set_seed()` deliberately uses `np.random.seed` (the legacy
*global* RNG), not the `Generator` API `data/synthetic.py` uses elsewhere — that module wants
a local, explicitly-threaded generator for pure data generation; this one wants to seed
*global* state, since other libraries in the training stack (scikit-learn internals, anything
calling `np.random.rand()` directly) read from that global state, not a `Generator` instance
this function has no way to hand them.

## Model selection for deployment: architecture and robustness, not score

M6's comparative evaluation selected the CNN-LSTM to carry forward to ONNX export (M7) and
the C++ inference engine (M8) — explicitly **not** because it scored highest. It didn't (the
momentum-artifact finding above means no model's score there reflects genuine predictive
skill in the first place). The reasons are architectural and operational: M7-M8 are
specifically about exporting and serving a neural network, ruling out the two baselines
regardless of score; and between the two neural candidates, M5's ablation sweep found the
Transformer collapsed to never predicting STATIONARY at 2 of 3 horizons tested under a short
training budget, while the CNN-LSTM showed no such collapse at any horizon — a concrete,
observed robustness difference, not a score-based tiebreak.

## ONNX export: a fixed batch size, not dynamic

`export/onnx_export.py`'s `export_to_onnx()` produces a graph with a **fixed** batch axis
(default 1), not dynamic — the original plan, abandoned only after actually testing it, not
assumed to fail. `torch.onnx.export`'s `dynamic_shapes` argument correctly kept the batch axis
symbolic in a minimal isolated `nn.LSTM` reproduction, but specialized it back down to the
concrete traced value for this model regardless, confirmed by inspecting the exported graph's
own declared input shape directly. This reads as a genuine limitation of the current
dynamo-based ONNX exporter's handling of `nn.LSTM`'s batch-dependent internal state, not a bug
in this project's usage of the API — and a fixed batch size is also the right fit for this
project's actual near-term need (M8's C++ engine specifically benchmarks batch-size-1
latency), so this wasn't treated as a blocker requiring a workaround. A companion finding from
the same investigation: the dynamo exporter defaults `external_data=True`, splitting weights
into a companion `<name>.onnx.data` file for every export — unnecessary for this project's
small (few-hundred-KB) models, and an extra file to keep in sync wherever a `.onnx` file gets
moved (e.g. as a C++ test fixture); disabled explicitly (`external_data=False`) so every
export stays one self-contained file.

## Quantization: pre-processing, and a real cross-platform tolerance gap

`export/onnx_export.py`'s `quantize_onnx_model()` needed one extra step beyond calling
onnxruntime's `quantize_dynamic()` directly: doing so failed with a `ShapeInferenceError`
("Inferred shape and existing shape differ in dimension 0: (64) vs (3)") — the dynamo
exporter's LSTM graph isn't directly consumable by `quantize_dynamic`'s own shape-inference
pass without pre-processing first. Fixed by running onnxruntime's own `quant_pre_process()`
first, exactly what onnxruntime's own warning message on the unprocessed path already pointed
at.

`export/parity.py`'s differential check — PyTorch vs. exported ONNX vs. quantized ONNX,
mirroring the sibling **liquibook-x** project's own "verify the optimized path against a
trusted reference" discipline (hash map vs. `std::unordered_map`, `OrderBook` vs.
`ReferenceOrderBook`) — measured a genuine **cross-platform difference**, not assumed
uniform: quantized-vs-PyTorch logits differ by ~2-3×10⁻⁴ on macOS ARM64 (the development
machine) but ~1.8×10⁻² on Linux x86_64 (CI), roughly 60-90× larger, caught by CI itself
failing a tolerance this test originally shipped with before that difference was known.
Presumably a different BLAS/CPU-instruction-set path in int8 quantization's own kernels
between platforms, not a broken export — both values are still tiny relative to typical logit
magnitudes. The test tolerance was widened to cover both platforms' actual measured behavior
(`_QUANTIZED_TOLERANCE = 0.05`), not loosened blindly until CI happened to pass — a dedicated
test confirms the metric still has real discriminating power (a deliberately garbage
comparison exceeds every tolerance used here by orders of magnitude).

## The C++ inference engine: const-correctness, RPATH, sanitizers vs. benchmarks

`cpp/inference/inference_engine.hpp`'s `InferenceEngine` wraps an ONNX Runtime C++ session.
Three real findings from actually building and running this, not from writing CMake and
assuming it would work:

- `Ort::Session::Run()` isn't `const` (an implementation detail of ONNX Runtime's own C API
  binding), while `infer()` is logically read-only from this project's own contract — the
  same input always produces the same output, exactly the property
  `InferIsDeterministicForTheSameInput` verifies. Resolved with a `mutable` session member,
  keeping `infer()`'s `const` signature for callers rather than leaking that ONNX Runtime
  implementation detail into `InferenceEngine`'s own public interface.
- Neither macOS's dylib IDs nor Linux's shared-object SONAME point anywhere a built
  executable finds automatically by default. `cpp/cmake/OnnxRuntime.cmake` sets an explicit
  `RPATH` at the FetchContent-downloaded ONNX Runtime location for every target — verified on
  macOS first (the development machine), then generalized to Linux (CI's actual runner)
  rather than assumed to work identically without checking.
- Building `cpp/bench/`'s benchmark executable together with ASan/UBSan enabled fails to link
  (`deeplob_inference`, a static library, gets sanitizer instrumentation baked in whenever
  `ENABLE_ASAN`/`ENABLE_UBSAN` are on; the benchmark executable deliberately requests none —
  mixing the two doesn't link). Not a bug to fix: confirmed as liquibook-x's own established
  pattern — benchmarks only ever build on sanitizer-off Release CI legs, matching `ci.yml`'s
  own `benchmarks: ON` appearing only alongside `sanitizers: OFF`.

`cpp/bench/latency_histogram.hpp` is liquibook-x's own `bench/latency_histogram.hpp`, mirrored
directly (same interface, same reasoning: Google Benchmark's own throughput-loop model reports
mean/median across timed-region *repetitions*, not a percentile distribution of *individual
operation* latencies, which is what P50/P99/P99.9 actually mean). The measured finding —
batching provides essentially no throughput benefit for this model (~715-760 rows/sec flat
across batch sizes 1-64) — is documented with full numbers and methodology in
[`BENCHMARKS.md`](BENCHMARKS.md), not repeated here.

## The trading-signal simulation: a naive translation, honest costs

`simulation/pnl.py`'s `predictions_to_positions()` is the simplest possible translation from a
3-class prediction to a position — short/flat/long, no sizing, no confidence-weighting —
deliberately, since the simulation's purpose is testing whether the *direction* signal
survives contact with real costs, not optimizing a position-sizing strategy on top of it.
`simulate_pnl()` charges a proportional transaction cost (basis points of notional) on every
step where the position changes from the prior one, including the very first entry from flat
— the standard backtest convention, and the mechanism that turns "the model predicts
direction" into "here's what that's actually worth after the friction any real venue charges."
Given every earlier milestone (M4-M6) already established this synthetic data has no genuine
predictive signal, the simulation's own result (see `README.md`'s M9 section for the actual
numbers) isn't a surprise to be explained away — it's the same honest finding continuing to
hold under one more, more economically meaningful lens.

## Findings the process caught

The single most concrete evidence behind this project's own practice of verifying by actually
running things, not by writing code and assuming it works — collected here in one place
rather than left scattered across nine milestones' worth of commit messages.

| # | Where | What broke | How it was caught | Fix |
|---|---|---|---|---|
| 1 | M3 | `LogisticRegressionBaseline`, unscaled, failed to converge on raw LOB features mixing prices (~100) and volumes (~1-500) on very different scales | sklearn's own `ConvergenceWarning` on synthetic data | Wrapped in a `StandardScaler` `Pipeline`, fit on training data only |
| 2 | M4 | `torch.onnx.export`-adjacent: none yet — first real training-infra finding was MLflow's filesystem store | Current MLflow (3.x) raises rather than silently using the plain `./mlruns` filesystem backend (put into maintenance mode) | Actually running `train.py` and reading the raised exception | Switched to a SQLite tracking URI (`sqlite:///mlflow.db`) |
| 3 | M4 | Lightning's default auto-checkpointing saved to a bare numbered-experiment-ID directory at the repo root, independent of MLflow's own `artifact_location` config | Inspecting what files a real training run actually created | `enable_checkpointing=False` (not a stated M4 requirement yet) |
| 4 | M6 | Comparative evaluation's logistic regression scored 57.4% — well above chance, contradicting every prior milestone's "near chance" story | Consistency across three disjoint splits (train/val/test) ruled out an overfit fluke, making it worth tracing rather than writing off | Traced to the labeling scheme's mechanical momentum bias (see above) — not a bug, a real structural property, documented rather than "fixed" |
| 5 | M6 | `LogisticRegressionBaseline`'s `ConvergenceWarning` (already fixed once, #1) resurfaced at `window_size=100`'s 4000 flattened features, 5x the size M3's own tests used | Re-running the same class at a larger scale for the M6 comparison | Raised `max_iter` to 5000 for that comparison's instantiation |
| 6 | M6 | `GradientBoostingClassifier`'s exact (non-histogram) split search scaled badly with feature count — 100 estimators at 4000 features took 7+ minutes and was killed; 30 estimators was still running past 10 minutes and was killed too | Actually running the comparison twice, not assuming a smaller `n_estimators` alone would be fast enough | Reduced `compare.py`'s own `window_size` to 30 (1200 features), scoped only to that script |
| 7 | M7 | `torch.onnx.export` defaults to the newer dynamo-based exporter (`dynamo=True`), which needs `onnxscript` | `ModuleNotFoundError` on the first export attempt | Added `onnxscript` as a dependency rather than opting into the older, being-deprecated TorchScript exporter |
| 8 | M7 | The exported model's batch axis stayed fixed despite `dynamic_shapes` (see "ONNX export" above) | Inspecting the exported graph's own declared input shape directly | Accepted a fixed batch size; documented as a real exporter limitation, not worked around |
| 9 | M7 | `quantize_dynamic()` failed with a `ShapeInferenceError` on the dynamo-exported LSTM graph | Actually running quantization, not assuming it would just work | Ran onnxruntime's own `quant_pre_process()` first |
| 10 | M7 | The dynamo exporter's default `external_data=True` split every export (even small models) into a companion `.onnx.data` file | Inspecting the files a real export actually produced while building M8's C++ test fixtures | `external_data=False`, with a regression test locking the single-file behavior in |
| 11 | M7/M8 | Quantized-vs-PyTorch parity tolerance, calibrated only on macOS, failed on Linux CI (~1.8×10⁻² vs. the ~2-3×10⁻⁴ macOS number) | CI's own Linux run of the M8 push | Widened the tolerance to `0.05`, covering both platforms' real measured behavior |
| 12 | M8 | `Ort::Session::Run()` isn't `const`, breaking `InferenceEngine::infer()`'s intended `const` signature | Compiling against the real ONNX Runtime C++ headers for the first time | `mutable` session member |
| 13 | M8 | Neither macOS's dylib IDs nor Linux's SONAME resolve automatically; built executables couldn't find `libonnxruntime` at runtime | Explicit testing on macOS first | Explicit `RPATH` for every target, generalized to Linux |
| 14 | M8 | Building `cpp/bench/` with ASan/UBSan enabled failed to link against the sanitizer-instrumented `deeplob_inference` static library | A deliberate local test of the Debug+ASan+Benchmarks combination | Confirmed as liquibook-x's own established non-supported combination; `ci.yml` only enables benchmarks on sanitizer-off Release legs |

A pattern worth naming, matching liquibook-x's own observation: several of these (#2, #7, #8,
#9, #10, #12, #13) were only ever catchable by *actually running* the relevant code against
real dependencies (a live MLflow install, the real dynamo exporter, the real ONNX Runtime C++
headers) — not by writing code that looked correct on inspection. That's the concrete argument
for why this project's stated discipline ("verify by running, not by assuming") is a real
practice here, not a checkbox.
