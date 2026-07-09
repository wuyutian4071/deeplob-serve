# deeplob-serve — Benchmarks

Numbers below were produced by actually running the benchmark executable in this repo,
on the machine described below — not estimated, not carried over from a different build, not
rounded up. If you re-run these yourself and get different numbers, trust yours; hardware, OS
scheduler state, and background load all matter more than most people expect for anything
measured in microseconds.

## Methodology

**Hardware/OS**: Apple Silicon, macOS, Apple clang, CMake. Release build
(`-DCMAKE_BUILD_TYPE=Release`), no sanitizers — ASan/UBSan overhead would make every number
below meaningless, which is also why `cpp/bench/CMakeLists.txt` never applies
`deeplob_apply_sanitizers()` to the benchmark target, matching the sibling **liquibook-x**
project's own `bench/CMakeLists.txt` reasoning exactly.

**CPU pinning**: `cpp/bench/latency_histogram.hpp`'s `pin_current_thread_to_cpu()` is real on
Linux (`pthread_setaffinity_np`) and a documented no-op everywhere else — macOS has no public
equivalent thread-affinity API. **The numbers below were collected without pinning**, since
this dev machine is macOS. CI's Linux runners would get real pinning if the benchmark suite
were run there, but CI only compile-checks this target (see `.github/workflows/ci.yml`'s
`benchmarks: ON` Release legs) — it doesn't run and publish results, since a shared, possibly
throttled CI runner would produce numbers even less representative than an unloaded dev
machine.

**Warmup and sampling**: the batch-size-1 latency benchmark runs 50 warmup iterations
(discarded) followed by 500 measured iterations, each timed individually via
`std::chrono::steady_clock` around just the `infer()` call under test —
`latency_histogram.hpp`'s `LatencyHistogram` then sorts the samples and reports
P50/P90/P99/P99.9/max plus the mean. Far fewer iterations than liquibook-x's own
order-book/matching-engine benchmarks (100,000 measured), deliberately: each `infer()` call is
a real neural-network forward pass through ONNX Runtime, several orders of magnitude more
expensive per call than an `OrderBook` operation, so 500 samples is what keeps this benchmark
practical to re-run without a multi-minute wait, while still giving a meaningful percentile
spread.

**Model and input**: `DeepLOBCNNLSTM`, `window_size=100` (matching `configs/config.yaml`'s own
training default), random (untrained-model-equivalent) input — this benchmark measures
inference *mechanics* (session dispatch, the conv+LSTM forward pass, ONNX Runtime overhead),
not anything dependent on what the model has learned, so an untrained model's timing is
identical to a trained one's.

## Results

### Batch-size-1 latency

500 samples (50 warmup) from `cpp/bench/bench_inference_latency.cpp`.

| Operation | Mean | P50 | P90 | P99 | P99.9 | Max |
|---|--:|--:|--:|--:|--:|--:|
| `InferenceEngine::infer` (batch=1) | 1,338.3µs | 1,292.2µs | 1,475.4µs | 1,969.1µs | 2,321.8µs | 4,593.0µs |

This is the deployment shape this project actually targets — matching M7's own decision to
export a fixed batch-size-1 model (see [`DESIGN.md`](DESIGN.md#onnx-export-a-fixed-batch-size-not-dynamic)
for why the export couldn't be made dynamic-batch instead). At roughly 1.3ms median, this
CNN-LSTM is firmly in "not microsecond-scale" territory compared to liquibook-x's own
sub-100ns order-book operations — expected and unsurprising: a multi-layer conv+LSTM forward
pass through a general-purpose ONNX Runtime session is fundamentally a different class of
operation than an in-memory data-structure update, not a target this project ever claimed to
hit.

### Throughput across batch sizes

| Batch size | Calls | Total time | Throughput |
|---|--:|--:|--:|
| 1 | 500 | 0.688s | 726.9 rows/sec |
| 8 | 62 | 0.691s | 718.2 rows/sec |
| 32 | 15 | 0.652s | 736.3 rows/sec |
| 64 | 7 | 0.617s | 726.1 rows/sec |

**Batching provides essentially no throughput benefit here — reported plainly, not the result
a reader might expect from a "throughput at larger batch sizes" section.** Throughput stays
flat (~715-760 rows/sec) regardless of batch size. The likely explanation: `InferenceEngine`
deliberately runs single-threaded (`SetIntraOpNumThreads(1)`, matching this project's
determinism ethos — multi-threaded ONNX Runtime execution can introduce run-to-run
non-determinism), and this CNN-LSTM's LSTM layer is inherently sequential, processing
`window_size=100` timesteps one at a time regardless of how many rows are batched alongside
each other — so total wall-clock time scales with total timesteps processed, not with the
number of `infer()` calls. That reads as a genuine architectural property of this network, not
a bug in the benchmark or the inference engine, but it's stated as the most likely
explanation, not independently confirmed via, say, profiling the ONNX graph's own per-op time
breakdown — a natural follow-up if batched throughput mattered for a real deployment decision.

## Reproducing these numbers

```bash
cmake -S cpp -B cpp/build-release -DCMAKE_BUILD_TYPE=Release -DDEEPLOB_BUILD_BENCHMARKS=ON
cmake --build cpp/build-release
./cpp/build-release/bench/deeplob_bench_inference_latency
```

Or simply `make cpp-bench`, which runs the same three commands.
