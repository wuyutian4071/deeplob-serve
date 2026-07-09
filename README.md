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

Built milestone by milestone. Current: **M1 — repo skeleton**: the Python side (uv, ruff,
mypy strict, pytest, `src/` layout) and the C++ side (CMake, GoogleTest, sanitizer-clean CI)
are both scaffolded and green, proving the combined toolchain works end to end before any
real module exists.

| Milestone | Scope | State |
|-----------|-------|-------|
| M1 | Repo skeleton, dual CI (Python + C++) | ✅ |
| M2 | Data pipeline: FI-2010 loader + synthetic LOB generator, windowed sequences, labels | ⬜ |
| M3 | Baselines (logistic regression, gradient boosting) + temporal-split evaluation harness | ⬜ |
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

## License

MIT — see [LICENSE](LICENSE).
