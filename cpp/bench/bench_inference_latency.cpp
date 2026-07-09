#include <benchmark/benchmark.h>

#include <chrono>
#include <cstdio>
#include <random>
#include <string>
#include <vector>

#include "inference_engine.hpp"
#include "latency_histogram.hpp"

using deeplob::InferenceEngine;
using deeplob::kNumFeatures;
using deeplob::bench::LatencyHistogram;
using deeplob::bench::pin_current_thread_to_cpu;

namespace {

constexpr std::size_t kWindowSize = 100;
constexpr std::size_t kWarmupIterations = 50;
constexpr std::size_t kMeasuredIterations = 500;

std::string fixture_path(std::size_t batch_size) {
    return std::string(DEEPLOB_BENCH_FIXTURES_DIR) + "/model_b" + std::to_string(batch_size) +
           ".onnx";
}

std::vector<float> random_input(std::size_t batch_size, std::mt19937_64& rng) {
    std::uniform_real_distribution<float> dist(-1.0F, 1.0F);
    std::vector<float> input(batch_size * kWindowSize * kNumFeatures);
    for (float& value : input) {
        value = dist(rng);
    }
    return input;
}

void print_report(const char* name, const LatencyHistogram::Percentiles& p) {
    std::printf("%-28s n=%-6zu mean=%9.1fns  p50=%8lldns  p90=%8lldns  p99=%8lldns  "
                "p99.9=%8lldns  max=%9lldns\n",
                name,
                p.sample_count,
                p.mean,
                static_cast<long long>(p.p50),
                static_cast<long long>(p.p90),
                static_cast<long long>(p.p99),
                static_cast<long long>(p.p999),
                static_cast<long long>(p.max));
}

// Batch-size-1 latency, the deployment shape M8 is specifically about (see M7's own
// fixed-batch-size export decision): warmup outside the timed region, one sample recorded
// per infer() call so the reported percentiles describe individual-call latency, not a
// throughput-loop average.
void bench_batch_size_1_latency() {
    InferenceEngine engine(fixture_path(1), /*batch_size=*/1, kWindowSize);
    std::mt19937_64 rng(42);

    for (std::size_t i = 0; i < kWarmupIterations; ++i) {
        auto warm_input = random_input(1, rng);
        auto warm_result = engine.infer(warm_input);
        benchmark::DoNotOptimize(warm_result);
    }

    LatencyHistogram hist(kMeasuredIterations);
    for (std::size_t i = 0; i < kMeasuredIterations; ++i) {
        auto input = random_input(1, rng);
        const auto start = std::chrono::steady_clock::now();
        auto result = engine.infer(input);
        const auto end = std::chrono::steady_clock::now();
        benchmark::DoNotOptimize(result);
        hist.record(std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count());
    }

    print_report("InferenceEngine::infer (b=1)", hist.compute());
}

// Throughput at larger batch sizes: total wall-clock time for a fixed number of infer()
// calls at each batch size, reported as rows/sec -- whether batching amortizes the CNN-LSTM's
// fixed per-call overhead (session dispatch, LSTM sequential-step cost) the way it would for
// a genuinely throughput-oriented serving path, distinct from the latency-focused
// batch-size-1 benchmark above.
void bench_throughput_at_batch_size(std::size_t batch_size) {
    InferenceEngine engine(fixture_path(batch_size), batch_size, kWindowSize);
    std::mt19937_64 rng(7);

    const std::size_t iterations = std::max<std::size_t>(kMeasuredIterations / batch_size, 5);

    for (std::size_t i = 0; i < 5; ++i) {
        auto warm_input = random_input(batch_size, rng);
        auto warm_result = engine.infer(warm_input);
        benchmark::DoNotOptimize(warm_result);
    }

    const auto start = std::chrono::steady_clock::now();
    for (std::size_t i = 0; i < iterations; ++i) {
        auto input = random_input(batch_size, rng);
        auto result = engine.infer(input);
        benchmark::DoNotOptimize(result);
    }
    const auto end = std::chrono::steady_clock::now();

    const double elapsed_seconds =
        std::chrono::duration_cast<std::chrono::duration<double>>(end - start).count();
    const double rows_per_second = static_cast<double>(iterations * batch_size) / elapsed_seconds;

    std::printf("batch_size=%-4zu calls=%-6zu total=%8.3fs  throughput=%10.1f rows/sec\n",
                batch_size,
                iterations,
                elapsed_seconds,
                rows_per_second);
}

} // namespace

int main() {
    const bool pinned = pin_current_thread_to_cpu(0);
    std::printf("CPU pinning: %s\n", pinned ? "active (core 0)" : "unavailable on this platform");
    std::printf("window_size=%zu warmup=%zu measured=%zu\n\n",
                kWindowSize,
                kWarmupIterations,
                kMeasuredIterations);

    bench_batch_size_1_latency();

    std::printf("\nThroughput at larger batch sizes:\n");
    for (std::size_t batch_size :
         {std::size_t {1}, std::size_t {8}, std::size_t {32}, std::size_t {64}}) {
        bench_throughput_at_batch_size(batch_size);
    }

    return 0;
}
