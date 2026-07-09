#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

namespace deeplob {

// Matches deeplob.data.lob.NUM_FEATURES / the labeling scheme's 3-class output on the
// Python side -- kept in sync deliberately, not re-derived from the model file itself.
inline constexpr std::size_t kNumFeatures = 40;
inline constexpr std::size_t kNumClasses = 3;

// Wraps an ONNX Runtime session for a DeepLOB-style CNN-LSTM model exported by this
// project's Python side (deeplob.export.onnx_export). The exported graph has a FIXED batch
// axis, not a dynamic one -- see that module's own docstring for why (the current ONNX
// exporter specializes nn.LSTM's batch dimension regardless of a requested dynamic axis) --
// so this engine is constructed for one specific batch_size and only accepts input laid out
// for exactly that many rows, matching the loaded model file.
class InferenceEngine {
public:
    InferenceEngine(const std::string& model_path, std::size_t batch_size, std::size_t window_size);

    // `input` must have exactly batch_size() * window_size() * kNumFeatures elements,
    // row-major as [batch, window_size, kNumFeatures] -- the exported graph's own layout.
    // Returns batch_size() * kNumClasses logits, same row-major convention. Throws
    // std::invalid_argument if `input`'s size doesn't match.
    [[nodiscard]] std::vector<float> infer(const std::vector<float>& input) const;

    [[nodiscard]] std::size_t batch_size() const noexcept { return batch_size_; }
    [[nodiscard]] std::size_t window_size() const noexcept { return window_size_; }

private:
    std::size_t batch_size_;
    std::size_t window_size_;
    Ort::Env env_;
    // ONNX Runtime's Session::Run() isn't const (an implementation detail of its own C API
    // binding), but infer() is logically read-only from this class's own contract -- the
    // same input always produces the same output (see the
    // InferIsDeterministicForTheSameInput test). mutable here preserves infer()'s const
    // signature for callers rather than leaking that implementation detail into this class's
    // public interface.
    mutable Ort::Session session_;
    std::string input_name_;
    std::string output_name_;
};

} // namespace deeplob
