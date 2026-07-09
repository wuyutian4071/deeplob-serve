#include "inference_engine.hpp"

#include <array>
#include <stdexcept>

namespace deeplob {

namespace {

Ort::SessionOptions make_session_options() {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(1);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    return options;
}

} // namespace

InferenceEngine::InferenceEngine(const std::string& model_path,
                                 std::size_t batch_size,
                                 std::size_t window_size)
    : batch_size_(batch_size), window_size_(window_size),
      env_(ORT_LOGGING_LEVEL_WARNING, "deeplob-serve"),
      session_(env_, model_path.c_str(), make_session_options()) {
    Ort::AllocatorWithDefaultOptions allocator;
    auto input_name_ptr = session_.GetInputNameAllocated(0, allocator);
    auto output_name_ptr = session_.GetOutputNameAllocated(0, allocator);
    input_name_ = input_name_ptr.get();
    output_name_ = output_name_ptr.get();
}

std::vector<float> InferenceEngine::infer(const std::vector<float>& input) const {
    const std::size_t expected_size = batch_size_ * window_size_ * kNumFeatures;
    if (input.size() != expected_size) {
        throw std::invalid_argument("InferenceEngine::infer: expected " +
                                    std::to_string(expected_size) + " input elements, got " +
                                    std::to_string(input.size()));
    }

    const std::array<int64_t, 3> input_shape {static_cast<int64_t>(batch_size_),
                                              static_cast<int64_t>(window_size_),
                                              static_cast<int64_t>(kNumFeatures)};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info,
                                                              const_cast<float*>(input.data()),
                                                              input.size(),
                                                              input_shape.data(),
                                                              input_shape.size());

    const char* input_names[] = {input_name_.c_str()};
    const char* output_names[] = {output_name_.c_str()};

    auto output_tensors =
        session_.Run(Ort::RunOptions {nullptr}, input_names, &input_tensor, 1, output_names, 1);

    const float* output_data = output_tensors.front().GetTensorData<float>();
    const std::size_t output_size = batch_size_ * kNumClasses;
    return std::vector<float>(output_data, output_data + output_size);
}

} // namespace deeplob
