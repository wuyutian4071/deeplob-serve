#include <gtest/gtest.h>

#include <cmath>
#include <stdexcept>
#include <string>
#include <vector>

#include "inference_engine.hpp"

namespace {

// DEEPLOB_TEST_FIXTURES_DIR is injected by CMake (see tests/CMakeLists.txt) as an absolute
// path -- avoids any assumption about the test binary's working directory when ctest runs it.
constexpr std::size_t kFixtureWindowSize = 20;
constexpr std::size_t kFixtureBatchSize = 1;

std::string fixture_model_path() {
    return std::string(DEEPLOB_TEST_FIXTURES_DIR) + "/test_model.onnx";
}

} // namespace

TEST(InferenceEngine, LoadsAModelAndReportsItsConfiguredShape) {
    deeplob::InferenceEngine engine(fixture_model_path(), kFixtureBatchSize, kFixtureWindowSize);
    EXPECT_EQ(engine.batch_size(), kFixtureBatchSize);
    EXPECT_EQ(engine.window_size(), kFixtureWindowSize);
}

TEST(InferenceEngine, InferProducesOneLogitPerClassPerBatchRow) {
    deeplob::InferenceEngine engine(fixture_model_path(), kFixtureBatchSize, kFixtureWindowSize);
    std::vector<float> input(kFixtureBatchSize * kFixtureWindowSize * deeplob::kNumFeatures, 0.1F);

    std::vector<float> logits = engine.infer(input);

    EXPECT_EQ(logits.size(), kFixtureBatchSize * deeplob::kNumClasses);
    for (float value : logits) {
        EXPECT_TRUE(std::isfinite(value));
    }
}

TEST(InferenceEngine, InferIsDeterministicForTheSameInput) {
    deeplob::InferenceEngine engine(fixture_model_path(), kFixtureBatchSize, kFixtureWindowSize);
    std::vector<float> input(kFixtureBatchSize * kFixtureWindowSize * deeplob::kNumFeatures, 0.5F);

    std::vector<float> first = engine.infer(input);
    std::vector<float> second = engine.infer(input);

    EXPECT_EQ(first, second);
}

TEST(InferenceEngine, InferRejectsWrongSizedInput) {
    deeplob::InferenceEngine engine(fixture_model_path(), kFixtureBatchSize, kFixtureWindowSize);
    std::vector<float> too_short(kFixtureWindowSize * deeplob::kNumFeatures - 1, 0.0F);

    EXPECT_THROW((void)engine.infer(too_short), std::invalid_argument);
}
