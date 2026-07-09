#include <gtest/gtest.h>

#include "version.hpp"

TEST(Version, ReturnsNonEmptyString) {
    EXPECT_FALSE(deeplob::version().empty());
}

TEST(Version, MatchesExpectedM1Tag) {
    EXPECT_EQ(deeplob::version(), "0.1.0-m1");
}
