// Standalone check of the Python output bridge, not an APSP benchmark.
// Compile with C++17 and include paths for Eigen and source/python/wrappers.
#include <Eigen/Dense>
#include "distance_output.h"

#include <array>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <utility>
#include <variant>

namespace {

std::array<uint64_t, 4> storage;
int allocation_calls = 0;
int64_t allocated_n = 0;
int32_t allocated_width = 0;
bool fail_allocation = false;

void Require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

void* Allocate(int64_t n, int32_t itemsize) {
    ++allocation_calls;
    allocated_n = n;
    allocated_width = itemsize;
    return fail_allocation ? nullptr : storage.data();
}

template <typename Scalar>
using Matrix = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;

using Result = std::variant<std::monostate, Matrix<uint8_t>, Matrix<uint16_t>,
                            Matrix<uint32_t>, Matrix<uint64_t>, Matrix<double>>;

template <typename Scalar>
void CheckType(Scalar high_value) {
    // Deliberately asymmetric test data exposes transposition mistakes. Native
    // source vectors are columns; the Python output stores source vectors as rows.
    Matrix<Scalar> matrix(2, 2);
    matrix(0, 0) = 0;
    matrix(1, 0) = 1;
    matrix(0, 1) = high_value;
    matrix(1, 1) = (std::numeric_limits<Scalar>::max)();
    const Result result = std::move(matrix);
    const std::array<Scalar, 4> expected{
        0, 1, high_value, (std::numeric_limits<Scalar>::max)()};

    allocation_calls = 0;
    fast_apsp::CopyNativeDistances(result, Allocate);
    Require(allocation_calls == 1, "Output must be allocated exactly once");
    Require(allocated_n == 2, "Wrong output dimension");
    Require(allocated_width == sizeof(Scalar), "Native integer width was lost");
    std::array<Scalar, 4> actual{};
    std::memcpy(actual.data(), storage.data(), sizeof(actual));
    Require(actual == expected, "Output values or source-major layout changed");

    fast_apsp::CopyNativeDistances(result, nullptr);
    Require(allocation_calls == 1, "No-return mode must not allocate");

    fail_allocation = true;
    bool failed = false;
    try {
        fast_apsp::CopyNativeDistances(result, Allocate);
    } catch (const std::exception&) {
        failed = true;
    }
    fail_allocation = false;
    Require(failed, "Null output allocation must be reported as an error");
    fast_apsp::CopyNativeDistances(result, Allocate);
    std::memcpy(actual.data(), storage.data(), sizeof(actual));
    Require(actual == expected, "Allocation failure corrupted later output");
}

}  // namespace

int main() {
    try {
        CheckType<uint8_t>(129);
        CheckType<uint16_t>(32769);
        CheckType<uint32_t>(UINT32_C(2147483649));
        // This value cannot be represented exactly in float64.
        CheckType<uint64_t>(UINT64_C(9007199254740993));

        Matrix<double> floating = Matrix<double>::Zero(2, 2);
        bool rejected = false;
        try {
            fast_apsp::CopyNativeDistances(Result{floating}, Allocate);
        } catch (const std::exception&) {
            rejected = true;
        }
        Require(rejected, "Unexpected floating-point native output must be rejected");

        for (const Result& invalid : {
                 Result{std::monostate{}}, Result{Matrix<uint8_t>(2, 3)},
                 Result{Matrix<uint8_t>(0, 0)}}) {
            rejected = false;
            try {
                fast_apsp::CopyNativeDistances(invalid, Allocate);
            } catch (const std::exception&) {
                rejected = true;
            }
            Require(rejected, "Absent, empty, or nonsquare output must be rejected");
        }
        std::cout << "PASS: all unsigned widths, exact uint64, layout, and allocation lifecycle\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
