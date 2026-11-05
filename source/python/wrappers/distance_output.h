#ifndef FAST_APSP_DISTANCE_OUTPUT_H
#define FAST_APSP_DISTANCE_OUTPUT_H

#include <cstdint>
#include <cstring>
#include <limits>
#include <new>
#include <stdexcept>
#include <type_traits>
#include <variant>

namespace fast_apsp {

// Called once, synchronously on the calling thread after native computation.
// The caller owns the returned writable n*n buffer. A null callback omits
// output; a null buffer reports allocation failure. itemsize denotes uint8/16/32/64.
using DistanceAllocator = void* (*)(int64_t n, int32_t itemsize);

template <typename MatrixVariant>
void CopyNativeDistances(const MatrixVariant& distances, DistanceAllocator allocate) {
    if (allocate == nullptr) return;
    std::visit([allocate](const auto& matrix) {
        using Matrix = std::decay_t<decltype(matrix)>;
        if constexpr (std::is_same_v<Matrix, std::monostate>) {
            throw std::runtime_error("Native distance matrix is unavailable.");
        } else if constexpr (!std::is_unsigned_v<typename Matrix::Scalar>) {
            throw std::runtime_error("Expected unsigned native distances.");
        } else {
            using Scalar = typename Matrix::Scalar;
            static_assert(!Matrix::IsRowMajor, "Native distance vectors must be columns.");
            if (matrix.rows() < 1 || matrix.rows() != matrix.cols()) {
                throw std::runtime_error("Expected a nonempty square distance matrix.");
            }
            const auto count = static_cast<size_t>(matrix.size());
            if (count > (std::numeric_limits<size_t>::max)() / sizeof(Scalar)) {
                throw std::overflow_error("Distance output is too large.");
            }
            void* output = allocate(matrix.rows(), sizeof(Scalar));
            if (output == nullptr) throw std::bad_alloc();
            // Native columns are source distance vectors. Their contiguous
            // storage is already NumPy's C-order [source, target] layout.
            std::memcpy(output, matrix.data(), count * sizeof(Scalar));
        }
    }, distances);
}

}  // namespace fast_apsp
#endif
