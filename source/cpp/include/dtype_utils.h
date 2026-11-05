#ifndef DTYPE_UTILS_H
#define DTYPE_UTILS_H

#include <Eigen/Dense>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#ifdef MATLAB_BUILD
#include "mex.h"
#endif

template <typename DistScalar>
using DistVec = Eigen::Matrix<DistScalar, Eigen::Dynamic, 1>;

template <typename DistScalar>
using DistMat = Eigen::Matrix<DistScalar, Eigen::Dynamic, Eigen::Dynamic>;

template <typename DistScalar>
using DenseDistMat = DistMat<DistScalar>;

using ResultMatrix = std::variant<
    std::monostate,
    DenseDistMat<uint8_t>,
    DenseDistMat<uint16_t>,
    DenseDistMat<uint32_t>,
    DenseDistMat<uint64_t>,
    DenseDistMat<double>
>;

template <typename DistScalar>
const char* DistTypeName() {
    return "unknown";
}

template <>
inline const char* DistTypeName<uint8_t>() {
    return "uint8";
}

template <>
inline const char* DistTypeName<uint16_t>() {
    return "uint16";
}

template <>
inline const char* DistTypeName<uint32_t>() {
    return "uint32";
}

template <>
inline const char* DistTypeName<uint64_t>() {
    return "uint64";
}

template <>
inline const char* DistTypeName<double>() {
    return "float64";
}

template <typename DistScalar>
struct DistVecCache {
    int64_t n;
    int64_t capacity;
    int64_t num_stored;
    bool fixed_slot_layout;

    DistMat<DistScalar> buf;
    std::vector<int64_t> src_to_slot;
    std::vector<char> has_vec;

    DistVecCache(int64_t n_, int64_t capacity_, bool fixed_slot_layout_ = false)
        : n(n_), capacity(capacity_), num_stored(0), fixed_slot_layout(fixed_slot_layout_) {
        if (capacity < 0) {
            throw std::runtime_error("DistVecCache: capacity must be nonnegative.");
        }
        if (fixed_slot_layout && capacity != n) {
            throw std::runtime_error("DistVecCache: fixed-slot layout requires capacity == n.");
        }
        if (capacity > 0) {
            buf.resize(n, capacity);
        }
        src_to_slot.assign(n, -1);
        has_vec.assign(n, false);
        if (fixed_slot_layout) {
            for (int64_t s = 0; s < n; ++s) {
                src_to_slot[s] = s;
            }
        }
    }

    inline bool Has(int64_t s) const {
        return has_vec[s];
    }

    inline Eigen::Ref<const DistVec<DistScalar>> Get(int64_t s) const {
        const int64_t slot = src_to_slot[s];
        return buf.col(slot);
    }

    void Store(int64_t s, const DistVec<DistScalar>& vec) {
        if (has_vec[s]) {
            return;
        }
    
        if (fixed_slot_layout) {
            num_stored++;
            has_vec[s] = true;
            buf.col(s) = vec;
            return;
        }
    
        if (num_stored >= capacity) {
            return;
        }
    
        const int64_t slot = num_stored;
        num_stored++;
        src_to_slot[s] = slot;
        has_vec[s] = true;
        buf.col(slot) = vec;
    }
};

template <typename DistScalar>
static int64_t DistVecCacheAllocatedBytes(const DistVecCache<DistScalar>& c) {
    return static_cast<int64_t>(c.buf.size() * sizeof(DistScalar));
}

template <typename DistScalar>
static int64_t DistMatAllocatedBytes(const DistMat<DistScalar>& D) {
    return static_cast<int64_t>(D.size() * sizeof(DistScalar));
}

template <typename DistScalar>
static void ConvertDistVecCacheToMatrix(DistVecCache<DistScalar>& D_mem,
                                        DistMat<DistScalar>& D_out) {
    const int64_t n = D_mem.n;
    for (int64_t s = 0; s < n; ++s) {
        if (!D_mem.Has(s)) {
            throw std::runtime_error("D_mem is missing a source distance vector.");
        }
    }
    if (D_mem.fixed_slot_layout) {
        D_out = std::move(D_mem.buf);
        return;
    }
    D_out.resize(n, n);
    for (int64_t s = 0; s < n; ++s) {
        D_out.col(s) = D_mem.Get(s);
    }
}

static Eigen::MatrixXd ResultMatrixToDouble(const ResultMatrix& D) {
    return std::visit([](const auto& D_typed) -> Eigen::MatrixXd {
        using MatrixT = std::decay_t<decltype(D_typed)>;
        if constexpr (std::is_same_v<MatrixT, std::monostate>) {
            return Eigen::MatrixXd();
        } else {
            return D_typed.template cast<double>();
        }
    }, D);
}

#ifdef MATLAB_BUILD
template <typename DistScalar>
static mxClassID MatlabClassId() {
    if constexpr (std::is_same_v<DistScalar, uint8_t>) {
        return mxUINT8_CLASS;
    } else if constexpr (std::is_same_v<DistScalar, uint16_t>) {
        return mxUINT16_CLASS;
    } else if constexpr (std::is_same_v<DistScalar, uint32_t>) {
        return mxUINT32_CLASS;
    } else if constexpr (std::is_same_v<DistScalar, uint64_t>) {
        return mxUINT64_CLASS;
    } else {
        return mxDOUBLE_CLASS;
    }
}

template <typename DistScalar>
static mxArray* CreateMatlabMatrixFromEigen(const DistMat<DistScalar>& D) {
    mxArray* out = mxCreateNumericMatrix(static_cast<mwSize>(D.rows()),
                                         static_cast<mwSize>(D.cols()),
                                         MatlabClassId<DistScalar>(),
                                         mxREAL);
    auto* out_ptr = static_cast<DistScalar*>(mxGetData(out));
    Eigen::Map<DistMat<DistScalar>>(out_ptr, D.rows(), D.cols()) = D;
    return out;
}

static mxArray* CreateMatlabMatrixFromResult(const ResultMatrix& D) {
    return std::visit([](const auto& D_typed) -> mxArray* {
        using MatrixT = std::decay_t<decltype(D_typed)>;
        if constexpr (std::is_same_v<MatrixT, std::monostate>) {
            throw std::runtime_error("Distance matrix is not available.");
        } else {
            return CreateMatlabMatrixFromEigen(D_typed);
        }
    }, D);
}
#endif

#endif // DTYPE_UTILS_H
