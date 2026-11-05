#ifndef PARALLEL_CC_MATLAB_ADJACENCY_LIST_H_
#define PARALLEL_CC_MATLAB_ADJACENCY_LIST_H_

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

#include "mex.h"

namespace parallel_cc::matlab {

using AdjacencyList = std::vector<std::vector<int64_t>>;

inline void ValidateBinaryAdjacencyMatrix(const mxArray* input,
                                          int64_t max_vertices) {
  if ((!mxIsDouble(input) && !mxIsLogical(input)) || mxIsComplex(input)) {
    throw std::runtime_error(
        "Adjacency matrix must be a real double or logical matrix.");
  }
  if (mxGetNumberOfDimensions(input) != 2) {
    throw std::runtime_error("Adjacency matrix must be two-dimensional.");
  }
  const int64_t rows = static_cast<int64_t>(mxGetM(input));
  const int64_t columns = static_cast<int64_t>(mxGetN(input));
  if (rows != columns) {
    throw std::runtime_error("Adjacency matrix must be square.");
  }
  if (rows < 1) {
    throw std::runtime_error("Adjacency matrix must be nonempty.");
  }
  if (rows > max_vertices) {
    throw std::runtime_error(
        "Vertex count exceeds the configured NodeID range.");
  }
  if (mxIsLogical(input)) return;

  const double* values = mxGetPr(input);
  const size_t count = mxIsSparse(input)
      ? static_cast<size_t>(mxGetJc(input)[mxGetN(input)])
      : static_cast<size_t>(mxGetNumberOfElements(input));
  for (size_t i = 0; i < count; ++i) {
    if (!std::isfinite(values[i]) ||
        (values[i] != 0.0 && values[i] != 1.0)) {
      throw std::runtime_error(
          "Adjacency matrix must contain only finite binary values 0 or 1.");
    }
  }
}

// Converts a previously validated MATLAB binary adjacency matrix directly
// to column-oriented adjacency rows. Sparse CSC input remains sparse: no
// dense or Eigen intermediate is created. Every row is resized once to its
// exact degree and then filled in sorted vertex order, so no growth slack is
// introduced by push_back.
inline AdjacencyList ParseValidatedBinaryAdjacencyList(const mxArray* input) {
  if ((!mxIsDouble(input) && !mxIsLogical(input)) || mxIsComplex(input) ||
      mxGetNumberOfDimensions(input) != 2 || mxGetM(input) != mxGetN(input)) {
    throw std::runtime_error(
        "ParseValidatedBinaryAdjacencyList requires a validated matrix.");
  }

  const size_t n = static_cast<size_t>(mxGetN(input));
  std::vector<size_t> degrees(n, 0);
  if (mxIsSparse(input)) {
    const mwIndex* column_offsets = mxGetJc(input);
    if (mxIsLogical(input)) {
      const mxLogical* values = mxGetLogicals(input);
      for (size_t column = 0; column < n; ++column) {
        for (mwIndex index = column_offsets[column];
             index < column_offsets[column + 1]; ++index) {
          if (values[index]) ++degrees[column];
        }
      }
    } else {
      const double* values = mxGetPr(input);
      for (size_t column = 0; column < n; ++column) {
        for (mwIndex index = column_offsets[column];
             index < column_offsets[column + 1]; ++index) {
          if (values[index] != 0.0) ++degrees[column];
        }
      }
    }
  } else if (mxIsLogical(input)) {
    const mxLogical* values = mxGetLogicals(input);
    for (size_t column = 0; column < n; ++column) {
      for (size_t row = 0; row < n; ++row) {
        if (values[row + column * n]) ++degrees[column];
      }
    }
  } else {
    const double* values = mxGetPr(input);
    for (size_t column = 0; column < n; ++column) {
      for (size_t row = 0; row < n; ++row) {
        if (values[row + column * n] != 0.0) ++degrees[column];
      }
    }
  }

  AdjacencyList adjacency(n);
  for (size_t column = 0; column < n; ++column) {
    adjacency[column].resize(degrees[column]);
  }
  std::vector<size_t> positions(n, 0);

  if (mxIsSparse(input)) {
    const mwIndex* row_indices = mxGetIr(input);
    const mwIndex* column_offsets = mxGetJc(input);
    if (mxIsLogical(input)) {
      const mxLogical* values = mxGetLogicals(input);
      for (size_t column = 0; column < n; ++column) {
        for (mwIndex index = column_offsets[column];
             index < column_offsets[column + 1]; ++index) {
          if (values[index]) {
            adjacency[column][positions[column]++] =
                static_cast<int64_t>(row_indices[index]);
          }
        }
      }
    } else {
      const double* values = mxGetPr(input);
      for (size_t column = 0; column < n; ++column) {
        for (mwIndex index = column_offsets[column];
             index < column_offsets[column + 1]; ++index) {
          if (values[index] != 0.0) {
            adjacency[column][positions[column]++] =
                static_cast<int64_t>(row_indices[index]);
          }
        }
      }
    }
  } else if (mxIsLogical(input)) {
    const mxLogical* values = mxGetLogicals(input);
    for (size_t column = 0; column < n; ++column) {
      for (size_t row = 0; row < n; ++row) {
        if (values[row + column * n]) {
          adjacency[column][positions[column]++] =
              static_cast<int64_t>(row);
        }
      }
    }
  } else {
    const double* values = mxGetPr(input);
    for (size_t column = 0; column < n; ++column) {
      for (size_t row = 0; row < n; ++row) {
        if (values[row + column * n] != 0.0) {
          adjacency[column][positions[column]++] =
              static_cast<int64_t>(row);
        }
      }
    }
  }
  return adjacency;
}

}  // namespace parallel_cc::matlab

#endif  // PARALLEL_CC_MATLAB_ADJACENCY_LIST_H_
