//
// Created by Junda on 2025/4/17.
//

#ifndef CPP_MEX_UTILS_H
#define CPP_MEX_UTILS_H
#include "mex.h"
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <variant>
#include "include/graph_utils.h" // Include for MatrixType definition

// Function to parse adjacency matrix from MATLAB input
Eigen::MatrixXd parseSparseDenseMatrix(const mxArray *matlabMatrix) {
    mwSize numRows = mxGetM(matlabMatrix);
    mwSize numCols = mxGetN(matlabMatrix);

    Eigen::MatrixXd adjMatrix(numRows, numCols);

    if (mxIsSparse(matlabMatrix)) {
        // Handle sparse matrix
        mwIndex* rowIndices = mxGetIr(matlabMatrix);
        mwIndex* colPointers = mxGetJc(matlabMatrix);

        // Initialize with zeros
        adjMatrix.setZero();

        if (mxIsLogical(matlabMatrix)) {
            // Handle sparse logical matrix
            mxLogical* logicalValues = mxGetLogicals(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    adjMatrix(rowIndices[idx], col) = static_cast<double>(logicalValues[idx]);
                }
            }
        } else {
            // Handle sparse double matrix
            double* values = mxGetPr(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    adjMatrix(rowIndices[idx], col) = values[idx];
                }
            }
        }
    } else {
        // Handle dense matrix
        if (mxIsLogical(matlabMatrix)) {
            // Convert logical to double
            mxLogical* logicalData = mxGetLogicals(matlabMatrix);
            for (mwSize i = 0; i < numRows; ++i) {
                for (mwSize j = 0; j < numCols; ++j) {
                    adjMatrix(i, j) = static_cast<double>(logicalData[i + j * numRows]);
                }
            }
        } else {
            // Map double matrix
            Eigen::Map<Eigen::MatrixXd> map(mxGetPr(matlabMatrix), numRows, numCols);
            adjMatrix = map;
        }
    }

    return adjMatrix;
}

Eigen::SparseMatrix<bool> parseSparseLogicalMatrix(const mxArray *matlabMatrix) {
    mwSize numRows = mxGetM(matlabMatrix);
    mwSize numCols = mxGetN(matlabMatrix);
    
    // Create triplets vector for building the sparse matrix
    std::vector<Eigen::Triplet<bool>> triplets;
    
    if (mxIsSparse(matlabMatrix)) {
        mwIndex* rowIndices = mxGetIr(matlabMatrix);
        mwIndex* colPointers = mxGetJc(matlabMatrix);
        
        // Calculate the number of non-zero elements for better reservation
        mwSize nnz = colPointers[numCols];
        triplets.reserve(nnz);

        if (mxIsLogical(matlabMatrix)) {
            // Handle sparse logical matrix
            mxLogical* logicalValues = mxGetLogicals(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    if (logicalValues[idx]) {
                        triplets.emplace_back(rowIndices[idx], col, true);
                    }
                }
            }
        } else {
            // Handle sparse double matrix
            double* values = mxGetPr(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    if (values[idx] != 0.0) {
                        triplets.emplace_back(rowIndices[idx], col, true);
                    }
                }
            }
        }
    } else {
        // Handle dense matrix (estimate triplets size as 10% of total elements)
        triplets.reserve(numRows * numCols / 10);
        
        if (mxIsLogical(matlabMatrix)) {
            // Handle dense logical matrix
            mxLogical* logicalData = mxGetLogicals(matlabMatrix);
            
            for (mwSize j = 0; j < numCols; ++j) {
                for (mwSize i = 0; i < numRows; ++i) {
                    if (logicalData[i + j * numRows]) {
                        triplets.emplace_back(i, j, true);
                    }
                }
            }
        } else {
            // Handle dense double matrix
            double* doubleData = mxGetPr(matlabMatrix);
            
            for (mwSize j = 0; j < numCols; ++j) {
                for (mwSize i = 0; i < numRows; ++i) {
                    if (doubleData[i + j * numRows] != 0.0) {
                        triplets.emplace_back(i, j, true);
                    }
                }
            }
        }
    }
    
    // Create sparse matrix from triplets
    Eigen::SparseMatrix<bool> sparseMatrix(numRows, numCols);
    sparseMatrix.setFromTriplets(triplets.begin(), triplets.end());
    return sparseMatrix;
}

/**
 * Parse an adjacency matrix from MATLAB input to a dense logical Eigen matrix.
 * This function handles both sparse and dense MATLAB matrices, as well as both logical and double types.
 */
Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> parseDenseLogicalMatrix(const mxArray *matlabMatrix) {
    mwSize numRows = mxGetM(matlabMatrix);
    mwSize numCols = mxGetN(matlabMatrix);

    Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> result(numRows, numCols);
    result.setZero();

    if (mxIsSparse(matlabMatrix)) {
        // Handle sparse matrix
        mwIndex* rowIndices = mxGetIr(matlabMatrix);
        mwIndex* colPointers = mxGetJc(matlabMatrix);

        if (mxIsLogical(matlabMatrix)) {
            // Handle sparse logical matrix
            mxLogical* logicalValues = mxGetLogicals(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    if (logicalValues[idx]) {
                        result(rowIndices[idx], col) = true;
                    }
                }
            }
        } else {
            // Handle sparse double matrix
            double* values = mxGetPr(matlabMatrix);

            for (mwIndex col = 0; col < numCols; ++col) {
                for (mwIndex idx = colPointers[col]; idx < colPointers[col + 1]; ++idx) {
                    if (values[idx] != 0.0) {
                        result(rowIndices[idx], col) = true;
                    }
                }
            }
        }
    } else {
        // Handle dense matrix
        if (mxIsLogical(matlabMatrix)) {
            // Convert logical to bool
            mxLogical* logicalData = mxGetLogicals(matlabMatrix);
            for (mwSize i = 0; i < numRows; ++i) {
                for (mwSize j = 0; j < numCols; ++j) {
                    result(i, j) = logicalData[i + j * numRows];
                }
            }
        } else {
            // Map double matrix to bool
            double* doubleData = mxGetPr(matlabMatrix);
            for (mwSize i = 0; i < numRows; ++i) {
                for (mwSize j = 0; j < numCols; ++j) {
                    result(i, j) = doubleData[i + j * numRows] != 0.0;
                }
            }
        }
    }

    return result;
}

/**
 * Parse an adjacency matrix from MATLAB input to either a sparse or dense logical Eigen matrix
 * based on the density of the matrix.
 * 
 * @param matlabMatrix The input MATLAB matrix
 * @return A MatrixType variant containing the appropriate matrix format
 */
MatrixType parseAdjacencyMatrix(const mxArray *matlabMatrix) {
    mwSize numRows = mxGetM(matlabMatrix);
    mwSize numCols = mxGetN(matlabMatrix);
    double totalElements = static_cast<double>(numRows) * static_cast<double>(numCols);
    
    // Calculate the original non-zero count
    mwSize originalNonZeros = 0;
    
    if (mxIsSparse(matlabMatrix)) {
        // For sparse matrices, we can directly get the number of non-zeros
        mwIndex* colPointers = mxGetJc(matlabMatrix);
        originalNonZeros = colPointers[numCols]; // Total number of non-zero elements
    } else {
        // For dense matrices, we need to count non-zeros
        if (mxIsLogical(matlabMatrix)) {
            mxLogical* logicalData = mxGetLogicals(matlabMatrix);
            for (mwSize i = 0; i < numRows * numCols; ++i) {
                if (logicalData[i]) originalNonZeros++;
            }
        } else {
            double* doubleData = mxGetPr(matlabMatrix);
            for (mwSize i = 0; i < numRows * numCols; ++i) {
                if (doubleData[i] != 0.0) originalNonZeros++;
            }
        }
    }
    
    // Calculate density of the matrix
    double density = static_cast<double>(originalNonZeros) / totalElements;
    
    // Choose sparse or dense format based on density
    MatrixType result;
    
    if (density < 0.15) {
        Eigen::SparseMatrix<bool> sparseResult = parseSparseLogicalMatrix(matlabMatrix);
        
        // Check if the number of non-zeros matches
        if (sparseResult.nonZeros() != originalNonZeros) {
            mexWarnMsgTxt("Warning: Number of edges in parsed sparse matrix does not match original matrix.");
        }
        
        result = sparseResult;
    } else {
        // For denser matrices, use a dense bool matrix
        Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic> denseResult = parseDenseLogicalMatrix(matlabMatrix);
        
        // Count non-zeros in dense matrix
        mwSize parsedNonZeros = 0;
        for (int i = 0; i < denseResult.rows(); ++i) {
            for (int j = 0; j < denseResult.cols(); ++j) {
                if (denseResult(i, j)) parsedNonZeros++;
            }
        }
        
        // Check if the number of non-zeros matches
        if (parsedNonZeros != originalNonZeros) {
            mexWarnMsgTxt("Warning: Number of edges in parsed dense matrix does not match original matrix.");
        }
        
        result = denseResult;
    }
    
    return result;
}

#endif //CPP_MEX_UTILS_H