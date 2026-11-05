//
// Created by Junda on 2025/5/16.
//
#ifndef CPP_GRAPH_UTILS_H
#define CPP_GRAPH_UTILS_H
#include <Eigen/Dense>
#include <Eigen/Sparse>
#include <variant>

// Define MatrixType as a variant that can hold any of the three supported matrix types
using MatrixType = std::variant<Eigen::MatrixXd, 
                               Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>,
                               Eigen::SparseMatrix<bool>>;

// Private implementation helpers for specific matrix types
namespace matrix_impl {
    // Process dense double matrix
    inline std::pair<Eigen::VectorXd, std::vector<std::vector<int64_t>>> 
    process_double_matrix(const Eigen::MatrixXd& matrix, const int64_t n) {
        Eigen::VectorXd DEG = Eigen::VectorXd::Zero(n);
        std::vector<std::vector<int64_t>> A(n);
        
        // Compute degrees
        for (int64_t i = 0; i < n; ++i) {
            DEG(i) = matrix.col(i).sum();
        }
        
        // Reserve space for adjacency lists
        for (int64_t i = 0; i < n; ++i) {
            A[i].reserve(static_cast<size_t>(DEG(i)));
        }
        
        // Build adjacency lists
        for (int64_t i = 0; i < n; ++i) {
            auto col = matrix.col(i);
            for (int64_t j = 0; j < n; ++j) {
                if (col(j) > 0) {
                    A[i].push_back(j);
                }
            }
        }
        
        return std::make_pair(DEG, A);
    }
    
    // Process dense bool matrix
    inline std::pair<Eigen::VectorXd, std::vector<std::vector<int64_t>>> 
    process_bool_matrix(const Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>& matrix, const int64_t n) {
        Eigen::VectorXd DEG = Eigen::VectorXd::Zero(n);
        std::vector<std::vector<int64_t>> A(n);
        
        // Compute degrees
        for (int64_t i = 0; i < n; ++i) {
            DEG(i) = matrix.col(i).cast<double>().sum();
        }
        
        // Reserve space for adjacency lists
        for (int64_t i = 0; i < n; ++i) {
            A[i].reserve(static_cast<size_t>(DEG(i)));
        }
        
        // Build adjacency lists
        for (int64_t i = 0; i < n; ++i) {
            auto col = matrix.col(i);
            for (int64_t j = 0; j < n; ++j) {
                if (col(j)) {
                    A[i].push_back(j);
                }
            }
        }
        
        return std::make_pair(DEG, A);
    }
    
    // Process sparse bool matrix
    inline std::pair<Eigen::VectorXd, std::vector<std::vector<int64_t>>> 
    process_sparse_bool_matrix(const Eigen::SparseMatrix<bool>& matrix, const int64_t n) {
        Eigen::VectorXd DEG = Eigen::VectorXd::Zero(n);
        std::vector<std::vector<int64_t>> A(n);
        
        // First pass: count degrees
        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<bool>::InnerIterator it(matrix, k); it; ++it) {
                if (it.value()) {
                    DEG(k) += 1.0;
                }
            }
        }
        
        // Reserve space for adjacency lists
        for (int64_t i = 0; i < n; ++i) {
            A[i].reserve(static_cast<size_t>(DEG(i)));
        }
        
        // Second pass: build adjacency lists
        for (int k = 0; k < matrix.outerSize(); ++k) {
            for (Eigen::SparseMatrix<bool>::InnerIterator it(matrix, k); it; ++it) {
                if (it.value()) {
                    A[k].push_back(it.row());
                }
            }
        }
        
        return std::make_pair(DEG, A);
    }
}

/**
 * Compute node degrees and adjacency list from an adjacency matrix of any supported type
 * 
 * @param adj_matrix The adjacency matrix (MatrixType variant)
 * @param n Number of nodes in the graph
 * @return A pair containing the degree vector and adjacency list
 */
inline std::pair<Eigen::VectorXd, std::vector<std::vector<int64_t>>> ComputeDegreeAndAdjList(
        const MatrixType& adj_matrix,
        const int64_t n) {
    
    if (std::holds_alternative<Eigen::MatrixXd>(adj_matrix)) {
        return matrix_impl::process_double_matrix(std::get<Eigen::MatrixXd>(adj_matrix), n);
    }
    else if (std::holds_alternative<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(adj_matrix)) {
        return matrix_impl::process_bool_matrix(std::get<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(adj_matrix), n);
    }
    else if (std::holds_alternative<Eigen::SparseMatrix<bool>>(adj_matrix)) {
        return matrix_impl::process_sparse_bool_matrix(std::get<Eigen::SparseMatrix<bool>>(adj_matrix), n);
    }
    
    // Should never reach here if all variant types are handled
    return std::make_pair(Eigen::VectorXd::Zero(n), std::vector<std::vector<int64_t>>(n));
}

// Helper extension methods for MatrixType
namespace matrix_ext {
    // Get number of rows
    inline int64_t rows(const MatrixType& mat) {
        if (std::holds_alternative<Eigen::MatrixXd>(mat)) {
            return static_cast<int64_t>(std::get<Eigen::MatrixXd>(mat).rows());
        }
        else if (std::holds_alternative<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(mat)) {
            return static_cast<int64_t>(std::get<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(mat).rows());
        }
        else if (std::holds_alternative<Eigen::SparseMatrix<bool>>(mat)) {
            return static_cast<int64_t>(std::get<Eigen::SparseMatrix<bool>>(mat).rows());
        }
        return 0;
    }
    
    // Get number of columns
    inline int64_t cols(const MatrixType& mat) {
        if (std::holds_alternative<Eigen::MatrixXd>(mat)) {
            return static_cast<int64_t>(std::get<Eigen::MatrixXd>(mat).cols());
        }
        else if (std::holds_alternative<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(mat)) {
            return static_cast<int64_t>(std::get<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(mat).cols());
        }
        else if (std::holds_alternative<Eigen::SparseMatrix<bool>>(mat)) {
            return static_cast<int64_t>(std::get<Eigen::SparseMatrix<bool>>(mat).cols());
        }
        return 0;
    }
    
    // Convert any MatrixType to dense double matrix
    inline Eigen::MatrixXd to_dense_double(const MatrixType& mat) {
        if (std::holds_alternative<Eigen::MatrixXd>(mat)) {
            return std::get<Eigen::MatrixXd>(mat);
        }
        else if (std::holds_alternative<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(mat)) {
            return std::get<Eigen::Matrix<bool, Eigen::Dynamic, Eigen::Dynamic>>(mat).cast<double>();
        }
        else if (std::holds_alternative<Eigen::SparseMatrix<bool>>(mat)) {
            const auto& sparse_mat = std::get<Eigen::SparseMatrix<bool>>(mat);
            Eigen::MatrixXd dense(sparse_mat.rows(), sparse_mat.cols());
            dense.setZero();
            for (int k = 0; k < sparse_mat.outerSize(); ++k) {
                for (Eigen::SparseMatrix<bool>::InnerIterator it(sparse_mat, k); it; ++it) {
                    if (it.value()) {
                        dense(it.row(), it.col()) = 1.0;
                    }
                }
            }
            return dense;
        }
        return Eigen::MatrixXd();
    }
}

#endif
