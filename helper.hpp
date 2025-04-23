#ifndef HELPER_HPP
#define HELPER_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>
#include <random>

using namespace std;
using namespace Eigen;

// Function to print the results of QR factorization (Q, R) and the vector tau
void Nice_QR_output(float *h_A, float *h_tau, float *h_R, float *h_Q, int m, int n) {
    // Print the vector tau
    std::cout << "Vector tau:\n";
    for (int i = 0; i < n; i++) {
        std::cout << h_tau[i] << std::endl;
    }

    // Convert h_A to Eigen matrix for easy manipulation and display
    MatrixXd A(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = h_A[j * m + i]; // Converting from column-major to Eigen's row-major
        }
    }
    std::cout << "Matrix A (after QR factorization):\n" << A << "\n\n";

    // Convert h_tau to Eigen vector for forming Q
    VectorXd tau(n);
    for (int i = 0; i < n; ++i) {
        tau(i) = h_tau[i];
    }

    // Convert h_Q to Eigen matrix for easy manipulation
    MatrixXd Q(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Q(i, j) = h_Q[j * m + i]; // Converting from column-major to Eigen's row-major
        }
    }
    std::cout << "Matrix Q:\n" << Q << "\n\n";

    // Construct R from h_R and extract the upper triangular matrix
    MatrixXd Rtemp(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            Rtemp(i, j) = h_R[j * m + i]; // Converting from column-major to Eigen's row-major
        }
    }
    MatrixXd R = Rtemp.topRows(n).triangularView<Upper>();  // Only take upper triangular part
    std::cout << "Matrix R:\n" << R << "\n\n";

    // Verify the result by multiplying Q and R to reconstruct A
    MatrixXd QR = Q * R;
    std::cout << "Reconstructed matrix (Q * R):\n" << QR << "\n\n";
}

// Function to print the result of Cholesky factorization (R)
void Nice_Chol_output(double *h_A, double *h_R, int n) {
    // Convert h_R to Eigen matrix for easy manipulation
    MatrixXd A(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            A(i, j) = h_R[j * n + i]; // Converting from column-major to Eigen's row-major
        }
    }

    // Extract the upper triangular matrix R from A
    MatrixXd R = A.triangularView<Upper>();
    std::cout << "Matrix R (after Cholesky factorization):\n" << R << "\n\n";
}


// Function to generate a random m x n matrix and return it as a 1D vector
std::vector<double> generateRandomMatrix(int m, int n, double minVal = 0.0, double maxVal = 1.0) {
    // Random number generator setup
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dis(minVal, maxVal);

    // Create an Eigen matrix and fill it with random values
    Eigen::MatrixXd mat(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            mat(i, j) = dis(gen);
        }
    }

    // Flatten the matrix into a 1D vector (column-major order)
    std::vector<double> matrixVector(m * n);
    for (int j = 0; j < n; ++j) {
        for (int i = 0; i < m; ++i) {
            matrixVector[i + j * m] = mat(i, j); // Column-major order
        }
    }

    return matrixVector;
}


// Function to print the matrix from a 1D vector
void printMatrixCMO(const std::vector<double>& matrixVector, int m, int n) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << matrixVector[i + j * m] << " "; // Column-major order
        }
        std::cout << "\n";
    }
}



#endif
