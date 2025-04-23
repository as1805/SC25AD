
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/copy.h>
//#include <thrust/system.h>
#include <cublas_v2.h>
#include <cuda_runtime.h>
#include <iostream>
#include <thrust/for_each.h>
#include <cusolverDn.h>
#include <cufft.h>
#include <cufftXt.h>
#include <cuComplex.h>
#include <assert.h>
#include <thrust/complex.h>
#include <thrust/random.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/gather.h>
#include <thrust/sequence.h>
#include <thrust/iterator/counting_iterator.h>
#include <tuple>
#include <cmath> 
#include<cusparse.h>
#include <type_traits>
#include <cuda_fp16.h>
#include <typeinfo>


#include "gpu_util.hpp"
#include "timer_gpu.hpp"
#include "rand.hpp"



// Error-checking macros
#define CUDA_CHECK_ERROR(ans) { gpuAssert((ans), __FILE__, __LINE__); }
#define CUBLAS_CHECK_ERROR(stat) { if (stat != CUBLAS_STATUS_SUCCESS) { std::cerr << "cuBLAS error at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1); } }
#define CUSOLVER_CHECK_ERROR(stat) { if (stat != CUSOLVER_STATUS_SUCCESS) { std::cerr << "cuSolver error at " << __FILE__ << ":" << __LINE__ << std::endl; exit(1); } }
#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return;                                                   \
    }                                                                          \
}

#define CHECK_CUSPARSE(func)                                                   \
{                                                                              \
    cusparseStatus_t status = (func);                                          \
    if (status != CUSPARSE_STATUS_SUCCESS) {                                   \
        printf("CUSPARSE API failed at line %d with error: %s (%d)\n",         \
               __LINE__, cusparseGetErrorString(status), status);              \
        return  ;                                                  \
    }                                                                          \
}


// Matrix multiplication on the GPU using Thrust and cuBLAS
void gpu_Smatrix_multiply(const float *h_A, const float *h_B, float *h_C, int m, int n, int k) {
    float alpha = 1.0f, beta = 0.0f;

    // Allocate device memory using Thrust
    thrust::device_vector<float> d_A(h_A, h_A + m * k); // m x k
    thrust::device_vector<float> d_B(h_B, h_B + k * n); // k x n
    thrust::device_vector<float> d_C(m * n);            // m x n result matrix

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    TimerGPU t;
    // Perform matrix multiplication: d_C = alpha * d_A * d_B + beta * d_C
    t.start();
    cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                n, m, k, &alpha, 
                thrust::raw_pointer_cast(d_B.data()), n, 
                thrust::raw_pointer_cast(d_A.data()), k, 
                &beta, 
                thrust::raw_pointer_cast(d_C.data()), n);

    t.stop();

std::cout<<"Single Precision Matrix multiplication \n";
t.show_elapsed_time();

    // Copy result from device to host
    thrust::copy(d_C.begin(), d_C.end(), h_C);

    // Cleanup
    cublasDestroy(handle);
}

void gpu_Dmatrix_multiply(const double *h_A, const double *h_B, double *h_C, int m, int n, int k) {
    double alpha = 1.0, beta = 0.0;

    // Allocate device memory using Thrust
    thrust::device_vector<double> d_A(h_A, h_A + m * k); // m x k
    thrust::device_vector<double> d_B(h_B, h_B + k * n); // k x n
    thrust::device_vector<double> d_C(m * n);            // m x n result matrix

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    TimerGPU t;
// Perform matrix multiplication: d_C = alpha * d_A * d_B + beta * d_C
    t.start(); //timer
    cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,  
                m, n, k, &alpha, 
                thrust::raw_pointer_cast(d_A.data()), m, 
                thrust::raw_pointer_cast(d_B.data()), k, 
                &beta, 
                thrust::raw_pointer_cast(d_C.data()), m);
t.stop();

//std::cout<<"Double Precision Matrix multiplication \n";
//t.show_elapsed_time();
    // Copy result from device to host
    thrust::copy(d_C.begin(), d_C.end(), h_C);

    // Cleanup
    cublasDestroy(handle);
}

void gpu_gram_matrix(const double *h_A, double *h_G, int m, int n) {
    double alpha = 1.0, beta = 0.0;

    // Allocate device memory using Thrust
    thrust::device_vector<double> d_A(h_A, h_A + m * n); // m x n
    thrust::device_vector<double> d_G(n * n);            // n x n result matrix (Gram matrix)
    

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);

    TimerGPU t;
  
    // Perform matrix multiplication: G = A^T * A
    t.start(); // Start timer
    cublasDgemm(handle, CUBLAS_OP_T, CUBLAS_OP_N,  
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_A.data()), m, 
                thrust::raw_pointer_cast(d_A.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G.data()), n);
    t.stop();

    std::cout << "Double Precision Gram matrix computation\n";
    t.show_elapsed_time();

    // Copy result from device to host
    thrust::copy(d_G.begin(), d_G.end(), h_G);

    // Cleanup
    cublasDestroy(handle);
}


// Perform QR factorization on the GPU using cuSolver
void gpu_QR(float *h_A, float *h_tau, float *h_R, float *h_Q, int m, int n) {
    const int lda = m;
    int lwork = 0, info = 0;
    int lwork_orgqr = 0;
    int *devInfo = nullptr;
    float *d_work = nullptr, *d_tau = nullptr;

    TimerGPU t; //for timer
    // Allocate device memory for A, Q, R, and tau using Thrust
    thrust::device_vector<float> d_A(h_A, h_A + m * n);  // m x n for A

    thrust::device_vector<float> d_R(m * n);               // m x n for R
    std::vector<float> tau(n, 0);
    
    // Allocate device memory for error information (devInfo) and tau
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(float)));

    // Create cuSolver handle
    cusolverDnHandle_t handle;
    cusolverStatus_t cusolverStat = cusolverDnCreate(&handle);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization error" << std::endl;
        goto cleanup;
    }
     
    //need to ask about the actualtime start. Should I count the query time? or is this "fixed"? 
    // Query workspace size for geqrf
    cusolverStat = cusolverDnSgeqrf_bufferSize(handle, m, n, thrust::raw_pointer_cast(d_A.data()), lda, &lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size" << std::endl;
        goto cleanup;
    }

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(float)));
     t.start();
    // Compute QR factorization
    cusolverStat = cusolverDnSgeqrf(handle, m, n, thrust::raw_pointer_cast(d_A.data()), lda, d_tau, d_work, lwork, devInfo);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during QR factorization" << std::endl;
        goto cleanup;
    }
    t.stop();

    std::cout << "Single Precision QR\n";
    t.show_elapsed_time();
    // Copy tau and devInfo back to host
    CUDA_CHECK(cudaMemcpyAsync(tau.data(), d_tau, sizeof(float) * tau.size(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpyAsync(&info, devInfo, sizeof(int), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaStreamSynchronize(0));

    std::printf("after Sgeqrf: info = %d\n", info);
    if (info < 0) {
        std::printf("%d-th parameter is wrong \n", -info);
        exit(1);
    }

    /*std::printf("tau = (matlab base-1)\n");
    print_float_matrix(n, 1, tau.data(), lda);
    std::printf("=====\n");*/

    // Copy the R matrix to host (upper triangular part of A)
    thrust::copy(d_A.begin(), d_A.end(), h_R);

    // Copy tau back to host
    cudaMemcpy(h_tau, d_tau, n * sizeof(float), cudaMemcpyDeviceToHost);

    //allocate memory for Q forming
cusolverStat = cusolverDnSorgqr_bufferSize(handle, m, n, n, thrust::raw_pointer_cast(d_A.data()), lda, d_tau, &lwork_orgqr);
if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
    std::cerr << "Error querying orgqr workspace size" << std::endl;
    goto cleanup;
}

CUDA_CHECK(cudaMalloc((void**)&d_work, lwork_orgqr * sizeof(float)));
   
    // Generate Q by overwriting A with the result of the QR factorization
    cusolverStat = cusolverDnSorgqr(handle, m, n, n, thrust::raw_pointer_cast(d_A.data()), lda, d_tau, d_work, lwork_orgqr, devInfo);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during orgqr to generate Q" << std::endl;
        goto cleanup;
    }

    // Copy the Q matrix to host
    thrust::copy(d_A.begin(), d_A.end(), h_Q);

cleanup:
    // Cleanup resources
    cusolverDnDestroy(handle);
    cudaFree(devInfo);
    cudaFree(d_tau);
    cudaFree(d_work);
}

// Perform Cholesky factorization on the GPU using cuSolver
void gpu_Chol(double *h_A, double *h_R, int n) {
    cusolverDnHandle_t handle;
   
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER; // Cholesky uses upper triangular matrix

    int lwork = 0;
    int *d_info = nullptr;
    double *d_work = nullptr;
    
    // Allocate device memory for A and R using Thrust
    thrust::device_vector<double> d_A(h_A, h_A + n * n); // n x n for A
    thrust::device_vector<double> d_R(n * n);             // n x n for R

    // Create cuSolver handle
    cusolverStatus_t cusolverStat = cusolverDnCreate(&handle);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization error" << std::endl;
        return;
    }
   
    // Query workspace size for Cholesky factorization
    cusolverStat = cusolverDnDpotrf_bufferSize(handle, uplo, n, thrust::raw_pointer_cast(d_A.data()), n, &lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size for Cholesky" << std::endl;
        goto cleanup;
    }

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    // Perform Cholesky factorization
    cusolverStat = cusolverDnDpotrf(handle, uplo, n, thrust::raw_pointer_cast(d_A.data()), n, d_work, lwork, d_info);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during Cholesky factorization" << std::endl;
        goto cleanup;
    }
   

    // Copy the result (R) from device to host
    thrust::copy(d_A.begin(), d_A.end(), h_R);

cleanup:
    // Cleanup resources
    cusolverDnDestroy(handle);
    cudaFree(d_info);
    cudaFree(d_work);
}

void gpu_Chol_solve(double *h_G, double *h_R, int n, double *h_A, int m,double *h_QT ) {
    cusolverDnHandle_t handle;
    cublasHandle_t cublashandle;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER; // Cholesky uses upper triangular matrix. 

    int lwork = 0;
    int *d_info = nullptr;
    double *d_work = nullptr;

    int ldg=n; //since G is n by n 
    int lda= m; // since A is m by n 
    int ldat=n;
    int nrhs =n;

    double alpha =1.0; //constants used for transposition and solve
    double beta =0.0; 

    TimerGPU t; 

    // Allocate device memory for A, G, and R using Thrust
    thrust::device_vector<double> d_G(h_G, h_G + n * n); // n x n for A^TA
    //thrust::device_vector<double> d_R(n * n);             // n x n for R
    thrust:: device_vector<double> d_A(h_A, h_A+m*n); // m x n for A
   thrust::device_vector<double> d_AT(m * n);             // n x n for R

    // Create cuSolver and cuBLAS status check
    cusolverStatus_t cusolverStat = cusolverDnCreate(&handle);
       cublasStatus_t cublasStat = cublasCreate(&cublashandle);

    if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization error" << std::endl;
        return;
    }
 

    // Query workspace size for Cholesky factorization
        cusolverStat = cusolverDnDpotrf_bufferSize(handle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, &lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size for Cholesky" << std::endl;
        goto cleanup;
    }

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    // Perform Cholesky factorization
    //cusolverStat = cusolverDnDpotrf(handle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, d_work, lwork, d_info);
    t.start();
    cusolverStat = cusolverDnDpotrf(handle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, d_work, lwork, d_info);
    t.stop();
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during Cholesky factorization" << std::endl;
        goto cleanup;
    }
        std::cout << "Double Precision Cholesky \n";
    t.show_elapsed_time();

    //copy R before transpose
    thrust::copy(d_G.begin(), d_G.end(), h_R); // note here that this copies the whole matrix 


   
   

    //Here we are going to solve a bunch of linear systems
    t.start();

   cublasStat=cublasDgeam(cublashandle,
                          CUBLAS_OP_T , CUBLAS_OP_N ,
                          n, m,
                          &alpha,
                          thrust::raw_pointer_cast(d_A.data()), m,
                          &beta,
                          thrust::raw_pointer_cast(d_A.data()), m,
                        thrust::raw_pointer_cast(d_AT.data()),  n);
    t.stop();
  std::cout << "Double Precision Transposition\n";

    t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);

    t.stop();
    std::cout << "Double Precision Triangular solve\n";
    t.show_elapsed_time();

    //Copy the result (Q^T) from device to host
    
    thrust::copy(d_AT.begin(), d_AT.end(), h_QT); //the solved QT
   
    cleanup:
    // Cleanup resources
    cusolverDnDestroy(handle);
    cublasDestroy(cublashandle);
    cudaFree(d_info);
    cudaFree(d_work);
}

void gpu_CholQR(double *h_A, double *h_R, int n, double *h_Q, int m,double&time ) {
    int lwork = 0;
    int *d_info = nullptr;
    double *d_work = nullptr;

    int ldg=n; //since G is n by n 
    int lda= m; // since A is m by n 
    int ldat=n;
    int nrhs =n;

    double alpha =1.0; //constants used for transposition and solve
    double beta =0.0; 
    //declare timer
    TimerGPU t;

    // Create cuBLAS and cuSolver handle
    cublasHandle_t cublashandle;
    cusolverDnHandle_t cusolverhandle;
  
    cusolverStatus_t cusolverStat = cusolverDnCreate(&cusolverhandle);
    cublasStatus_t cublasStat = cublasCreate(&cublashandle);

  if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization error" << std::endl;
        return;
    }
    // determine the orientation
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

 // Allocate device memory for A, G, and R using Thrust
    thrust::device_vector<double> d_G(n * n); // n x n for A^TA
    thrust::device_vector<double> d_R(n * n);             // n x n for R
    thrust:: device_vector<double> d_A(h_A, h_A+m*n); // m x n for A
   thrust::device_vector<double> d_AT(n * m);             // m x n for AT
   thrust::device_vector<double> d_Q(m*n);                 // m x n for Q


    //query the workspaces for Cholesky and linear solve

  // Query workspace size for Cholesky factorization
        cusolverStat = cusolverDnDpotrf_bufferSize(cusolverhandle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, &lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size for Cholesky" << std::endl;
        goto cleanup;
    }

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    // Perform matrix multiplication: G = A^T * A
    t.start(); // Start timer
    cublasDgemm(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N,  
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_A.data()), m, 
                thrust::raw_pointer_cast(d_A.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G.data()), n);
    //t.stop();

    // Perform Cholesky factorization
    //t.start();
    cusolverStat = cusolverDnDpotrf(cusolverhandle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, d_work, lwork, d_info);
    //t.stop();
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during Cholesky factorization" << std::endl;
        goto cleanup;
    }

   cublasStat=cublasDgeam(cublashandle,
                          CUBLAS_OP_T , CUBLAS_OP_N ,
                          n, m,
                          &alpha,
                          thrust::raw_pointer_cast(d_A.data()), m,
                          &beta,
                          thrust::raw_pointer_cast(d_A.data()), m,
                        thrust::raw_pointer_cast(d_AT.data()),  n);
    //t.stop();


    //t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
    t.stop();
        //std::cout << "Total CholeskyQR time\n";
        //t.show_elapsed_time();
    time=t.elapsed_time();

    //here we do clean up via transposition and prepare for copying back to the host

    /*cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         nullptr,                  // No second matrix
                         n, 
                         thrust::raw_pointer_cast(d_Q.data()), n);  // Output matrix Q (ldC = m)*/
    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 



    //copy things back to the host: 
    thrust::copy(d_G.begin(), d_G.end(), h_R);
    //thrust::copy(d_AT.begin(), d_AT.end(), h_Q);
    thrust::copy(d_Q.begin(),d_Q.end(),h_Q);
    //thrust::copy(d_G.begin(), d_G.end(), h_R);

    cleanup:
    // Cleanup resources
    cusolverDnDestroy(cusolverhandle);
    cublasDestroy(cublashandle);
    cudaFree(d_info);
    cudaFree(d_work);
}

void gpu_QRD(double *h_A, double *h_tau, double *h_R, double *h_Q, int m, int n,double& time) {
    const int lda = m;
    int lwork = 0, info = 0;
    int lwork_orgqr = 0;
    int *devInfo = nullptr;
    double *d_work = nullptr, *d_tau = nullptr;

    TimerGPU t;
    
    // Allocate device memory for A using Thrust
    thrust::device_vector<double> d_A(h_A, h_A + m * n);  // m x n for A

    // Only allocate device memory for d_tau
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(double)));

    // Allocate device memory for devInfo (for error handling)
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    cusolverDnHandle_t handle;
    cusolverStatus_t cusolverStat = cusolverDnCreate(&handle);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization error" << std::endl;
        goto cleanup;
    }

    // Query workspace size for geqrf
    cusolverStat = cusolverDnDgeqrf_bufferSize(handle, m, n, thrust::raw_pointer_cast(d_A.data()), lda, &lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size" << std::endl;
        goto cleanup;
    }

    // Allocate workspace memory for geqrf
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    t.start();
    // Compute QR factorization
    cusolverStat = cusolverDnDgeqrf(handle, m, n, thrust::raw_pointer_cast(d_A.data()), lda, d_tau, d_work, lwork, devInfo);
    t.stop();

    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during QR factorization" << std::endl;
        goto cleanup;
    }

    //std::cout << "Double Precision QR time:\n";
    //t.show_elapsed_time();
    time=t.elapsed_time();
    // Copy the R matrix (upper triangular part) from A
    thrust::copy(d_A.begin(), d_A.begin() + n * n, h_R);  // Only copy the upper triangular part

    // Allocate workspace for orgqr
    cusolverStat = cusolverDnDorgqr_bufferSize(handle, m, n, n, thrust::raw_pointer_cast(d_A.data()), lda, d_tau, &lwork_orgqr);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying orgqr workspace size" << std::endl;
        goto cleanup;
    }

    // Allocate workspace for orgqr
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork_orgqr * sizeof(double)));

    // Generate Q by overwriting A with the result of the QR factorization
    cusolverStat = cusolverDnDorgqr(handle, m, n, n, thrust::raw_pointer_cast(d_A.data()), lda, d_tau, d_work, lwork_orgqr, devInfo);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during orgqr to generate Q" << std::endl;
        goto cleanup;
    }

    // Copy the Q matrix to host
    thrust::copy(d_A.begin(), d_A.end(), h_Q);

cleanup:
    // Cleanup resources
    if (handle) cusolverDnDestroy(handle);
    if (devInfo) cudaFree(devInfo);
    if (d_tau) cudaFree(d_tau);
    if (d_work) cudaFree(d_work);
}

struct upper_triangular_copy
{
    int m;
    int n;
    double *old_R;  // source pointer
    double *new_R;  // destination pointer

    // Constructor to initialize the pointers
    __host__ __device__
    upper_triangular_copy(double *old_R_ptr, double *new_R_ptr,int m, int n) : old_R(old_R_ptr), new_R(new_R_ptr),m(m),n(n) {}
  __host__ __device__
  void operator()(int x)
  {
    int row= x/n;
    int col =x %n ;

    if(row<=col && row<n){

        new_R[row+n*col]=old_R[row + m * col];
    }
  }
};


void gpu_CholQR2(double *h_A, double *h_R, int n, double *h_Q, int m, double& time,double* operationtimes ) {
    int lwork = 0;
    int *d_info = nullptr;
    double *d_work = nullptr;

    int ldg=n; //since G is n by n 
    int lda= m; // since A is m by n 
    int ldat=n;
    int nrhs =n;

    double alpha =1.0; //constants used for transposition and solve
    double beta =0.0; 
    //declare timer
    TimerGPU t;
     TimerGPU t1;

    // Create cuBLAS and cuSolver handle
    cublasHandle_t cublashandle;
    cusolverDnHandle_t cusolverhandle;
  
    cusolverStatus_t cusolverStat = cusolverDnCreate(&cusolverhandle);
    cublasStatus_t cublasStat = cublasCreate(&cublashandle);

  if (cublasStat != CUBLAS_STATUS_SUCCESS) {
        std::cerr << "CUBLAS initialization error" << std::endl;
        std::cerr<<"Code "<<cublasStat<<std::endl;
        return;
    }
    // determine the orientation
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

 // Allocate device memory for A, G, and R using Thrust
    thrust::device_vector<double> d_G(n * n); // n x n for A^TA
    thrust::device_vector<double> d_G1(n * n); // n x n for Q^TQ
    thrust::device_vector<double> d_R(n * n);             // n x n for R
    thrust:: device_vector<double> d_A(h_A, h_A+m*n); // m x n for A
   thrust::device_vector<double> d_AT(n * m);             // m x n for AT
   thrust::device_vector<double> d_Q(m*n);                 // m x n for Q

   thrust::device_vector<double> d_R1TR(n*n); //will be upper triangular versions of the 2 Rs
    thrust::device_vector<double> d_R2TR(n*n, 0.0); 
     thrust::counting_iterator<unsigned int> index(0);

    //query the workspaces for Cholesky and linear solve

  // Query workspace size for Cholesky factorization
        cusolverStat = cusolverDnDpotrf_bufferSize(cusolverhandle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, &lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size for Cholesky" << std::endl;
        goto cleanup;
    }

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    // Perform matrix multiplication: G = A^T * A
    t1.start(); // Start timer
    t.start();
    cublasDgemm(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N,  
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_A.data()), m, 
                thrust::raw_pointer_cast(d_A.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G.data()), n);
    t.stop();
    operationtimes[0]=t.elapsed_time();//tracking the time of various operations

    // Perform Cholesky factorization
    t.start();
    cusolverStat = cusolverDnDpotrf(cusolverhandle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, d_work, lwork, d_info);
    t.stop();
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during Cholesky factorization" << std::endl;
        goto cleanup;
    }
    operationtimes[1]=t.elapsed_time();//tracking the time of various operations

   cublasStat=cublasDgeam(cublashandle,
                          CUBLAS_OP_T , CUBLAS_OP_N ,
                          n, m,
                          &alpha,
                          thrust::raw_pointer_cast(d_A.data()), m,
                          &beta,
                          thrust::raw_pointer_cast(d_A.data()), m,
                        thrust::raw_pointer_cast(d_AT.data()),  n);
    //t.stop();


    t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);

    t.stop();
    operationtimes[2]=t.elapsed_time();//tracking the time of various operations
    //here we do clean up via transposition and prepare for CholQR2





    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 

    //We now need to do CholQR again, but on d_Q

    //forms Q^TQ
    t.start();
    cublasDgemm(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N,  
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G1.data()), n);
    t.stop();
     operationtimes[0]=operationtimes[0]+t.elapsed_time();

        //Does Cholesky
    t.start();
    cusolverStat = cusolverDnDpotrf(cusolverhandle, uplo, n, thrust::raw_pointer_cast(d_G1.data()), n, d_work, lwork, d_info);
    t.stop();
     operationtimes[1]=operationtimes[1]+t.elapsed_time();
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during second Cholesky factorization" << std::endl;
        goto cleanup;
    }
    t.start();
    //Does linear solve again, note here that d_AT holds QO^T
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G1.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
   t.stop();
    operationtimes[2]+=t.elapsed_time();
   // std::cout<<"QR2 triangular solve\n";
   // t.show_elapsed_time();


   // t.start();
    thrust::for_each(index,index+(m*n),upper_triangular_copy(thrust::raw_pointer_cast(d_G.data()),thrust::raw_pointer_cast(d_R1TR.data()),n,n));
    thrust::for_each(index,index+(m*n),upper_triangular_copy(thrust::raw_pointer_cast(d_G1.data()),thrust::raw_pointer_cast(d_R2TR.data()),n,n));
   // t.stop();
   // std::cout<<"R and G clean up to make upper triangular\n";
   // t.show_elapsed_time();
   
         //Recover original R 
         t.start();
 cublasDgemm(cublashandle, CUBLAS_OP_N, CUBLAS_OP_N,  //note that we really only care about the upper triangular part, and since they are both square it works out nicely that way ( the g1 needs to be transposed because of an earloer transposition.)
                n, n, n, &alpha, 
                thrust::raw_pointer_cast(d_R2TR.data()), n, 
                thrust::raw_pointer_cast(d_R1TR.data()), n, 
                &beta, 
                thrust::raw_pointer_cast(d_R.data()), n);
        t.stop();
         operationtimes[0]=operationtimes[0]+t.elapsed_time();
    t1.stop();
   // std::cout<<d_R[1];
    //std::cout << "Total CholeskyQR2 time\n";
    //t.show_elapsed_time();
    time=t1.elapsed_time();
    //turns the Q back to right transpose
    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 

    
    //copy things back to the host: 
    thrust::copy(d_R.begin(), d_R.end(), h_R);
    //thrust::copy(d_AT.begin(), d_AT.end(), h_Q);
    thrust::copy(d_Q.begin(),d_Q.end(),h_Q);
    //thrust::copy(d_G.begin(), d_G.end(), h_R);

    cleanup:
      // Free device memory
    if (d_info != nullptr) {
        cudaFree(d_info);
        d_info = nullptr;
    }

    if (d_work != nullptr) {
        cudaFree(d_work);
        d_work = nullptr;
    }

    // Destroy cuBLAS handle
    if (cublashandle) {
        cublasDestroy(cublashandle);
        cublashandle = nullptr;
    }

    // Destroy cuSolver handle
    if (cusolverhandle) {
        cusolverDnDestroy(cusolverhandle);
        cusolverhandle = nullptr;
    }

    // Clear and release Thrust device vectors
    d_G.clear();
    d_G.shrink_to_fit();

    d_G1.clear();
    d_G1.shrink_to_fit();

    d_R.clear();
    d_R.shrink_to_fit();

    d_A.clear();
    d_A.shrink_to_fit();

    d_AT.clear();
    d_AT.shrink_to_fit();

    d_Q.clear();
    d_Q.shrink_to_fit();

    d_R1TR.clear();
    d_R1TR.shrink_to_fit();

    d_R2TR.clear();
    d_R2TR.shrink_to_fit();
}


void gpu_fft_w_sampling(float *h_A, int m, int n1,int* d_S,int c) {
    const int numRows = m;       // Number of rows (N_VEC)
    const int numCols = n1;      // Number of columns
    const int halfRows = numRows / 2 + 1;  // Output size for R2C   
          int rows_per_col = numRows / 2 + 1;
    TimerGPU t;

    // Host and device vectors
    thrust::host_vector<float> h_input(h_A, h_A + numRows * numCols);
    thrust::device_vector<float> d_floatInput = h_input;
    thrust::device_vector<cufftComplex> d_complexOutput(numCols * halfRows);

    // Host vector to store real parts of selected frequencies
    thrust::host_vector<float> h_selected;



    cufftHandle fftPlan;
    int n[1] = {numRows}; // FFT size (rows per column)
    cufftResult result = cufftPlanMany(
        &fftPlan,
        1,           // Rank of the transform (1D)
        n,           // Dimensions of the transform
        n, 1, numRows,  // Input layout (inembed, istride, idist)
        n, 1, halfRows,  // Output layout (onembed, ostride, odist)
        CUFFT_R2C,   // Complex-to-complex transform
        numCols      // Batch size (number of columns)
    );

    assert(result == CUFFT_SUCCESS);

    // Execute FFT
    t.start();
    cufftExecR2C(
        fftPlan, 
        thrust::raw_pointer_cast(d_floatInput.data()), 
        thrust::raw_pointer_cast(d_complexOutput.data())
    );
    t.stop();
    std::cout << "Real to Complex FFT:\n";
    t.show_elapsed_time();



    t.start();
    for (int col = 0; col < numCols; ++col) {
        for (int i = 0; i < c; ++i) {
            int row = d_S[i]; 
       
            if (row <= numRows / 2) {
                // Positive frequency
                cufftComplex val = d_complexOutput[row + col * rows_per_col];
                h_selected.push_back(val.x);  // Take real part
            
            } else {
                // Negative frequency
                int neg_index = numRows - row;  // Map to positive equivalent
                cufftComplex val = d_complexOutput[neg_index + col * rows_per_col];
                h_selected.push_back(val.x);  // Take real part (conjugate symmetry)
              
            }
        }
    }
    t.stop();
    std::cout<<"Sampling time\n";
    t.show_elapsed_time();
    // Print the selected real parts
    std::cout << "Selected real parts (from full spectrum):\n";
    for (size_t i = 0; i < h_selected.size(); ++i) {
        std::cout << h_selected[i] << " ";
       
    }

    // Cleanup
    cufftDestroy(fftPlan);



}

void gpu_fft_w_sampling_double(double *h_A, int m, int n1,int* d_S,int c) {
    const int numRows = m;       // Number of rows (N_VEC)
    const int numCols = n1;      // Number of columns
    const int halfRows = numRows / 2 + 1;  // Output size for R2C transform
          int rows_per_col = numRows / 2 + 1;
    TimerGPU t;

    // Host and device vectors
    thrust::host_vector<double> h_input(h_A, h_A + numRows * numCols);
    thrust::device_vector<double> d_doubleInput = h_input;
    thrust::device_vector<cufftDoubleComplex> d_complexOutput(numCols * halfRows);

    // Host vector to store real parts of selected frequencies
    thrust::host_vector<double> h_selected;



    cufftHandle fftPlan;
    int n[1] = {numRows}; // FFT size (rows per column)
    cufftResult result = cufftPlanMany(
        &fftPlan,
        1,           // Rank of the transform (1D)
        n,           // Dimensions of the transform
        n, 1, numRows,  // Input layout (inembed, istride, idist)
        n, 1, halfRows,  // Output layout (onembed, ostride, odist)
        CUFFT_D2Z,   // Complex-to-complex transform
        numCols      // Batch size (number of columns)
    );

    assert(result == CUFFT_SUCCESS);

    // Execute FFT
    t.start();
    cufftExecD2Z(
        fftPlan, 
        thrust::raw_pointer_cast(d_doubleInput.data()), 
        thrust::raw_pointer_cast(d_complexOutput.data())
    );
    t.stop();
    std::cout << "Real to Complex FFT:\n";
    t.show_elapsed_time();



    t.start();
    for (int col = 0; col < numCols; ++col) {
        for (int i = 0; i < c; ++i) {
            int row = d_S[i]; 
       
            if (row <= numRows / 2) {
                // Positive frequency
                cufftDoubleComplex val = d_complexOutput[row + col * rows_per_col];
                h_selected.push_back(val.x);  // Take real part
            
            } else {
                // Negative frequency
                int neg_index = numRows - row;  // Map to positive equivalent
                cufftDoubleComplex val = d_complexOutput[neg_index + col * rows_per_col];
                h_selected.push_back(val.x);  // Take real part (conjugate symmetry)
              
            }
        }
    }
    t.stop();
    std::cout<<"sampling time\n";
    t.show_elapsed_time();
    // Print the selected real parts
    std::cout << "Selected real parts (from full spectrum):\n";
    for (size_t i = 0; i < h_selected.size(); ++i) {
        std::cout << h_selected[i] << " ";
       
    }

    // Cleanup
    cufftDestroy(fftPlan);
}



void gpu_diag_multiply(float*h_A, int m, int n, float* h_D, float* h_DA){

    // Allocate device memory using Thrust
    thrust::device_vector<float> d_A(h_A, h_A + m * n); // m x n original matrix
    thrust::device_vector<float> d_D(h_D, h_D + m); // diagonal vector
    thrust::device_vector<float> d_C(m * n);            // m x n result matrix

    // Create cuBLAS handle
    cublasHandle_t handle;
    cublasCreate(&handle);
    TimerGPU t;
    // Perform matrix multiplication: d_C = alpha * d_A * d_B + beta * d_C
    t.start();
    cublasSdgmm(handle, 
    CUBLAS_SIDE_LEFT,
    m,//rows
    n, //columns
    thrust::raw_pointer_cast(d_A.data()), m,//lda=m
    thrust::raw_pointer_cast(d_D.data()), 1, //diagonal and its stride. 
    thrust::raw_pointer_cast(d_C.data()), m);//result matrix
      
    t.stop();

std::cout<<"Single Precision Diagonal Matrix multiplication \n";
t.show_elapsed_time();

    // Copy result from device to host
    thrust::copy(d_C.begin(), d_C.end(), h_DA);

    // Cleanup
    cublasDestroy(handle);

}
struct Selector {
    const cufftComplex* d_complexOutput;
    int rows_per_col;
    int numRows;
    int numCols;

    __host__ __device__
    Selector(const cufftComplex* d_complexOutput, int rows_per_col, int numRows, int numCols)
        : d_complexOutput(d_complexOutput), rows_per_col(rows_per_col), numRows(numRows), numCols(numCols) {}

    __device__
    float operator()(thrust::tuple<int, int> index) const {
        int row = thrust::get<0>(index);
        int col = thrust::get<1>(index);

       /* if (row <= numRows / 2) {
            // Positive frequency
            cufftComplex val = d_complexOutput[row + col * rows_per_col];
            return val.x;  // Real part
        } else {
            // Negative frequency
            int neg_index = numRows - row;  // Map to positive equivalent
            cufftComplex val = d_complexOutput[neg_index + col * rows_per_col];
            return val.x;  // Real part (conjugate symmetry)
        }*/
              // Use ternary operator for symmetry logic (reduced branching)
      
        // Load the real part directly
        int actual_row = (row <= numRows / 2) ? row : (numRows - row);
        //printf("Row: %d, Col: %d, ActualRow: %d,Index grabbed: %d\n", row, col, actual_row,actual_row + col * rows_per_col);
 
        return d_complexOutput[actual_row + col * rows_per_col].x;
    }
};


struct SpecifiedRowIndexer {
    
    int numRows;  // Total number of rows
    int numCols;  // Total number of columns
    int* rowIndices;  // Pointer to specified rows

    SpecifiedRowIndexer(int _numRows, int _numCols, int* _rowIndices)
        : numRows(_numRows), numCols(_numCols), rowIndices(_rowIndices) {}

    __host__ __device__
    thrust::tuple<int, int> operator()(int index) const {
     int col = index / numRows;
        int rowIdx = index % numRows;

          
        return thrust::make_tuple(rowIndices[rowIdx], col);
    }
};

struct fftSelector{
    const cufftComplex* d_complexOutput;
    
 int numRows;  // Total number of rows in output
    int numCols;  // Total number of columns
    int* rowIndices;  // Pointer to specified rows
    int LDA;


    fftSelector(const cufftComplex* d_complexOutput, int _numRows, int _numCols, int* _rowIndices, int _LDA)
        : d_complexOutput(d_complexOutput), numRows(_numRows), numCols(_numCols), rowIndices(_rowIndices), LDA(_LDA) {}

    __host__ __device__
    float operator()(int index)const {
        int col = (index / numRows);
        int row = rowIndices[index % numRows];
         int rows_per_col = LDA / 2 + 1;
        int actual_row = (row <= LDA / 2) ? row : (LDA - row);

            //printf("Index: %d, Row: %d, Col: %d, ActualRow: %d, Index grabbed: %d\n", 
           //    index, row,col, actual_row,actual_row + col*rows_per_col );
        // Load the real part directly
        return d_complexOutput[actual_row + col*rows_per_col].x;
    }
};

struct DoublefftSelector{
    const cufftDoubleComplex* d_complexOutput;
    
 int numRows;  // Total number of rows in output
    int numCols;  // Total number of columns
    int* rowIndices;  // Pointer to specified rows
    int LDA;


    DoublefftSelector(const cufftDoubleComplex* d_complexOutput, int _numRows, int _numCols, int* _rowIndices, int _LDA)
        : d_complexOutput(d_complexOutput), numRows(_numRows), numCols(_numCols), rowIndices(_rowIndices), LDA(_LDA) {}

    __host__ __device__
    float operator()(int index)const {
        int col = (index / numRows);
        int row = rowIndices[index % numRows];
         int rows_per_col = LDA / 2 + 1;
        int actual_row = (row <= LDA / 2) ? row : (LDA - row);

            //printf("Index: %d, Row: %d, Col: %d, ActualRow: %d, Index grabbed: %d\n", 
           //    index, row,col, actual_row,actual_row + col*rows_per_col );
        // Load the real part directly
        return d_complexOutput[actual_row + col*rows_per_col].x;
    }
};

void Select_Matrix_Rows(thrust::device_vector<int>& d_S, cufftComplex* d_complexOutput, int m, int n, int LD ,thrust::device_vector<float>& d_selected) {

  // m: # rows in output
  // LD: input matrix
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter1 = thrust::make_transform_iterator(zero, fftSelector(d_complexOutput ,d_S.size(), n,thrust::raw_pointer_cast(d_S.data()),LD)); 
  //auto elem1 = thrust::make_permutation_iterator(d_complexOutput, iter1);

  thrust::device_vector<float>B(d_S.size()*n);
  thrust::copy_n(iter1, d_S.size()*n, d_selected.data());
  //  std::cout<<"New Copy Method" <<std::endl;
  //for(int i =0; i<d_S.size()*n;++i){
  //  std::cout<<B[i]<<std::endl;
//}
}


void Double_Select_Matrix_Rows(thrust::device_vector<int>& d_S, cufftDoubleComplex* d_complexOutput, int m, int n, int LD ,thrust::device_vector<double>& d_selected) {

  // m: # rows in output
  // LD: input matrix
  auto zero = thrust::make_counting_iterator<int>(0);
  auto iter1 = thrust::make_transform_iterator(zero, DoublefftSelector(d_complexOutput ,d_S.size(), n,thrust::raw_pointer_cast(d_S.data()),LD)); 
  //auto elem1 = thrust::make_permutation_iterator(d_complexOutput, iter1);

  thrust::device_vector<float>B(d_S.size()*n);
  thrust::copy_n(iter1, d_S.size()*n, d_selected.data());
  //  std::cout<<"New Copy Method" <<std::endl;
  //for(int i =0; i<d_S.size()*n;++i){
  //  std::cout<<B[i]<<std::endl;
//}
}




void thrust_selection( thrust::device_vector<int>& d_S, 
                      const cufftComplex* d_complexOutput,
                      thrust::device_vector<float>& d_selected,
                      int numRows, int numCols, int rows_per_col) {

    // Calculate the number of (row, col) pairs
    int total_elements = d_S.size() * numCols;

    // Output vector for (row, col) pairs
    thrust::device_vector<thrust::tuple<int, int>> d_indices(total_elements);

    // Generate (row, col) pairs
    thrust::transform(
        thrust::counting_iterator<int>(0),                              // Start at 0
        thrust::counting_iterator<int>(total_elements),                 // End at total_elements
        d_indices.begin(),                                              // Output
        SpecifiedRowIndexer(d_S.size(), numCols,            // Functor
                            thrust::raw_pointer_cast(d_S.data()))
    );

 /*   // Print the result
    std::cout << "Specified rows in column-major order:\n";
    for (int i = 0; i < total_elements2; ++i) {
        auto pair = d_indices2[i];
        int row = thrust::get<0>(static_cast<thrust::tuple<int, int>>(pair));
    int col = thrust::get<1>(static_cast<thrust::tuple<int, int>>(pair));
        std::cout << "Row: " << row << ", Col: " << col << std::endl;
    }*/

    // Resize the output vector
    d_selected.resize(total_elements);

    // Apply the selector in parallel
    thrust::transform(  d_indices.begin(), d_indices.end(),
                      d_selected.begin(),
                      Selector(d_complexOutput, rows_per_col, numRows, numCols));
}

struct scalar_multiply {
    const double scalar;
    scalar_multiply(float s) : scalar(s) {}
    __host__ __device__
    double operator()(const double& x) const {
        return x * scalar;
    }
};
void mprpCholeskyQR(double* h_A,int m, int n, float c, double* h_Q,double* h_R,double& time,double* operationtimes ){
    //declarations
    const int numRows = m;       // Number of rows (N_VEC)
    const int numCols = n;      // Number of columns
    const int halfRows = numRows / 2 + 1;  // Output size for R2C transform
    int rows_per_col = numRows / 2 + 1;
    int n1[1] = {numRows}; // FFT size (rows per column)
    const int lda = m;
    int lwork = 0, info = 0;
    int *devInfo = nullptr;
    float *d_work = nullptr, *d_tau = nullptr;
    double*d_workchol=nullptr;
    int *d_info = nullptr; 
    double alpha =1.0;double beta=0.0;

    double normalizer= sqrt(c/m);
    TimerGPU t; 
    TimerGPU t1;

    //apparently adviasable to do this first, before memory allocation. lets try
       cufftHandle fftPlan;
      cufftResult result = cufftPlanMany(
        &fftPlan,
        1,           // Rank of the transform (1D)
        n1,           // Dimensions of the transform
        n1, 1, numRows,  // Input layout (inembed, istride, idist)
        n1, 1, halfRows,  // Output layout (onembed, ostride, odist)
        CUFFT_R2C,   // Real-to-complex transform
        numCols      // Batch size (number of columns)
    );

       cublasHandle_t cublashandle;
   // cublasCreate(&cublashandle);
    cublasStatus_t cublasStat = cublasCreate(&cublashandle);
 
        // Create cuSolver handle
    cusolverDnHandle_t cusolverhandle;
    cusolverStatus_t cusolverStat = cusolverDnCreate(&cusolverhandle);
    //assert(cusolverStat == CUSOLVER_STATUS_SUCCESS) ;

    //Allocation of Memory
    //t.start();
    thrust::device_vector<float> d_As(m*n); // m x n original matrix
    thrust::device_vector<double>d_A(h_A, h_A + m * n);
    thrust::device_vector<double>d_AT(m*n); //for A^T 

      thrust::transform(d_A.begin(), d_A.end(),
                      d_As.begin(), [] __device__ (double val) {
                          return static_cast<float>(val); 
                      }); //creates the float matrix
    thrust::device_vector<float> d_D(m); //random diagonal
    thrust::device_vector<int> d_S(c); //random samples
    thrust::device_vector<float> d_FA(m * n);     //transformed matrix to be sampled from
    thrust::device_vector<cufftComplex> d_FA_complex(m*n);
    thrust::device_vector<cufftComplex> d_FA_real(m*n);
    thrust::device_vector<cufftComplex> d_complexOutput(numCols * halfRows);//output from fft
    thrust::device_vector<cufftComplex> d_complexOutput2(numCols * numRows);//output from  C2C fft
    thrust::host_vector<cufftComplex> h_F_A_complex(m*n);


    thrust::device_vector<float> d_selected(c*n); // Host vector to store real parts of selected frequencies
    
    thrust::device_vector<double> d_R_s(c*n);      //does not need to be square  
    thrust::device_vector<double> d_G(n*n);      //gram matrix 
  

    thrust::device_vector<double> d_RsTR(n*n, 0.0); //will be upper triangular versions
    thrust::device_vector<double> d_R2TR(n*n, 0.0); 
    thrust::counting_iterator<unsigned int> index(0);

    thrust::device_vector<double> d_Q(m*n);      //device Q matrix 
    thrust::device_vector<double> d_R(m*n);      //device R matrix 

          
     thrust::device_vector<int> d_linear_indices(d_S.size() *  numCols);
        thrust::device_vector<float> d_real_output(numRows * numCols);
        thrust::device_vector<float> d_real_sampled_output(d_S.size() * numCols);


    std::vector<float> tau(n, 0); //constants used for householder qr 
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int))); // Allocate device memory for error information (devInfo) and tau
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(float)));


     std::vector<float>h_D(m);
      std::vector<int> h_S(c); //random samples, debugging
   // t.stop();
   // std::cout<<"Allocation of Memory\n";
   // t.show_elapsed_time();


    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    // Create cuBLAS, cufft, and cusolver handle
      if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization error" << std::endl;
        std::cerr<<"Code "<<cusolverStat;
        return;
    }
                 //assert(result == CUFFT_SUCCESS);
        if (result != CUFFT_SUCCESS) {
        std::cerr << "Error during FFT Plan" << std::endl;
        std::cerr << "Code " << result<< std::endl;
        goto cleanup;
    }
    //plan making and workspace querying. I think this is fair to have as a fixed overhead.


      // Query workspace size for geqrf
    cusolverStat = cusolverDnSgeqrf_bufferSize(cusolverhandle, c, n, thrust::raw_pointer_cast(d_selected.data()), c, &lwork);
    assert (cusolverStat == CUSOLVER_STATUS_SUCCESS);
      

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(float)));

    // Query workspace size for Cholesky factorization
    cusolverStat = cusolverDnDpotrf_bufferSize(cusolverhandle, CUBLAS_FILL_MODE_UPPER, n, thrust::raw_pointer_cast(d_G.data()), n, &lwork);
    assert (cusolverStat == CUSOLVER_STATUS_SUCCESS);

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_workchol, lwork * sizeof(double)));

    //establish fft plan
       
  
    t1.start();

     t.start();
    init_random_gpu_rad(thrust::raw_pointer_cast(d_D.data()),m); //initializes the random diagonal matrix
 //  t.stop();
   //std::cout<<"Generating D\n";
   // t.show_elapsed_time(); 
 

    init_random_gpu_samp(thrust::raw_pointer_cast(d_S.data()),c); //initializes the sampling matrix
    t.stop();
    operationtimes[3]=t.elapsed_time();
   // std::cout<<"Generating S and D\n";
   // t.show_elapsed_time();


  t.start();
   
    // Perform matrix multiplication: d_C = alpha * d_A * d_B + beta * d_C
   //t.start();   
    cublasSdgmm(cublashandle, 
    CUBLAS_SIDE_LEFT,
    m,//rows
    n, //columns
    thrust::raw_pointer_cast(d_As.data()), m,//lda=m
    thrust::raw_pointer_cast(d_D.data()), 1, //diagonal and its stride. 
    thrust::raw_pointer_cast(d_FA.data()), m);//result matrix
  t.stop();
  operationtimes[6]=t.elapsed_time();  

  //  std::cout<<"Single Precision Diagonal Matrix multiplication \n";
   // t.show_elapsed_time();

   
  t.start();  
    cufftExecR2C(fftPlan, 
               thrust::raw_pointer_cast(d_FA.data()),
                thrust::raw_pointer_cast(d_complexOutput.data()));
  t.stop();
    operationtimes[4]=t.elapsed_time();
    t.start();
Select_Matrix_Rows(d_S, 
thrust::raw_pointer_cast(d_complexOutput.data()),
 m, 
 n, 
 m,d_selected);
thrust::transform(d_selected.begin(), d_selected.end(), d_selected.begin(), scalar_multiply(normalizer));
t.stop(); 
operationtimes[7]=t.elapsed_time();
//
 
//std::cout<<"New Selection Time\n";
//t.show_elapsed_time();
   




    // Perform scalar multiplication on the GPU in parallel

    //We now perform the QR of the sampled matrix to find the preconditioner R_s
  
    //need to ask about the actualtime start. Should I count the query time? or is this "fixed"? 
 
    t.start();
    // Compute QR factorization
    cusolverStat = cusolverDnSgeqrf(cusolverhandle, c, n, thrust::raw_pointer_cast(d_selected.data()), c, d_tau, d_work, lwork, devInfo);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);

   t.stop();
    operationtimes[5]=t.elapsed_time();
    //std::cout << "Single Precision QR\n";
   // t.show_elapsed_time();
  //  t.start();
  thrust::transform(d_selected.begin(), d_selected.end(),
                      d_R_s.begin(), [] __device__ (float val) {
                          return static_cast<double>(val); 
                      });
  //  t.stop();
   //std::cout<<"Conversion of R\n";


 //   t.show_elapsed_time();
    //make R square to reduce triangular solve time
    thrust::for_each(index,index+(c*n),upper_triangular_copy(thrust::raw_pointer_cast(d_R_s.data()),thrust::raw_pointer_cast(d_RsTR.data()),c,n));



 
  //transposes to make the correct rhs
   cublasStat=cublasDgeam(cublashandle,
                          CUBLAS_OP_T , CUBLAS_OP_N ,
                          n, m,
                          &alpha,
                          thrust::raw_pointer_cast(d_A.data()), m,
                          &beta,
                          thrust::raw_pointer_cast(d_A.data()), m,
                        thrust::raw_pointer_cast(d_AT.data()),  n);

    t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                           uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_RsTR.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
    t.stop();
operationtimes[2]=t.elapsed_time();

//std::cout<<"Preconditioning triangular solve\n";    
//t.show_elapsed_time();

    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 

    // Perform matrix multiplication: G = A^T * A

t.start();
    cublasDgemm(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N,  //currently d_AT is preconditioned matrix in storage
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G.data()), n);

    t.stop();
    operationtimes[0]=t.elapsed_time();
  // std::cout<<"Transposed Gram Matrix Formation\n";
 //  t.show_elapsed_time();


    t.start();
    cusolverStat = cusolverDnDpotrf(cusolverhandle, CUBLAS_FILL_MODE_UPPER, n, thrust::raw_pointer_cast(d_G.data()), n, d_workchol, lwork, d_info);
    t.stop();
    operationtimes[1]=t.elapsed_time();
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS); 
   // std::cout<<"Cholesky\n";
    //t.show_elapsed_time();



t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_UPPER,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
t.stop();
operationtimes[2]+=t.elapsed_time();
//std::cout<<"Triangular solve to recover Q\n";
//t.show_elapsed_time();


    //there might be a better way to do this, but it needs to happen for both so it is apples to apples
  //  t.start();
    thrust::for_each(index,index+(n*n),upper_triangular_copy(thrust::raw_pointer_cast(d_G.data()),thrust::raw_pointer_cast(d_R2TR.data()),n,n));
  //  thrust::for_each(index,index+(c*n),upper_triangular_copy(thrust::raw_pointer_cast(d_R_s.data()),thrust::raw_pointer_cast(d_RsTR.data()),c,n));
   // t.stop();
   // std::cout<<"R and G clean up to make upper triangular\n";
   // t.show_elapsed_time();


  //multiply the newly minted upper triangular parts
t.start();
 cublasDgemm(cublashandle, CUBLAS_OP_N, CUBLAS_OP_N,  
                n, n, n, &alpha, 
                thrust::raw_pointer_cast(d_R2TR.data()), n, 
                thrust::raw_pointer_cast(d_RsTR.data()), n, 
                &beta, 
                thrust::raw_pointer_cast(d_R.data()), n);
   
    t.stop();
operationtimes[0]=operationtimes[0]+t.elapsed_time();
//std::cout<<"Forming R\n";
//t.show_elapsed_time();
  t1.stop();
  time =t1.elapsed_time();
  
    //turns the Q back to right transpose
    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 

    
    //copy things back to the host: 
    thrust::copy(d_R.begin(), d_R.end(), h_R);
    
    thrust::copy(d_Q.begin(),d_Q.end(),h_Q);
   
cleanup:
// Cleanup device memory allocations
    if (devInfo != nullptr) {
        cudaFree(devInfo);
        devInfo = nullptr;
    }
    if (d_tau != nullptr) {
        cudaFree(d_tau);
        d_tau = nullptr;
    }
    if (d_work != nullptr) {
        cudaFree(d_work);
        d_work = nullptr;
    }
    if (d_workchol != nullptr) {
        cudaFree(d_workchol);
        d_workchol = nullptr;
    }

    // Destroy cuBLAS handle
    if (cublashandle) {
        cublasDestroy(cublashandle);
        cublashandle = nullptr;
    }

    // Destroy cuSolver handle
    if (cusolverhandle) {
        cusolverDnDestroy(cusolverhandle);
        cusolverhandle = nullptr;
    }

    // Destroy cuFFT plan
    if (fftPlan) {
        cufftDestroy(fftPlan);
        fftPlan = 0;
    }

    // Cleanup Thrust device vectors
    d_As.clear();
    d_As.shrink_to_fit();

    d_A.clear();
    d_A.shrink_to_fit();

    d_AT.clear();
    d_AT.shrink_to_fit();

    d_D.clear();
    d_D.shrink_to_fit();

    d_S.clear();
    d_S.shrink_to_fit();

    d_FA.clear();
    d_FA.shrink_to_fit();

    d_complexOutput.clear();
    d_complexOutput.shrink_to_fit();

    d_selected.clear();
    d_selected.shrink_to_fit();

    d_R_s.clear();
    d_R_s.shrink_to_fit();

    d_G.clear();
    d_G.shrink_to_fit();

    d_RsTR.clear();
    d_RsTR.shrink_to_fit();

    d_R2TR.clear();
    d_R2TR.shrink_to_fit();

    d_Q.clear();
    d_Q.shrink_to_fit();

    d_R.clear();
    d_R.shrink_to_fit();

    // Cleanup host vectors
    tau.clear();
    tau.shrink_to_fit();

    h_D.clear();
    h_D.shrink_to_fit();

    h_S.clear();
    h_S.shrink_to_fit();

}
template <typename T>
void printArray(const std::string& name, const thrust::device_vector<T>& d_array, int rows, int cols) {
    std::cout << name << ": " << std::endl;
    thrust::host_vector<T> h_array = d_array; // Copy to host
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            std::cout << h_array[i * cols + j] << " ";
        }
        std::cout << std::endl;
    }
}
void rand_cholQR(double* h_A,int m, int n,double* h_Q,double* h_R, double& time, double* operationtimes ){

  //int m = 5;   // Number of rows in sparse matrix
  //  int n = 5;   // Number of columns in sparse matrix
    int nnz = n; // Number of non-zero elements in the sparse matrix
    int A_col = n;   // Number of columns in the dense matrix
    int A_row =m;
    
    int S_row =8.24*((n*n)+n);//need to make this right
    int S_col=n;

    int S2_row=74.3*log(S_row); //need to adjust this
    std::cout<<S2_row<<std::endl;
    int S2_col= S_row;

    // Define the CSC representation of the sparse matrix
    
               // Row indices

    // Dense matrix to multiply with
  

    thrust::device_vector<double>d_A(h_A, h_A + S_col * A_row);
    thrust::device_vector<double>d_AT(A_col * A_row);
    thrust::device_vector<double>d_Q0(A_col * A_row);
    thrust::device_vector<double>d_Q(A_col * A_row);
    thrust::device_vector<double>d_G(A_col * A_col);
    thrust::device_vector<double>d_R(A_col * A_col);
    thrust::device_vector<double>d_R2TR(A_col * A_col);
    thrust::device_vector<double>d_R1TR(A_col * A_col);

    // Allocate device memory for CSC matrix
    thrust::device_vector<int> d_colOffsets(S_col + 1); thrust::sequence(d_colOffsets.begin(), d_colOffsets.end(), 0); //one thing in each column
    thrust::device_vector<int> d_rows(nnz);  init_random_gpu_samp(thrust::raw_pointer_cast(d_rows.data()),S_row); //row order
    thrust::device_vector<double> d_values(nnz);  double_init_random_gpu_rad(thrust::raw_pointer_cast(d_values.data()),nnz); //values of +/-1

    //Gaussian sketch
    thrust::device_vector<double> d_S2(S2_row*S_row);  double_gaussian_init_random_gpu_rad(thrust::raw_pointer_cast(d_S2.data()),S2_row*S_row); //Gaussian
    thrust::device_vector<double> d_test(S2_row*A_col,1.0); 
    thrust::device_vector<double> d_ones(S2_row*A_col,1.0); 

    // Output matrix dimensions
    thrust::device_vector<double> d_output(S_row * A_col, 0.0);
    thrust::device_vector<double> d_sketched(S2_row * A_col, 0.0);
    


    // cuSPARSE handle and descriptors
    cusparseHandle_t cusparseHandle;
    cusparseSpMatDescr_t matSparse;
    cusparseDnMatDescr_t matA, matOutput;

    CHECK_CUSPARSE(cusparseCreate(&cusparseHandle));

       // cublasCreate(&cublashandle);
    cublasHandle_t cublashandle;
    cublasStatus_t cublasStat = cublasCreate(&cublashandle);

    // Create sparse matrix descriptor
    CHECK_CUSPARSE(cusparseCreateCsc(&matSparse, S_row, S_col, nnz,
                                     thrust::raw_pointer_cast(d_colOffsets.data()),
                                     thrust::raw_pointer_cast(d_rows.data()),
                                     thrust::raw_pointer_cast(d_values.data()),
                                     CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F));

    // Create dense matrix descriptors
    CHECK_CUSPARSE(cusparseCreateDnMat(&matA, S_col, A_col, A_col,
                                       thrust::raw_pointer_cast(d_A.data()),
                                       CUDA_R_64F, CUSPARSE_ORDER_ROW));
    CHECK_CUSPARSE(cusparseCreateDnMat(&matOutput, S_row, A_col, A_col,
                                       thrust::raw_pointer_cast(d_output.data()),
                                       CUDA_R_64F, CUSPARSE_ORDER_ROW));

    // Define scalars
    double alpha = 1.0;
    double beta = 0.0;

    // Buffer size and allocation
    size_t bufferSize = 0;
    void* dBuffer = nullptr;
    CHECK_CUSPARSE(cusparseSpMM_bufferSize(cusparseHandle,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           CUSPARSE_OPERATION_NON_TRANSPOSE,
                                           &alpha, matSparse, matA, &beta, matOutput,
                                           CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize));
    CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));

    // Perform SpMM
    CHECK_CUSPARSE(cusparseSpMM(cusparseHandle,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                CUSPARSE_OPERATION_NON_TRANSPOSE,
                                &alpha, matSparse, matA, &beta, matOutput,
                                CUDA_R_64F, CUSPARSE_SPMM_ALG_DEFAULT, dBuffer));
  
    //do dense Gaussian sketch

     cublasDgemm(cublashandle, CUBLAS_OP_N, CUBLAS_OP_N,  
                S2_row, A_col, S_row, &alpha, 
                thrust::raw_pointer_cast(d_S2.data()), S2_row, 
                thrust::raw_pointer_cast(d_output.data()), S_row, 
                &beta, 
                thrust::raw_pointer_cast(d_sketched.data()), S2_row);
    //printArray("Q0", d_output, S2_row,A_col);
       
        const int lda = m;
    int lwork = 0, info = 0;
    int lwork_orgqr = 0;
    int *devInfo = nullptr;
    double *d_work = nullptr, *d_tau = nullptr;
    double *d_workchol = nullptr;
      int lworkchol = 0;
         int *d_info = nullptr;
            thrust::counting_iterator<unsigned int> index(0);
    TimerGPU t;
     cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    // Only allocate device memory for d_tau
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(double)));

    // Allocate device memory for devInfo (for error handling)
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int)));

    cusolverDnHandle_t cusolverhandle;
    cusolverStatus_t cusolverStat = cusolverDnCreate(&cusolverhandle);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization error" << std::endl;
      //  goto cleanup;
    }

    // Query workspace size for geqrf
    cusolverStat = cusolverDnDgeqrf_bufferSize(cusolverhandle, S2_row, A_col, thrust::raw_pointer_cast(d_sketched.data()), S2_row, &lwork);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size" << std::endl;
    //    goto cleanup;
    }

    // Allocate workspace memory for geqrf
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(double)));

    //t.start();
    // Compute QR factorization
    cusolverStat = cusolverDnDgeqrf(cusolverhandle, S2_row, A_col, thrust::raw_pointer_cast(d_sketched.data()), S2_row, d_tau, d_work, lwork, devInfo);
    //t.stop();
     printArray("Q0", d_sketched, S2_row, A_col);
  thrust::for_each(index,index+(S2_row*A_col),upper_triangular_copy(thrust::raw_pointer_cast(d_sketched.data()),thrust::raw_pointer_cast(d_R1TR.data()),S2_row,A_col));

    printArray("R0", d_R1TR, A_col, A_col);
    

   if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error during QR factorization, info = " << cusolverStat << std::endl;
       // goto cleanup;
    }
    
    
       cublasStat=cublasDgeam(cublashandle,
                          CUBLAS_OP_T , CUBLAS_OP_N ,
                          n, m,
                          &alpha,
                          thrust::raw_pointer_cast(d_A.data()), m,
                          &beta,
                          thrust::raw_pointer_cast(d_A.data()), m,
                        thrust::raw_pointer_cast(d_AT.data()),  n);
    printArray("AT", d_AT, A_col, A_row);
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_UPPER,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           A_col,
                            A_row,
                           &alpha,
                           thrust::raw_pointer_cast(d_R1TR.data()), 
                           A_col,
                           thrust::raw_pointer_cast(d_AT.data()), A_col);

printArray("Q0", d_AT, A_col, A_row);    


  



   
            //printArray("Q0 Transpose", d_sketched, A_col, A_row);
       cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q0.data()), m); 
   
    //We now need to do CholQR again, but on d_Q

    //forms Q^TQ
    t.start();
    cublasDgemm(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N,  
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_Q0.data()), m, 
                thrust::raw_pointer_cast(d_Q0.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G.data()), n);
    t.stop();
        cusolverStat = cusolverDnDpotrf_bufferSize(cusolverhandle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, &lworkchol);
    if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "Error querying workspace size for Cholesky" << std::endl;
       // goto cleanup;
    }

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_workchol, lworkchol * sizeof(double)));
    //does Cholesky
    //    t.start();
    cusolverStat = cusolverDnDpotrf(cusolverhandle, uplo, n, thrust::raw_pointer_cast(d_G.data()), n, d_workchol, lworkchol, d_info);
    t.stop();
    // Copy and print the result
  


        cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);

     
    thrust::for_each(index,index+(S2_row*n),upper_triangular_copy(thrust::raw_pointer_cast(d_sketched.data()),thrust::raw_pointer_cast(d_R1TR.data()),n,n));//d_sketched I think is in row major 

     //printArray("Output Preconditioner", d_R1TR, A_col, A_col);
    thrust::for_each(index,index+(n*n),upper_triangular_copy(thrust::raw_pointer_cast(d_G.data()),thrust::raw_pointer_cast(d_R2TR.data()),n,n));
     //printArray("Output chol R", d_R2TR, A_col, A_col);
   // t.stop();
   // std::cout<<"R and G clean up to make upper triangular\n";
   // t.show_elapsed_time();
   
         //Recover original R 
         t.start();
 cublasDgemm(cublashandle, CUBLAS_OP_N, CUBLAS_OP_N, //( the g1 needs to be transposed because of an earloer transposition.)
                n, n, n, &alpha, 
                thrust::raw_pointer_cast(d_R2TR.data()), n, 
                thrust::raw_pointer_cast(d_R1TR.data()), n, 
                &beta, 
                thrust::raw_pointer_cast(d_R.data()), n);

    t.stop();
    //turns the Q back to right transpose
    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 


    //copy things back to the host: 
    thrust::copy(d_R.begin(), d_R.end(), h_R);
    //thrust::copy(d_AT.begin(), d_AT.end(), h_Q);
    thrust::copy(d_Q.begin(),d_Q.end(),h_Q);
    //thrust::copy(d_G.begin(), d_G.end(), h_R);



    // Cleanup
    CHECK_CUDA(cudaFree(dBuffer));
    CHECK_CUSPARSE(cusparseDestroySpMat(matSparse));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matA));
    CHECK_CUSPARSE(cusparseDestroyDnMat(matOutput));
    CHECK_CUSPARSE(cusparseDestroy(cusparseHandle));

/*
//subspace embedding parameters
double ep1=0.9;
int p1=8.24*((n*n)+n);
double ep2= 0.49;
    int p2= 74.3*log(n);
    int S_num_cols=m;
    int S_num_rows=p1; //this may actually need to be p1, ill change that when I get it working
    std::cout<<"Number of Rows in S "<<p1<<std::endl;
   
    int S_nnz=m; //number of nonzero elements, one in each column.
    int A_num_rows=m;
    int A_num_cols=n;
    int lda=m;
    int   ldc             = S_num_rows;
    int S_size= S_num_cols*S_num_rows;
     int A_size= A_num_cols*A_num_rows;
     int C_size= S_num_rows*A_num_cols;

       float alpha           = 1.0f;
    float beta            = 0.0f;
   
    //device memory management
    thrust::device_vector<int> dS_cscOffsets(S_num_cols+1);
    thrust::device_vector<int>dS_rows(S_nnz);
    thrust::device_vector<float>dS_values(S_nnz);
    thrust::device_vector<double>d_A(h_A,h_A+A_size); //original matrix
    thrust::device_vector<double>dS(S_size); //countsketch
    thrust::device_vector<double>d_C(C_size); //countsketch


//makes the appropriate offsets and column ordering
    thrust::sequence(dS_cscOffsets.begin(), dS_cscOffsets.end(), 0);
    init_random_gpu_samp(thrust::raw_pointer_cast(dS_rows.data()),S_num_rows); //row order
    init_random_gpu_rad(thrust::raw_pointer_cast(dS_values.data()),S_nnz); //values of +/-1


      std::cout<<"Offset Entries"<<std::endl;
    for(int i=0; i<S_num_cols+1;++i){
    std::cout<<dS_cscOffsets[i]<<std::endl;
}
    std::cout<<"Row Entries"<<std::endl;
    for(int i=0; i<S_nnz;++i){
    std::cout<<dS_rows[i]<<std::endl;
}
     // Print the dense matrix (optional, for verification)
    std::cout << "Dense matrix representation:" << std::endl;
    for (int row = 0; row < p1; ++row) {
        for (int col = 0; col < m; ++col) {
            double value = 0.0;
            for (int idx = dS_cscOffsets[col]; idx < dS_cscOffsets[col + 1]; ++idx) {
                if (dS_rows[idx] == row) {
                    value = dS_values[idx];
                    break;
                }
            }
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }


    //for the entries of the sparse matrix: 

    cusparseHandle_t     cusparsehandle = NULL;
    cusparseSpMatDescr_t matS;
    cusparseDnMatDescr_t matA, matC;
    void*                dBuffer    = NULL;
    size_t               bufferSize = 0;
    //CHECK_CUSPARSE( cusparseCreate(&cusparsehandle) );
 
    cusparseStatus_t cusparseStat = cusparseCreate(&cusparsehandle);
    // Create sparse matrix S in CSC format
   CHECK_CUSPARSE( cusparseCreateCsc(&matS, S_num_rows, S_num_cols, S_nnz,
                                    thrust::raw_pointer_cast(dS_cscOffsets.data()), thrust::raw_pointer_cast( dS_rows.data()),  thrust::raw_pointer_cast(dS_values.data()),
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                                     CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F) );

        // Create dense matrix A
    CHECK_CUSPARSE( cusparseCreateDnMat(&matA, S_num_cols, A_num_cols, lda, thrust::raw_pointer_cast( d_A.data()),
                                        CUDA_R_64F, CUSPARSE_ORDER_COL) );
                                              
       
     // Create dense matrix C
    CHECK_CUSPARSE( cusparseCreateDnMat(&matC, S_num_rows, A_num_cols, ldc, thrust::raw_pointer_cast( d_C.data()),
                                        CUDA_R_64F, CUSPARSE_ORDER_COL) );
  

        // allocate an external buffer if needed
    CHECK_CUSPARSE( cusparseSpMM_bufferSize(
                                 cusparsehandle,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matS, matA, &beta, matC, CUDA_R_64F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize) )
    CHECK_CUDA( cudaMalloc(&dBuffer, bufferSize) );




    // execute SpMM
  cusparseStat=cusparseSpMM(cusparsehandle,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                               CUSPARSE_OPERATION_NON_TRANSPOSE,
                                 &alpha, matS, matA, &beta, matC, CUDA_R_64F,
                                 CUSPARSE_SPMM_ALG_DEFAULT, dBuffer);
        assert(cusparseStat == CUSPARSE_STATUS_SUCCESS); 
   


  
    // Example: Accessing matrix element (i, j)
int j = dS_cscOffsets[1]; // col index
int i =dS_rows[1]; // row index
 //std::cout << "Value at (" << i << ", " << j << ") is: ";
// Check if the element (i, j) is non-zero
int start_idx = dS_cscOffsets[j] ; // Start index for col i
int end_idx =dS_cscOffsets[j+1]; // End index for col i

// Find the column index within the row
bool found = false;
for (int idx = start_idx; idx < end_idx; ++idx) {
    if (dS_rows[idx] == i) {
        found = true;
        std::cout << "Value at (" << i << ", " << j << ") is: " << dS_values[idx] << std::endl;
        break;
    }
}


if (!found) {
    std::cout << "Element (" << i << ", " << j << ") is zero." << std::endl;
}

for(int i=0; i<A_num_rows*A_num_cols;++i){
    std::cout<<d_A[i]<<std::endl;
}
   std::cout << "Dense matrix representation:" << std::endl;
    for (int row = 0; row < S_row; ++row) {
        for (int col = 0; col < S_col; ++col) {
            double value = 0.0;
            for (int idx = d_colOffsets[col]; idx < d_colOffsets[col + 1]; ++idx) {
                if (d_rows[idx] == row) {
                    value = d_values[idx];
                    break;
                }
            }
            std::cout << value << " ";
        }
        std::cout << std::endl;
    }
*/
}

void rpCholeskyQR(double* h_A,int m, int n, float c, double* h_Q,double* h_R,double& time,double* operationtimes ){
    //declarations
    const int numRows = m;       // Number of rows (N_VEC)
    const int numCols = n;      // Number of columns
    const int halfRows = numRows / 2 + 1;  // Output size for R2C transform
    int rows_per_col = numRows / 2 + 1;
    int n1[1] = {numRows}; // FFT size (rows per column)
    const int lda = m;
    int lwork = 0, info = 0;
    int *devInfo = nullptr;
    double *d_work = nullptr, *d_tau = nullptr;
    double*d_workchol=nullptr;
    int *d_info = nullptr; 
    double alpha =1.0;double beta=0.0;

    double normalizer= sqrt(c/m);
    TimerGPU t; 
    TimerGPU t1;

    //apparently adviasable to do this first, before memory allocation. lets try
       cufftHandle fftPlan;
      cufftResult result = cufftPlanMany(
        &fftPlan,
        1,           // Rank of the transform (1D)
        n1,           // Dimensions of the transform
        n1, 1, numRows,  // Input layout (inembed, istride, idist)
        n1, 1, halfRows,  // Output layout (onembed, ostride, odist)
        CUFFT_D2Z,   // Real-to-complex transform
        numCols      // Batch size (number of columns)
    );

       cublasHandle_t cublashandle;
   // cublasCreate(&cublashandle);
    cublasStatus_t cublasStat = cublasCreate(&cublashandle);
 
        // Create cuSolver handle
    cusolverDnHandle_t cusolverhandle;
    cusolverStatus_t cusolverStat = cusolverDnCreate(&cusolverhandle);
    //assert(cusolverStat == CUSOLVER_STATUS_SUCCESS) ;

    //Allocation of Memory
    //t.start();
    
    thrust::device_vector<double>d_A(h_A, h_A + m * n);
    thrust::device_vector<double>d_AT(m*n); //for A^T 

    
    thrust::device_vector<double> d_D(m); //random diagonal
    thrust::device_vector<int> d_S(c); //random samples
    thrust::device_vector<double> d_FA(m * n);     //transformed matrix to be sampled from
    thrust::device_vector<cufftDoubleComplex> d_FA_complex(m*n);
    thrust::device_vector<cufftDoubleComplex> d_FA_real(m*n);
    thrust::device_vector<cufftDoubleComplex> d_complexOutput(numCols * halfRows);//output from fft
    thrust::device_vector<cufftDoubleComplex> d_complexOutput2(numCols * numRows);//output from  C2C fft
    thrust::host_vector<cufftDoubleComplex> h_F_A_complex(m*n);


    thrust::device_vector<double> d_selected(c*n); // Host vector to store real parts of selected frequencies
    
    thrust::device_vector<double> d_R_s(c*n);      //does not need to be square  
    thrust::device_vector<double> d_G(n*n);      //gram matrix 
  

    thrust::device_vector<double> d_RsTR(n*n, 0.0); //will be upper triangular versions
    thrust::device_vector<double> d_R2TR(n*n, 0.0); 
    thrust::counting_iterator<unsigned int> index(0);

    thrust::device_vector<double> d_Q(m*n);      //device Q matrix 
    thrust::device_vector<double> d_R(m*n);      //device R matrix 

          
     thrust::device_vector<int> d_linear_indices(d_S.size() *  numCols);
        thrust::device_vector<float> d_real_output(numRows * numCols);
        thrust::device_vector<float> d_real_sampled_output(d_S.size() * numCols);


    std::vector<float> tau(n, 0); //constants used for householder qr 
    CUDA_CHECK(cudaMalloc((void**)&devInfo, sizeof(int))); // Allocate device memory for error information (devInfo) and tau
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(float)));



    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    // Create cuBLAS, cufft, and cusolver handle
      if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {
        std::cerr << "CUSOLVER initialization error" << std::endl;
        std::cerr<<"Code "<<cusolverStat;
        return;
    }
                 //assert(result == CUFFT_SUCCESS);
        if (result != CUFFT_SUCCESS) {
        std::cerr << "Error during FFT Plan" << std::endl;
        std::cerr << "Code " << result<< std::endl;
        goto cleanup;
    }
    //plan making and workspace querying. I think this is fair to have as a fixed overhead.


      // Query workspace size for geqrf
    cusolverStat = cusolverDnDgeqrf_bufferSize(cusolverhandle, c, n, thrust::raw_pointer_cast(d_selected.data()), c, &lwork);
    assert (cusolverStat == CUSOLVER_STATUS_SUCCESS);
      

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_work, lwork * sizeof(float)));

    // Query workspace size for Cholesky factorization
    cusolverStat = cusolverDnDpotrf_bufferSize(cusolverhandle, CUBLAS_FILL_MODE_UPPER, n, thrust::raw_pointer_cast(d_G.data()), n, &lwork);
    assert (cusolverStat == CUSOLVER_STATUS_SUCCESS);

    // Allocate workspace memory
    CUDA_CHECK(cudaMalloc((void**)&d_workchol, lwork * sizeof(double)));

    //establish fft plan
       
  
    t1.start();

     t.start();
    double_init_random_gpu_rad(thrust::raw_pointer_cast(d_D.data()),m); //initializes the random diagonal matrix
 //  t.stop();
   //std::cout<<"Generating D\n";
   // t.show_elapsed_time(); 
 
  //  for(int i=0;i<c*n;++i){
  //  std::cout<<d_D[i]<<std::endl;
//} 
   
    init_random_gpu_samp(thrust::raw_pointer_cast(d_S.data()),c); //initializes the sampling matrix
    t.stop();
    operationtimes[3]=t.elapsed_time();
   // std::cout<<"Generating S and D\n";
   // t.show_elapsed_time();


  t.start();
   
    // Perform matrix multiplication: d_C = alpha * d_A * d_B + beta * d_C
   //t.start();   
    cublasDdgmm(cublashandle, 
    CUBLAS_SIDE_LEFT,
    m,//rows
    n, //columns
    thrust::raw_pointer_cast(d_A.data()), m,//lda=m
    thrust::raw_pointer_cast(d_D.data()), 1, //diagonal and its stride. 
    thrust::raw_pointer_cast(d_FA.data()), m);//result matrix
  t.stop();
  operationtimes[6]=t.elapsed_time();  

  //  std::cout<<"Single Precision Diagonal Matrix multiplication \n";
   // t.show_elapsed_time();

  t.start();  
    cufftExecD2Z(fftPlan, 
               thrust::raw_pointer_cast(d_FA.data()),
                thrust::raw_pointer_cast(d_complexOutput.data()));
  t.stop();
    operationtimes[4]=t.elapsed_time();
    t.start();
Double_Select_Matrix_Rows(d_S, 
thrust::raw_pointer_cast(d_complexOutput.data()),
 m, 
 n, 
 m,d_selected);
thrust::transform(d_selected.begin(), d_selected.end(), d_selected.begin(), scalar_multiply(normalizer));
t.stop(); 
operationtimes[7]=t.elapsed_time();
//


//std::cout<<"New Selection Time\n";
//t.show_elapsed_time();
   




    // Perform scalar multiplication on the GPU in parallel

    //We now perform the QR of the sampled matrix to find the preconditioner R_s
  
    //need to ask about the actualtime start. Should I count the query time? or is this "fixed"? 
 
    t.start();
    // Compute QR factorization
    cusolverStat = cusolverDnDgeqrf(cusolverhandle, c, n, thrust::raw_pointer_cast(d_selected.data()), c, d_tau, d_work, lwork, devInfo);
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS);
        
        
   t.stop();
    operationtimes[5]=t.elapsed_time();
    //std::cout << "Single Precision QR\n";
   // t.show_elapsed_time();
  //  t.start();
  
  //  t.stop();
   //std::cout<<"Conversion of R\n";


 //   t.show_elapsed_time();
    //make R square to reduce triangular solve time
    thrust::for_each(index,index+(c*n),upper_triangular_copy(thrust::raw_pointer_cast(d_selected.data()),thrust::raw_pointer_cast(d_RsTR.data()),c,n));
     



 
  //transposes to make the correct rhs
   cublasStat=cublasDgeam(cublashandle,
                          CUBLAS_OP_T , CUBLAS_OP_N ,
                          n, m,
                          &alpha,
                          thrust::raw_pointer_cast(d_A.data()), m,
                          &beta,
                          thrust::raw_pointer_cast(d_A.data()), m,
                        thrust::raw_pointer_cast(d_AT.data()),  n);

    t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                           uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_RsTR.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
    t.stop();
operationtimes[2]=t.elapsed_time();

//std::cout<<"Preconditioning triangular solve\n";    
//t.show_elapsed_time();

    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 

    // Perform matrix multiplication: G = A^T * A

t.start();
    cublasDgemm(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N,  //currently d_AT is preconditioned matrix in storage
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G.data()), n);

    t.stop();
    operationtimes[0]=t.elapsed_time();
  // std::cout<<"Transposed Gram Matrix Formation\n";
 //  t.show_elapsed_time();


    t.start();
    cusolverStat = cusolverDnDpotrf(cusolverhandle, CUBLAS_FILL_MODE_UPPER, n, thrust::raw_pointer_cast(d_G.data()), n, d_workchol, lwork, d_info);
    t.stop();
    operationtimes[1]=t.elapsed_time();
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS); 
   // std::cout<<"Cholesky\n";
    //t.show_elapsed_time();



t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_UPPER,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
t.stop();
operationtimes[2]+=t.elapsed_time();
//std::cout<<"Triangular solve to recover Q\n";
//t.show_elapsed_time();


    //there might be a better way to do this, but it needs to happen for both so it is apples to apples
  //  t.start();
    thrust::for_each(index,index+(n*n),upper_triangular_copy(thrust::raw_pointer_cast(d_G.data()),thrust::raw_pointer_cast(d_R2TR.data()),n,n));
  //  thrust::for_each(index,index+(c*n),upper_triangular_copy(thrust::raw_pointer_cast(d_R_s.data()),thrust::raw_pointer_cast(d_RsTR.data()),c,n));
   // t.stop();
   // std::cout<<"R and G clean up to make upper triangular\n";
   // t.show_elapsed_time();


  //multiply the newly minted upper triangular parts
t.start();
 cublasDgemm(cublashandle, CUBLAS_OP_N, CUBLAS_OP_N,  
                n, n, n, &alpha, 
                thrust::raw_pointer_cast(d_R2TR.data()), n, 
                thrust::raw_pointer_cast(d_RsTR.data()), n, 
                &beta, 
                thrust::raw_pointer_cast(d_R.data()), n);
   
    t.stop();
operationtimes[0]=operationtimes[0]+t.elapsed_time();
//std::cout<<"Forming R\n";
//t.show_elapsed_time();
  t1.stop();
  time =t1.elapsed_time();
  
    //turns the Q back to right transpose
    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 

    
    //copy things back to the host: 
    thrust::copy(d_R.begin(), d_R.end(), h_R);
    
    thrust::copy(d_Q.begin(),d_Q.end(),h_Q);
   
cleanup:
// Cleanup device memory allocations
    if (devInfo != nullptr) {
        cudaFree(devInfo);
        devInfo = nullptr;
    }
    if (d_tau != nullptr) {
        cudaFree(d_tau);
        d_tau = nullptr;
    }
    if (d_work != nullptr) {
        cudaFree(d_work);
        d_work = nullptr;
    }
    if (d_workchol != nullptr) {
        cudaFree(d_workchol);
        d_workchol = nullptr;
    }

    // Destroy cuBLAS handle
    if (cublashandle) {
        cublasDestroy(cublashandle);
        cublashandle = nullptr;
    }

    // Destroy cuSolver handle
    if (cusolverhandle) {
        cusolverDnDestroy(cusolverhandle);
        cusolverhandle = nullptr;
    }

    // Destroy cuFFT plan
    if (fftPlan) {
        cufftDestroy(fftPlan);
        fftPlan = 0;
    }

    // Cleanup Thrust device vectors
   

    d_A.clear();
    d_A.shrink_to_fit();

    d_AT.clear();
    d_AT.shrink_to_fit();

    d_D.clear();
    d_D.shrink_to_fit();

    d_S.clear();
    d_S.shrink_to_fit();

    d_FA.clear();
    d_FA.shrink_to_fit();

    d_complexOutput.clear();
    d_complexOutput.shrink_to_fit();

    d_selected.clear();
    d_selected.shrink_to_fit();

    d_R_s.clear();
    d_R_s.shrink_to_fit();

    d_G.clear();
    d_G.shrink_to_fit();

    d_RsTR.clear();
    d_RsTR.shrink_to_fit();

    d_R2TR.clear();
    d_R2TR.shrink_to_fit();

    d_Q.clear();
    d_Q.shrink_to_fit();

    d_R.clear();
    d_R.shrink_to_fit();

    // Cleanup host vectors
    tau.clear();
    tau.shrink_to_fit();


}




//In this Section, I will adapt things to be a template so that everything is cleaner and you can be flexible in your choice of precision. 
/*****************FUNCTIONS FOR RNG********************/
template <typename T>
struct flex_prg : public thrust::unary_function<unsigned int, T> {
    T a, b;

    __host__ __device__
    flex_prg(T _a = static_cast<T>(0), T _b = static_cast<T>(1)) : a(_a), b(_b) {}

    __host__ __device__
    T operator()(const unsigned int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_real_distribution<T> dist(a, b);
        rng.discard(n);
        T random_value = dist(rng);
        return random_value < static_cast<T>(0.5) ? static_cast<T>(-1) : static_cast<T>(1);
    }
};

// Templated random generator struct (for random integer values)
template <typename T>
struct flex_prg2 : public thrust::unary_function<unsigned int, T> {
    T a, b;

    __host__ __device__
    flex_prg2(T _a = 0, T _b = 1) : a(_a), b(_b) {}

    __host__ __device__
    T operator()(const unsigned int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_int_distribution<T> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};

// Templated function for random initialization (real numbers)
template <typename T>
void flex_init_random_gpu_rad(T* d_ptr, int N) {
    thrust::device_ptr<T> A(d_ptr);  // Wrap the raw pointer
    thrust::counting_iterator<unsigned int> index(0);
    thrust::transform(index, index + N, A, flex_prg<T>());
}

// Templated function for random initialization (integers)
template <typename T>
void flex_init_random_gpu_samp(T* d_ptr, int N) {
    thrust::device_ptr<T> A(d_ptr);  // Wrap the raw pointer
    thrust::counting_iterator<unsigned int> index(0);
    thrust::transform(index, index + N, A, flex_prg2<T>(0, N-1));
}

/*******************FUNCTIONS FOR FFT AND SKETCH******************/

template <typename T>
struct flex_fftSelector {
    const T* d_complexOutput;
    
    int numRows;  // Total number of rows in output
    int numCols;  // Total number of columns
    int* rowIndices;  // Pointer to specified rows
    int LDA;

    __host__ __device__
    flex_fftSelector(const T* d_complexOutput, int _numRows, int _numCols, int* _rowIndices, int _LDA)
        : d_complexOutput(d_complexOutput), numRows(_numRows), numCols(_numCols), rowIndices(_rowIndices), LDA(_LDA) {}

    __host__ __device__
    auto operator()(int index) const {
        int col = (index / numRows);
        int row = rowIndices[index % numRows];
        int rows_per_col = LDA / 2 + 1;
        int actual_row = (row <= LDA / 2) ? row : (LDA - row);

        if constexpr (std::is_same<T, __half2>::value) {
        return __half22float2(d_complexOutput[actual_row + col * rows_per_col]).x; 
    } 
        else{return d_complexOutput[actual_row + col * rows_per_col].x;}

    }
};

template <typename T>
struct flex_scalar_multiply {
    const T scalar;

    __host__ __device__
    flex_scalar_multiply(T s) : scalar(s) {}

    __host__ __device__
    T operator()(const T& x) const {
        return x * scalar;
    }
};

template <typename T>
struct flex_upper_triangular_copy {
    int m;
    int n;
    T* old_R;  // Source pointer
    T* new_R;  // Destination pointer

    __host__ __device__
    flex_upper_triangular_copy(T* old_R_ptr, T* new_R_ptr, int m, int n)
        : old_R(old_R_ptr), new_R(new_R_ptr), m(m), n(n) {}

    __host__ __device__
    void operator()(int x) const {
        int row = x / n;
        int col = x % n;

        if (row <= col && row < n) {
            new_R[row + n * col] = old_R[row + m * col];
        }
    }
};

template <typename T,typename T1>
void flex_Select_Matrix_Rows(thrust::device_vector<int>& d_S, T* d_complexOutput, int m, int n, int LD, thrust::device_vector<T1>& d_selected) {
    auto zero = thrust::make_counting_iterator<int>(0);
    auto iter1 = thrust::make_transform_iterator(zero, flex_fftSelector<T>(d_complexOutput, d_S.size(), n, thrust::raw_pointer_cast(d_S.data()), LD));

    thrust::copy_n(iter1, d_S.size() * n, d_selected.data());
}

// Dummy half-precision type (for example purposes)


// Precision selector struct (maps int condition to type)
template <int Precision>
struct PrecisionSelector {
    using Type = double;  // Default precision
};

template <>
struct PrecisionSelector<1> {
    using Type = float;   // Single precision
};

template <>
struct PrecisionSelector<2> {
    using Type = __half;    // Half precision
};

template <typename T>
__global__ void diag_matrix_mult_kernel(const T* diag, T* matrix, int rows, int cols) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    
    if (row < rows && col < cols) {
        int index = row + col * rows; // Column-major order
        matrix[index] *= diag[row];   // Scale each row by corresponding diagonal element
    }
}

// Function to launch the kernel
template <typename T>
void diag_matrix_mult(thrust::device_vector<T>& diag, thrust::device_vector<T>& matrix, int rows, int cols) {
    const int BLOCK_SIZE = 16;
    dim3 threadsPerBlock(BLOCK_SIZE, BLOCK_SIZE);
    dim3 numBlocks((rows + BLOCK_SIZE - 1) / BLOCK_SIZE, (cols + BLOCK_SIZE - 1) / BLOCK_SIZE);

    diag_matrix_mult_kernel<<<numBlocks, threadsPerBlock>>>(
        thrust::raw_pointer_cast(diag.data()),
        thrust::raw_pointer_cast(matrix.data()),
        rows, cols
    );
    cudaDeviceSynchronize();
}



void flex_rpCholeskyQR(double* h_A, int m, int n, float c, double* h_Q, double* h_R, double& time, double* operationtimes) {

    const int Precision= 1; //this will need to be specified after a conditional ( based on a condition number estimator) 0, double, 1 single, 2 half. 
    using PrecisionType = typename PrecisionSelector<Precision>::Type; //this 

    // Choose precision dynamically

    //since fft matrices require certain types, this chooses the appropriate one based on the specified precision. 
using fftMatType = typename std::conditional<
    (Precision == 0), 
    cufftDoubleComplex, 
    typename std::conditional<(Precision == 2), __half2, cufftComplex>::type
>::type;

using selected_vector_type = typename std::conditional<
    Precision == 2,              // If Precision is 2 (half precision)
    thrust::device_vector<float>, //  for half precision
    thrust::device_vector<PrecisionType>  // Otherwise
>::type;

/*
    std::cout << "Preconditioning precision: "
              << (std::is_same<PrecisionType, double>::value ? "double" :
                  std::is_same<PrecisionType, float>::value ? "float" : "half")
              << std::endl;*/

    /********DECLARATION OF CONSTANTS************/ 
    //things for FFT
    const long long int numRows = m;       // Number of rows (N_VEC)
    const long long int numCols = n;      // Number of columns
    const int halfRows = numRows / 2 + 1;  // Output size for R2C transform
    int rows_per_col = numRows / 2 + 1;
    long long int n1[1] = {numRows}; // FFT size (rows per column) (long long int to be flexible in the FFT executer, per cuFFT specifications)
    size_t workSize; 
    
    const int lda = m;

      int *d_info = nullptr;    /* error info */									 
      size_t d_lwork = 0;     /* size of workspace */									 
      void *d_work = nullptr; /* device workspace */									 
      size_t h_lwork = 0;     /* size of workspace */									 
      void *h_work = nullptr; /* host workspace */	
        int *d_info2 = nullptr;    /* error info */									 
      int d_lwork2 = 0;     /* size of workspace for Cholesky */									 
      double *d_work2 = nullptr; /* device workspace for Cholesky*/									 
      size_t h_lwork2 = 0;     /* size of workspace */									 
      void *h_work2 = nullptr; /* host workspace */		
        void *d_tau=nullptr; /*tau is used in QR*/

    //double*d_workchol=nullptr; //this one is double since Cholesky always happens in double
 
    double alpha =1.0, beta=0.0;

    PrecisionType normalizer= sqrt(c/m);
    TimerGPU t; 
    TimerGPU t1;



    /*************INITIALIZATION OF CUDA HANDLES and other re=levant setup items****************/
        // Declare CUDA data types outside the if-statement so they are always accessible
    
    cudaDataType_t CUDA_PRECISION;
    cudaDataType_t FFT_OUTPUT_PRECISION;
    cudaDataType_t QR_CUDA_PRECISION; //this needs to be added because geqrf does not support half. I don't think it will matter, but need to check with ilse


    // Set values conditionally
    if (Precision == 1) {
        CUDA_PRECISION=CUDA_R_32F;
        FFT_OUTPUT_PRECISION=CUDA_C_32F;
        QR_CUDA_PRECISION=CUDA_PRECISION;

    } else if (Precision == 2) { //half
        CUDA_PRECISION=CUDA_R_16F;
        FFT_OUTPUT_PRECISION=CUDA_C_16F;
        QR_CUDA_PRECISION=CUDA_R_32F;
      
    } else {  // Double precision
        CUDA_PRECISION=CUDA_R_64F;
        FFT_OUTPUT_PRECISION=CUDA_C_64F;
        QR_CUDA_PRECISION=CUDA_PRECISION;
    }
    //for peak performance, we query work size and make FFT plan before memory allocation. 
    cufftHandle fftPlan;
    cufftResult_t fftresult=cufftCreate(&fftPlan);
    fftresult = cufftXtGetSizeMany(
        fftPlan,
        1,           // Rank of the transform (1D)
        n1,           // Dimensions of the transform
        n1, 1, numRows,  // Input layout (inembed, istride, idist)
        CUDA_PRECISION, 
        n1, 1, halfRows,  // Output layout (onembed, ostride, odist)    
        FFT_OUTPUT_PRECISION,
        numCols,      // Batch size (number of columns)
        &workSize, //querying worksize
        CUDA_PRECISION //execution precision
    );
        if (fftresult != CUFFT_SUCCESS) {std::cerr << "CUFFT setup failed!" << std::endl<<"Code "<<fftresult; return;}

        //MAke FFT Plan
     fftresult =cufftXtMakePlanMany(
        fftPlan,
        1,           // Rank of the transform (1D)
        n1,           // Dimensions of the transform
        n1, 1, numRows,  // Input layout (inembed, istride, idist)
        CUDA_PRECISION, 
        n1, 1, halfRows,  // Output layout (onembed, ostride, odist)    
        FFT_OUTPUT_PRECISION, //Output precision, is different because complex
        numCols,      // Batch size (number of columns)
        &workSize, //querying worksize
        CUDA_PRECISION //execution precision
    );
            if (fftresult != CUFFT_SUCCESS) {std::cerr << "CUFFT plan failed!" << std::endl<<"Code "<<fftresult; return;}
            
    /***************HANDLE DECLARATION****************/
    cublasHandle_t cublashandle;
    cublasStatus_t cublasStat = cublasCreate(&cublashandle);
        if (cublasStat != CUBLAS_STATUS_SUCCESS) {std::cerr << "CUBlAS initialization error" << std::endl<<"Code "<<cublasStat; return;}
        // Create cuSolver handle
    cusolverDnHandle_t cusolverhandle;
    cusolverStatus_t cusolverStat = cusolverDnCreate(&cusolverhandle);
          if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {std::cerr << "CUSOLVER initialization error" << std::endl<<"Code "<<cusolverStat; return;}

      cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER; //specified for triangular solves downstream

    /**********ALLOCATION OF MEMORY******************/
     thrust::device_vector<double>d_A(h_A, h_A + m * n);
    thrust::device_vector<double>d_AT(m*n); //for A^T 

    thrust::device_vector<PrecisionType> d_D(m); //random diagonal
    //thrust::device_vector<PrecisionType> d_tau(n); //tau, for QR
    thrust::device_vector<int> d_S(c); //random samples
    thrust::device_vector<PrecisionType> d_FA(m * n);     //transformed matrix to be sampled from

        //Here, we transform d_FA into the appropriate precision for preconditioning. 
      thrust::transform(d_A.begin(), d_A.end(), d_FA.begin(), [] __device__ (double val) {return static_cast<PrecisionType>(val); });
          for(int i =0;i<m*n;++i){
   // std::cout<<d_A[i]<< " " <<__half2float(d_FA[i])<<std::endl;
}
    thrust::device_vector<fftMatType>d_complexOutput(numCols * halfRows);//output from fft
   thrust::device_vector<PrecisionType> d_selected(c*n); // Host vector to store real parts of selected frequencies

    selected_vector_type d_selectedQR(c*n); 

    //everything below is double since it is not included in preconditioner

    thrust::device_vector<double> d_R_s(c*n);      //does not need to be square , preconditioner 
    thrust::device_vector<double> d_G(n*n);      //gram matrix 
  

    thrust::device_vector<double> d_RsTR(n*n, 0.0); //will be upper triangular versions
    thrust::device_vector<double> d_R2TR(n*n, 0.0); 

    thrust::counting_iterator<unsigned int> index(0); //for thrust transformations

    thrust::device_vector<double> d_Q(m*n);      //device Q matrix 
    thrust::device_vector<double> d_R(m*n);      //device R matrix 
    
    //for QR
    CUDA_CHECK(cudaMalloc((void**)&d_tau, n * sizeof(float)));

    /******QUERY WORKSPACES AND ALLOCATED MEMORY FOR CUBLAS AND CUSOLVER OPERATIONS*************/

    

          // Query workspace size for geqrf
    cusolverStat = cusolverDnXgeqrf_bufferSize(cusolverhandle, NULL,  c, n,
    QR_CUDA_PRECISION, thrust::raw_pointer_cast(d_selectedQR.data()), c,
    QR_CUDA_PRECISION,&d_tau,QR_CUDA_PRECISION, &d_lwork, &h_lwork);
     if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {std::cerr << "QR initialization failed!" << std::endl<<"Code "<<cusolverStat; return;}
        t.stop();assert (cusolverStat == CUSOLVER_STATUS_SUCCESS);
    
    // Allocate workspace memory for QR
    CUDA_CHECK(cudaMalloc((void**)&d_work, d_lwork * sizeof(PrecisionType)));
    

    // Query workspace size for Cholesky factorization
    cusolverStat = cusolverDnDpotrf_bufferSize(cusolverhandle, CUBLAS_FILL_MODE_UPPER, n, thrust::raw_pointer_cast(d_G.data()), n, &d_lwork2);
    assert (cusolverStat == CUSOLVER_STATUS_SUCCESS);



    // Allocate workspace memory

    CUDA_CHECK(cudaMalloc((void**)&d_work2, d_lwork2 * sizeof(double)));
   
/**************BEGINNING OPERATIONS**************
Note on timing: 
t1 clocks the whole algorithm
t will clock various operations for debugging purposes and stores it in the operationtimes array
0:matrix Multiplication, 1:Cholesky,2:Triangular Solves, 3: RNG, 4: FFT, 5: QR, 6:selection+scaling, 7: diag mult
************************************************/

    t1.start();

    t.start();
    flex_init_random_gpu_rad(thrust::raw_pointer_cast(d_D.data()),m); //initializes the random diagonal matrix

    

    flex_init_random_gpu_samp(thrust::raw_pointer_cast(d_S.data()),c); //initializes the sampling matrix
    t.stop();
    operationtimes[3]=t.elapsed_time();


    //Multiply by D
    t.start();
    diag_matrix_mult(d_D, d_FA, m, n); //diagonal multiply kernel
    t.stop();

    operationtimes[6]=t.elapsed_time();  
  


    //Perform FFT
    t.start();
    fftresult=cufftXtExec(fftPlan, static_cast<void*>(thrust::raw_pointer_cast(d_FA.data())),
                    static_cast<void*>(thrust::raw_pointer_cast(d_complexOutput.data())),CUFFT_FORWARD);
        if (fftresult != CUFFT_SUCCESS) {std::cerr << "CUFFT execution failed!" << std::endl<<"Code "<<fftresult; return;}
    t.stop();
        operationtimes[4]=t.elapsed_time();
  
    //Cleverly select the real parts from the FFT matrix
    t.start();
    flex_Select_Matrix_Rows(d_S, 
    thrust::raw_pointer_cast(d_complexOutput.data()),
    m, 
    n, 
    m,d_selected);      


    //multiply by a normalizing factor
        thrust::transform(d_selected.begin(), d_selected.end(), d_selected.begin(), flex_scalar_multiply<PrecisionType>(normalizer));
        t.stop();
        operationtimes[7]=t.elapsed_time();

        /**TEMPORARY CONDITIONAL ON HALF PRECISION****/

        if(Precision==2){
                  thrust::transform(d_selected.begin(), d_selected.end(),
                      d_selectedQR.begin(), [] __device__ (PrecisionType val) {
                          return static_cast<float>(val); 
                      });
        }else{
           // thrust::transform(d_selected.begin(), d_selected.end(),
             //    d_selectedQR.begin(), [] __device__ (PrecisionType val) {
               //      return static_cast<PrecisionType>(val); 
                // });

		thrust::copy(d_selected.begin(),d_selected.end(),d_selectedQR.begin());	
        }



        //Perform QR
        t.start();
        cusolverStat=cusolverDnXgeqrf(cusolverhandle, NULL, c, n, QR_CUDA_PRECISION, static_cast<void*>(thrust::raw_pointer_cast(d_selectedQR.data())),
                                        c, QR_CUDA_PRECISION, d_tau,
                                    QR_CUDA_PRECISION, d_work, d_lwork, h_work,
                                        h_lwork, d_info);	
                if (cusolverStat != CUSOLVER_STATUS_SUCCESS) {std::cerr << "QR execution failed!" << std::endl<<"Code "<<cusolverStat; return;}
        t.stop();
        operationtimes[5]=t.elapsed_time();

  


    //convert the whole matrix to double
         thrust::transform(d_selectedQR.begin(), d_selectedQR.end(),
                      d_R_s.begin(), [] __device__ (PrecisionType val) {
                          return static_cast<double>(val); 
                      });

    //make R square to reduce triangular solve time
    thrust::for_each(index,index+(c*n),flex_upper_triangular_copy<double>(thrust::raw_pointer_cast(d_R_s.data()),thrust::raw_pointer_cast(d_RsTR.data()),(int)c,n));


  //transposes to make the correct rhs
   cublasStat=cublasDgeam(cublashandle,
                          CUBLAS_OP_T , CUBLAS_OP_N ,
                          n, m,
                          &alpha,
                          thrust::raw_pointer_cast(d_A.data()), m,
                          &beta,
                          thrust::raw_pointer_cast(d_A.data()), m,
                        thrust::raw_pointer_cast(d_AT.data()),  n);
    //Preconditioning Solve
    t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                           uplo,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_RsTR.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
    t.stop();
operationtimes[2]=t.elapsed_time();

    //transposing back for forming Gram matrix
    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 


// Perform matrix multiplication: G = A^T * A

    t.start();
    cublasDgemm(cublashandle, CUBLAS_OP_T, CUBLAS_OP_N,  //currently d_AT is preconditioned matrix in storage
                n, n, m, &alpha, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                thrust::raw_pointer_cast(d_Q.data()), m, 
                &beta, 
                thrust::raw_pointer_cast(d_G.data()), n);

    t.stop();
    operationtimes[0]=t.elapsed_time();

    //CHOLESKY
    t.start();
    cusolverStat = cusolverDnDpotrf(cusolverhandle, CUBLAS_FILL_MODE_UPPER, n, thrust::raw_pointer_cast(d_G.data()), n, d_work2, d_lwork2, d_info);
    t.stop();
    operationtimes[1]=t.elapsed_time();
    assert(cusolverStat == CUSOLVER_STATUS_SUCCESS); 


    //Solving for Q
    t.start();
    cublasStat=cublasDtrsm(cublashandle,
                           CUBLAS_SIDE_LEFT,
                            CUBLAS_FILL_MODE_UPPER,
                           CUBLAS_OP_T, 
                           CUBLAS_DIAG_NON_UNIT,
                           n,
                            m,
                           &alpha,
                           thrust::raw_pointer_cast(d_G.data()), 
                           n,
                           thrust::raw_pointer_cast(d_AT.data()), n);
    t.stop();
    operationtimes[2]+=t.elapsed_time();



    thrust::for_each(index,index+(n*n),flex_upper_triangular_copy<double>(thrust::raw_pointer_cast(d_G.data()),thrust::raw_pointer_cast(d_R2TR.data()),n,n));


  //multiply the newly minted upper triangular parts
    t.start();
    cublasDgemm(cublashandle, CUBLAS_OP_N, CUBLAS_OP_N,  
                n, n, n, &alpha, 
                thrust::raw_pointer_cast(d_R2TR.data()), n, 
                thrust::raw_pointer_cast(d_RsTR.data()), n, 
                &beta, 
                thrust::raw_pointer_cast(d_R.data()), n);
   
    t.stop();
operationtimes[0]=operationtimes[0]+t.elapsed_time();

  t1.stop();
  time =t1.elapsed_time();
  
    //turns the Q back to right transpose
    cublasStat = cublasDgeam(cublashandle,
                         CUBLAS_OP_T, CUBLAS_OP_N,  // Transpose operation
                         m, n,                     // Correct dimensions for transpose
                         &alpha,                   // Alpha scalar
                         thrust::raw_pointer_cast(d_AT.data()), n,  // Input matrix Q^T (ldA = n)
                         &beta,                    // Beta scalar
                         thrust::raw_pointer_cast(d_AT.data()),                  // No second matrix
                         m, 
                         thrust::raw_pointer_cast(d_Q.data()), m); 

    
    //copy things back to the host: 
    thrust::copy(d_R.begin(), d_R.end(), h_R);
    
    thrust::copy(d_Q.begin(),d_Q.end(),h_Q);
    
    cleanup:

// Cleanup device memory allocations
    if (d_work) {
    cudaFree(d_work);
}


if (d_work2) {
    cudaFree(d_work2);
}

if (d_info) {
    cudaFree(d_info);
}
if (d_info2) {
    cudaFree(d_info2);
}

if (h_work) {
    free(h_work);
}
if (h_work2) {
    free(h_work2);
}

    // Destroy cuBLAS handle
    if (cublashandle) {
        cublasDestroy(cublashandle);
        cublashandle = nullptr;
    }

    // Destroy cuSolver handle
    if (cusolverhandle) {
        cusolverDnDestroy(cusolverhandle);
        cusolverhandle = nullptr;
    }

    // Destroy cuFFT plan
    if (fftPlan) {
        cufftDestroy(fftPlan);
        fftPlan = 0;
    }

    // Cleanup Thrust device vectors
   

    d_A.clear();
    d_A.shrink_to_fit();

    d_AT.clear();
    d_AT.shrink_to_fit();

    d_D.clear();
    d_D.shrink_to_fit();

    d_S.clear();
    d_S.shrink_to_fit();

    d_FA.clear();
    d_FA.shrink_to_fit();

    d_complexOutput.clear();
    d_complexOutput.shrink_to_fit();

    d_selected.clear();
    d_selected.shrink_to_fit();

    d_R_s.clear();
    d_R_s.shrink_to_fit();

    d_G.clear();
    d_G.shrink_to_fit();
/*
    d_RsTR.clear();
    d_RsTR.shrink_to_fit();

    d_R2TR.clear();
    d_R2TR.shrink_to_fit();

    d_Q.clear();
    d_Q.shrink_to_fit();

    d_R.clear();
    d_R.shrink_to_fit();


    cudaDeviceSynchronize();

// Reset CUDA device (optional, but ensures resources are freed)
cudaDeviceReset();

*/
}
