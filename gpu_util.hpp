#ifndef GPU_UTIL_HPP
#define GPU_UTIL_HPP

#define CUDA_CHECK(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        assert(false);                                                         \
    }                                                                          \
}

void print_float_matrix(int m, int n, float *A, int lda) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[i + j * lda] << " ";
        }
        std::cout << std::endl;
    }
}

void print_double_matrix(int m, int n, double *A, int lda) {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << A[i + j * lda] << " ";
        }
        std::cout << std::endl;
    }
}
#endif