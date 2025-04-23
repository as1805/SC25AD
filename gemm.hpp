// gemm.hpp
#ifndef GEMM_HPP
#define GEMM_HPP

//Computes and copies two a matrix matrix mulitply A*B (no transposes) in Single
void gpu_Smatrix_multiply(const float *h_A, const float *h_B, float *h_C, int m, int n, int k);

//Computes and copies two a matrix matrix mulitply A*B (no transposes) in Double
void gpu_Dmatrix_multiply(const double *h_A, const double *h_B, double *h_C, int m, int n, int k);

//computes the gram matrix A^T A in Double Precision
void gpu_gram_matrix(const double *h_A, double *h_G, int m, int n);
//computes and copies the QR factorization of a tall and skinny matrix. Returns the modified H_A( upper triangular part is R) as well as the Q formed. in Single
void gpu_QR(float *h_A, float *h_tau, float *h_R,float *h_Q,int m, int n );
//computes QR of a tall and skinny double matrix
void gpu_QRD(double *h_A, double *h_tau, double *h_R, double *h_Q, int m, int n,double&time);

//takes the pointers form gpu_QR and gives a nice output of all the appropriate data using Eigen in Single
void Nice_QR_output(float *h_A, float *h_tau, float *h_R,float *h_Q,int m, int n);

//calculates the cholesky factorization, and returns a pointer to A, where the upper triangular part of A is the triangular factor. This fucntion also copies it back to the CPU. 
void gpu_Chol(double *h_A, double *h_R, int n );

//takes the output of gpu_chol and displays it nicely
void Nice_Chol_output(double *h_A,double *h_R, int n);

//capitalizes on things from gpu_chol to copute cholesky of gram matrix and perform linear solve using original
void gpu_Chol_solve(double *h_G, double *h_R, int n, double *h_A, int m, double *h_QT );

//does CholQR and allocates memory
void gpu_CholQR(double *h_A, double *h_R, int n, double *h_Q, int m,double& time ); 

//does CholQR2 and allocates memory
void gpu_CholQR2(double *h_A, double *h_R, int n, double *h_Q, int m, double& time,double* operationtimes );

//perform fft in single, needs original A to be full rank
void gpu_fft_w_sampling(float*h_A, int m,int n1,int* d_S, int c);

void gpu_fft_w_sampling_double(double*h_A, int m,int n1,int* d_S, int c);

void gpu_diag_multiply(float*h_A, int m, int n,float* h_D, float* h_DA);

void mprpCholeskyQR(double* h_A,int m, int n1, float c,double* h_Q,double* h_R,double& time, double* operationtimes );

void rpCholeskyQR(double* h_A,int m, int n1, float c,double* h_Q,double* h_R,double& time, double* operationtimes );

void rand_cholQR(double* h_A,int m, int n,double* h_Q,double* h_R, double& time, double* operationtimes );

void flex_rpCholeskyQR(double* h_A,int m, int n1, float c,double* h_Q,double* h_R,double& time, double* operationtimes );
#endif
