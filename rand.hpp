
#ifndef RNAD_GPU_HPP
#define RNAD_GPU_HPP


void init_random_gpu_rad(float* d_ptr, int N);
void double_init_random_gpu_rad(double* d_ptr, int N);
void init_random_gpu_samp(int* d_ptr, int N);
void double_gaussian_init_random_gpu_rad(double* d_ptr, int N);


#endif
