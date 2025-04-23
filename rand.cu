#include <thrust/random.h>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/host_vector.h>
#include <thrust/functional.h>
#include <thrust/transform.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/copy.h>
#include <curand_kernel.h>
#include <iostream>
#include <cmath>



struct prg : public thrust::unary_function<unsigned int, float> {
    float a, b;
    __host__ __device__
    prg(float _a = 0.f, float _b = 1.f) : a(_a), b(_b) {};
    
    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_int_distribution<float> dist(a, b);
        rng.discard(n);
       int random_value = dist(rng);
        return random_value == 0 ? -1.0f : 1.0f;
    }
};

struct prg2 : public thrust::unary_function<unsigned int, float> {
    int a, b;
    __host__ __device__
    prg2(int _a = 0, int _b = 1) : a(_a), b(_b) {};
    
    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_int_distribution<int> dist(a, b);
        rng.discard(n);
        return dist(rng);
    }
};


struct prg3 : public thrust::unary_function<unsigned int, double> {
    double a, b;
    __host__ __device__
    prg3(double _a = 0, double _b = 1) : a(_a), b(_b) {};
    
    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng(n);
        thrust::uniform_int_distribution<float> dist(a, b);
        rng.discard(n);
       int random_value = dist(rng);
        return random_value == 0 ? -1.0 : 1.0;
    }
};
struct prg4 : public thrust::unary_function<unsigned int, double> {
    double a, b;
    __host__ __device__
    prg4(double _a = 0, double _b = 1) : a(_a), b(_b) {};
    
    __host__ __device__
    float operator()(const unsigned int n) const {
        thrust::default_random_engine rng(n);
        thrust::normal_distribution<float> dist(a, b);
        rng.discard(n);
       int random_value = dist(rng);
        return random_value;
    }
};




void init_random_gpu_rad(float* d_ptr, int N) {
    thrust::device_ptr<float> A(d_ptr); //wraps the pointer so I dontr have to copy
    //thrust::output_iterat0r<float> A(N);
    thrust::counting_iterator<unsigned int> index(0);  //

    thrust::transform(index, index + N, A, prg());

//thrust::copy(A.begin(),A.end(),d_ptr);

    
}

void init_random_gpu_samp(int* d_ptr, int N){
    thrust::device_ptr<int> A(d_ptr);
    
    thrust::counting_iterator<unsigned int> index(0);  
  
    thrust::transform(index, index + N, A, prg2(0,N-1));
    

}
void double_init_random_gpu_rad(double* d_ptr, int N){
    thrust::device_ptr<double> A(d_ptr);
    
    thrust::counting_iterator<unsigned int> index(0);  
  
    thrust::transform(index, index + N, A, prg3());
    

}

void double_gaussian_init_random_gpu_rad(double* d_ptr, int N){
    thrust::device_ptr<double> A(d_ptr);
    
    thrust::counting_iterator<unsigned int> index(0);  
  
    thrust::transform(index, index + N, A, prg4());
    

}
