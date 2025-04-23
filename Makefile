EXE=driver
CXX=g++

LIB=libgemm_new.so

# Set CUDA path


# Location of the CUDA Toolkit

CUDA_PATH       := /usr/local/apps/cuda/cuda-12.0
#CUDA_PATH       := /usr/local/apps/cuda/cuda-8.0/bin
NVCC            := nvcc


# Compiler flags
EIGEN_ROOT=/share/matrix/jegarri3/VSC/Archive/eigen-3.4.0/
CXXFLAGS= -Wall -O3 -I$(EIGEN_ROOT)  -std=c++14 
NVCCFLAGS=  -arch=native -std=c++14 -I./util/ -Xcompiler -fPIC --extended-lambda

#to run on different architectures ( may also depend on cuda version. The HPC website says the default is CUDA 12.0), you need different compiler flags: -arch=sm_80 -code=sm_80 for a100, sm_86 for a_10,arch=sm_90 for h100 (1/7/25)

# Linker flags (using CUDA_PATH)
LDFLAGS= -L$(CUDA_HOME)/lib64 -lcublas -lcudart -lcusolver -lcufft -lcusparse


# Object files
KERNEL=gemm.o timer_gpu.o rand.o
OBJS=driver.o $(KERNEL)

# Target for the executable
$(EXE): $(OBJS) $(LIB)
	$(CXX) $(OBJS) -o $(EXE) $(LDFLAGS)

# Compile driver.cpp with g++
driver.o: driver.cpp
	$(CXX) -c driver.cpp $(CXXFLAGS) -o driver.o

# Compile the shared library from the kernel
$(LIB): $(KERNEL)
	$(CXX) -shared $(KERNEL) -o $(LIB)

# Compile gemm.cu with nvcc, including -fPIC
gemm.o: gemm.cu
	$(NVCC) -c gemm.cu $(NVCCFLAGS) -o gemm.o

timer_gpu.o: timer_gpu.cu
	$(NVCC) -c timer_gpu.cu $(NVCCFLAGS) -o timer_gpu.o

rand.o: rand.cu
	$(NVCC) -c rand.cu $(NVCCFLAGS) -o rand.o

# Run the program
run: $(EXE)
	./$(EXE)

# Clean up
clean:
	rm -rf *.o *.so $(EXE)
