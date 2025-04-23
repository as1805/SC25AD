\


// driver.cpp
#include <iostream>
#include <vector>
#include "gemm.hpp"
#include "helper.hpp"
#include <Eigen/Dense>
#include "timer_gpu.hpp"
#include "rand.hpp"

using namespace Eigen;

using namespace std;



int main() {
    //int m =620000, n=3, trials = 10;
   
    TimerGPU t;

    double time = 0.0, condPow=9;
    int m =1048576, n = 100, trials=10 ;
    std::vector<double> times(trials);
  
    std::vector<double> operationtimes(8); //need to specify exactly how many categories need to go here. 1:gemm;2:chol;3:triangular solve
     std::vector<std::vector<double>> collectedoperationtimes(8, std::vector<double>(trials, 0.0));
    // Storage for average times: First row for ChoeskyQR2, second row for mprpCholesky
    std::vector<std::vector<double>> averageTimes(3, std::vector<double>(50, 0.0)); //10 is the number of experiments for each column
    std::vector<std::vector<double>> QR2averageOperationTimes(3, std::vector<double>(50, 0.0)); //10 is the number of experiments for each column
    std::vector<std::vector<double>> MRCQRaverageOperationTimes(8, std::vector<double>(50, 0.0)); //10 is the number of experiments for each column
    std::vector<std::vector<double>> RpaverageOperationTimes(8, std::vector<double>(50, 0.0)); //10 is the number of experiments for each column

    std::vector<double> h_A = generateRandomMatrix(m, n, -5.0, 5.0);
    //std::vector<double> h_A = generateRandomMatrix(m, n, -5.0, 5.0);
      //std::vector<double> h_A = {1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16};
           std::vector<double> h_Q(m *n);
    std::vector<double> h_R(m * n); 
    //std::vector<double> h_tau(n);   

//section for generating a random matrix of a certain condition number 


       // std::vector<double> h_A(m*n);

//std::cout << "Matrix Sigma:\n" << S << "\n\n";




//Then we will generate a diagonal matrix of a certain cond number and multiply them all. 
  
  
 //rand_cholQR(h_A.data(),m, n, h_Q.data(),  h_R.data(), time, operationtimes.data());
 //mprpCholeskyQR(h_A.data(), m, n,  n, h_Q.data(), h_R.data(), time,operationtimes.data());

 //rpCholeskyQR(h_A.data(), m, n,   n, h_Q.data(), h_R.data(), time,operationtimes.data());
 /*   
std::cout<< "Number of rows :"<< m <<std::endl;
std::cout<< "Precision: Double"<<std::endl;
 std::vector<double> h_A(m*n);
      std::vector<double> h_U1(m * n), h_U(m * n),h_Q(m*n);
    std::vector<double> h_V1(n * n), h_V(n * n);
    std::vector<double> h_Sigma(n * n, 0.0);
    std::vector<double> h_tau(n), h_R(m * n),h_RV(n*n);
       // Generate random matrices
    h_U1 = generateRandomMatrix(m, n, 0.0, 1.0);
    h_V1 = generateRandomMatrix(n, n, 0.0, 1.0);


    // Orthogonalization (QR factorization)
    gpu_QRD(h_U1.data(), h_tau.data(), h_R.data(), h_U.data(), m, n, time);
    gpu_QRD(h_V1.data(), h_tau.data(), h_RV.data(), h_V.data(), n, n, time);
  
    // Construct Sigma matrix with singular values
   
    double maxSVpow = condPow / 2.0;
    double minSVpow = -1.0 * condPow / 2.0;

        for (int i = 0; i < n; i++) {
        double currentpow = maxSVpow - (maxSVpow - minSVpow) / (n - 1) * i;
        h_Sigma[i * n + i] = pow(10.0, currentpow);
    }



    // Compute h_A = h_U * h_Sigma
    std::vector<double> h_temp(m * n);
  // matrixMultiply(h_U, h_Sigma, h_temp, m, n, n);

 gpu_Dmatrix_multiply(h_U.data(), h_Sigma.data(), h_temp.data(), m, n, n);    

   std::vector<double> h_VT(n * n);
    
    // Transpose h_V to get h_VT
    for (int i = 0; i < n; i++)
     for (int j = 0; j < n; j++)
            h_VT[j * n + i] = h_V[i * n + j];

    //matrixMultiply(h_temp, h_VT, h_A, m, n, n);
    gpu_Dmatrix_multiply(h_temp.data(), h_VT.data(), h_A.data(), m, n, n); 
    MatrixXd A(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          A(i,j)=h_A[i+j*m]; // Column-major order
        }
    }

JacobiSVD<MatrixXd> svd(A);
double cond = svd.singularValues()(0)/svd.singularValues()(svd.singularValues().size()-1);

    std::cout<<"Condition Number of A: "<< cond<<std::endl;
    std::cout<< "Frobenius Norm of A"<<A.norm()<<std::endl;

flex_rpCholeskyQR(h_A.data(), m, n,   3*n, h_Q.data(), h_R.data(), time, operationtimes.data());
       

MatrixXd Q(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          Q(i,j)=h_Q[i+j*m]; // Column-major order
        }
    }
//std::cout<<"Computed Q:\n"<<Q;
MatrixXd R(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          R(i,j)=h_R[i+j*n]; // Column-major order
        }
    }
//std::cout<<"Computed R:\n"<<R;

auto I = Eigen::MatrixXd::Identity(n,n);
MatrixXd QR= Q*R;
//std::cout<<"Reconstructed A\n"<<QR<<endl;

MatrixXd DEV= (Q.transpose()*Q)-I;
MatrixXd RESID= A-QR;


double relresid = RESID.norm()/A.norm();
std::cout<<"The Frobenius MRCQR Deviation from orthonormality is "<<DEV.norm()<<endl;
cout<<"The Frobenius MRCQR Relative Residual is "<<relresid<<endl;

*/


  
    for (int j = 1; j <= 10; j++) {
      
      int n = 240+ 10* j;
      std::vector<double> h_A(m*n);
      std::vector<double> h_U1(m * n), h_U(m * n),h_Q(m*n);
    std::vector<double> h_V1(n * n), h_V(n * n);
    std::vector<double> h_Sigma(n * n, 0.0);
    std::vector<double> h_tau(n), h_R(m * n),h_RV(n*n);
   
    // Generate random matrices
    h_U1 = generateRandomMatrix(m, n, 0.0, 1.0);
    h_V1 = generateRandomMatrix(n, n, 0.0, 1.0);


    // Orthogonalization (QR factorization)
    gpu_QRD(h_U1.data(), h_tau.data(), h_R.data(), h_U.data(), m, n, time);
    gpu_QRD(h_V1.data(), h_tau.data(), h_RV.data(), h_V.data(), n, n, time);
  
    // Construct Sigma matrix with singular values
   
    double maxSVpow = condPow / 2.0;
    double minSVpow = -1.0 * condPow / 2.0;
    for (int i = 0; i < n; i++) {
        double currentpow = maxSVpow - (maxSVpow - minSVpow) / (n - 1) * i;
        h_Sigma[i * n + i] = pow(10.0, currentpow);
    }



    // Compute h_A = h_U * h_Sigma
    std::vector<double> h_temp(m * n);
  // matrixMultiply(h_U, h_Sigma, h_temp, m, n, n);

 gpu_Dmatrix_multiply(h_U.data(), h_Sigma.data(), h_temp.data(), m, n, n);    

    // Compute h_A = (h_U * h_Sigma) * h_V^T
    //std::vector<double> h_A(m * n);
   std::vector<double> h_VT(n * n);
    
    // Transpose h_V to get h_VT
    for (int i = 0; i < n; i++)
     for (int j = 0; j < n; j++)
            h_VT[j * n + i] = h_V[i * n + j];

    //matrixMultiply(h_temp, h_VT, h_A, m, n, n);
    gpu_Dmatrix_multiply(h_temp.data(), h_VT.data(), h_A.data(), m, n, n); 
       MatrixXd A(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          A(i,j)=h_A[i+j*m]; // Column-major order
        }
    }

//JacobiSVD<MatrixXd> svd(A);
//double cond = svd.singularValues()(0)/svd.singularValues()(svd.singularValues().size()-1);

//    std::cout<<"Condition Number of A: "<< cond<<std::endl;


        //std::cout << "We are performing experiments on a " << m << " by " << n << " matrix\n";
         // Measure average time for mprpCholesky
        gpu_CholQR2(h_A.data(), h_R.data(), n, h_Q.data(), m, time, operationtimes.data()); //warmup
        for (int i = 0; i < trials; ++i) {
           gpu_CholQR2(h_A.data(), h_R.data(), n, h_Q.data(), m, time, operationtimes.data());
            times[i] = time;
            collectedoperationtimes[0][i]=operationtimes[0];//assigning gemm time
            collectedoperationtimes[1][i]=operationtimes[1];//assigning chol time
            collectedoperationtimes[2][i]=operationtimes[2];//assigning trsm time
          
        }
       

MatrixXd Q(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          Q(i,j)=h_Q[i+j*m]; // Column-major order
        }
    }
//std::cout<<"Computed Q:\n"<<Q;
MatrixXd R(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          R(i,j)=h_R[i+j*n]; // Column-major order
        }
    }
//std::cout<<"Computed R:\n"<<R;

auto I = Eigen::MatrixXd::Identity(n,n);
MatrixXd QR= Q*R;
//std::cout<<"Reconstructed A\n"<<QR<<endl;

MatrixXd DEV= (Q.transpose()*Q)-I;
MatrixXd RESID= A-QR;

double relresid = RESID.lpNorm<2>()/A.lpNorm<2>();
std::cout<<"The Chol QR2 Deviation from orthonormality is "<<DEV.lpNorm<2>()<<endl;
cout<<"The CholQR2 Relative Residual is "<<relresid<<endl;
        double averageTimeCholQR2 = std::accumulate(times.begin(), times.end(), 0.0) / trials;
    

        // Store average time for CholeskyQR2
        averageTimes[0][j - 1] = averageTimeCholQR2;
        QR2averageOperationTimes[0][j-1]=  std::accumulate(collectedoperationtimes[0].begin(), collectedoperationtimes[0].end(), 0.0) / trials;
        QR2averageOperationTimes[1][j-1]=  std::accumulate(collectedoperationtimes[1].begin(), collectedoperationtimes[1].end(), 0.0) / trials;
        QR2averageOperationTimes[2][j-1]=  std::accumulate(collectedoperationtimes[2].begin(), collectedoperationtimes[2].end(), 0.0) / trials;
/*
        for (int i = 0; i < trials; ++i) {
            mprpCholeskyQR(h_A.data(), m, n, 3.0 * n, h_Q.data(), h_R.data(), time,operationtimes.data());
            times[i] = time;
            collectedoperationtimes[0][i]=operationtimes[0];//assigning gemm time
            collectedoperationtimes[1][i]=operationtimes[1];//assigning chol time
            collectedoperationtimes[2][i]=operationtimes[2];//assigning solve time
            collectedoperationtimes[3][i]=operationtimes[3];//assigning RNG time
            collectedoperationtimes[4][i]=operationtimes[4];//assigning sketch time
            collectedoperationtimes[5][i]=operationtimes[5];//assigning little QR time
            collectedoperationtimes[6][i]=operationtimes[6]; //assigning diagonal multiplication
            collectedoperationtimes[7][i]=operationtimes[7]; //assigning selection
        }
        double averageTimeMprpChol = std::accumulate(times.begin(), times.end(), 0.0) / trials;
        MRCQRaverageOperationTimes[0][j-1]=  std::accumulate(collectedoperationtimes[0].begin(), collectedoperationtimes[0].end(), 0.0) / trials;
        MRCQRaverageOperationTimes[1][j-1]=  std::accumulate(collectedoperationtimes[1].begin(), collectedoperationtimes[1].end(), 0.0) / trials;
        MRCQRaverageOperationTimes[2][j-1]=  std::accumulate(collectedoperationtimes[2].begin(), collectedoperationtimes[2].end(), 0.0) / trials;
        MRCQRaverageOperationTimes[3][j-1]=  std::accumulate(collectedoperationtimes[3].begin(), collectedoperationtimes[3].end(), 0.0) / trials;
        MRCQRaverageOperationTimes[4][j-1]=  std::accumulate(collectedoperationtimes[4].begin(), collectedoperationtimes[4].end(), 0.0) / trials;
        MRCQRaverageOperationTimes[5][j-1]=  std::accumulate(collectedoperationtimes[5].begin(), collectedoperationtimes[5].end(), 0.0) / trials;
        MRCQRaverageOperationTimes[6][j-1]=  std::accumulate(collectedoperationtimes[6].begin(), collectedoperationtimes[6].end(), 0.0) / trials;
        MRCQRaverageOperationTimes[7][j-1]=  std::accumulate(collectedoperationtimes[7].begin(), collectedoperationtimes[7].end(), 0.0) / trials;

        // Store average time for mprpCholesky
        averageTimes[1][j - 1] = averageTimeMprpChol;
*/
           flex_rpCholeskyQR(h_A.data(), m, n, 3.0 * n, h_Q.data(), h_R.data(), time,operationtimes.data()); //warmup
            for (int i = 0; i < trials; ++i) {
            flex_rpCholeskyQR(h_A.data(), m, n, 3.0 * n, h_Q.data(), h_R.data(), time,operationtimes.data());
            times[i] = time;
            collectedoperationtimes[0][i]=operationtimes[0];//assigning gemm time
            collectedoperationtimes[1][i]=operationtimes[1];//assigning chol time
            collectedoperationtimes[2][i]=operationtimes[2];//assigning solve time
            collectedoperationtimes[3][i]=operationtimes[3];//assigning RNG time
            collectedoperationtimes[4][i]=operationtimes[4];//assigning sketch time
            collectedoperationtimes[5][i]=operationtimes[5];//assigning little QR time
            collectedoperationtimes[6][i]=operationtimes[6]; //assigning diagonal multiplication
            collectedoperationtimes[7][i]=operationtimes[7]; //assigning selection
        }
        double averageTimerpChol = std::accumulate(times.begin(), times.end(), 0.0) / trials;
        RpaverageOperationTimes[0][j-1]=  std::accumulate(collectedoperationtimes[0].begin(), collectedoperationtimes[0].end(), 0.0) / trials;
        RpaverageOperationTimes[1][j-1]=  std::accumulate(collectedoperationtimes[1].begin(), collectedoperationtimes[1].end(), 0.0) / trials;
        RpaverageOperationTimes[2][j-1]=  std::accumulate(collectedoperationtimes[2].begin(), collectedoperationtimes[2].end(), 0.0) / trials;
        RpaverageOperationTimes[3][j-1]=  std::accumulate(collectedoperationtimes[3].begin(), collectedoperationtimes[3].end(), 0.0) / trials;
        RpaverageOperationTimes[4][j-1]=  std::accumulate(collectedoperationtimes[4].begin(), collectedoperationtimes[4].end(), 0.0) / trials;
        RpaverageOperationTimes[5][j-1]=  std::accumulate(collectedoperationtimes[5].begin(), collectedoperationtimes[5].end(), 0.0) / trials;
        RpaverageOperationTimes[6][j-1]=  std::accumulate(collectedoperationtimes[6].begin(), collectedoperationtimes[6].end(), 0.0) / trials;
        RpaverageOperationTimes[7][j-1]=  std::accumulate(collectedoperationtimes[7].begin(), collectedoperationtimes[7].end(), 0.0) / trials;

        // Store average time for mprpCholesky
        averageTimes[2][j - 1] = averageTimerpChol;

 
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          A(i,j)=h_A[i+j*m]; // Column-major order
        }
    }



    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          Q(i,j)=h_Q[i+j*m]; // Column-major order
        }
    }
//std::cout<<"Computed Q:\n"<<Q;

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          R(i,j)=h_R[i+j*n]; // Column-major order
        }
    }
//std::cout<<"Computed R:\n"<<R;


MatrixXd QR1= Q*R;
//std::cout<<"Reconstructed A\n"<<QR<<endl;

MatrixXd DEV1= (Q.transpose()*Q)-I;
MatrixXd RESID1= A-QR1;

double relresid1 = RESID1.lpNorm<2>()/A.lpNorm<2>();
std::cout<<"The MRCQR Deviation from orthonormality is "<<DEV1.lpNorm<2>()<<endl;
cout<<"The MRCQR Relative Residual is "<<relresid1<<endl;
cout<<"The number of columns is"<<n;
}   

    

    // std::cout << "CholeskyQR2  Average Operation Time  "  <<endl;
     //      for (int i=1;i<=50;i++){
     // std::cout << "With "<< 10*i<< " columns :\n Matrix Multiplication: "<<QR2averageOperationTimes[0][i-1]<< " Cholesky: "<< QR2averageOperationTimes[1][i-1] << " Triangular Solve: "<< QR2averageOperationTimes[2][i-1]<<endl; 
   // }

   //   std::cout << "MRCQR Average Operation Time "  <<endl;
   //        for (int i=1;i<=10;i++){
   //   std::cout << "With "<< 10 *i<< " columns :\n Matrix Multiplication: "<<MRCQRaverageOperationTimes[0][i-1]<< ", Cholesky: "<< MRCQRaverageOperationTimes[1][i-1]<< ", Triangular Solve: "<< MRCQRaverageOperationTimes[2][i-1]<< ", RNG: "<< MRCQRaverageOperationTimes[3][i-1]<< ", Small QR: "<< MRCQRaverageOperationTimes[5][i-1]<<", FFT: "<< MRCQRaverageOperationTimes[4][i-1]<<", Diagonal Multiplication: "<< MRCQRaverageOperationTimes[6][i-1]<<", Selection and Scaling: "<< MRCQRaverageOperationTimes[7][i-1]<< endl; 
  //  }

      //    std::cout << "rpCholesky Average Operation Time "  <<endl;
      //     for (int i=1;i<=50;i++){
      // std::cout << "With "<< 10 *i<< " columns :\n Matrix Multiplication: "<<RpaverageOperationTimes[0][i-1]<< ", Cholesky: "<< RpaverageOperationTimes[1][i-1]<< ", Triangular Solve: "<< RpaverageOperationTimes[2][i-1]<< ", RNG: "<< RpaverageOperationTimes[3][i-1]<< ", Small QR: "<< RpaverageOperationTimes[5][i-1]<<", FFT: "<< RpaverageOperationTimes[4][i-1]<<", Diagonal Multiplication: "<< RpaverageOperationTimes[6][i-1]<<", Selection and Scaling: "<< RpaverageOperationTimes[7][i-1]<< endl; 
  //  }
    // Print all stored average times
   std::cout << "\nSummary of Average Times:\n";
    std::cout << "CholeskyQR2:\n ";

    for (int j = 1; j <= 50; j++) {
      std::cout << averageTimes[0][j - 1]<< endl; }

 // std::cout << "mprpCholesky:\n ";
  // for (int j=1;j<=10;j++){
   //  std::cout << averageTimes[1][j - 1] <<endl;
   // }

  std::cout << "rpCholesky:\n ";
   for (int j=1;j<=50;j++){
     std::cout << averageTimes[2][j - 1] <<endl;
    }

//In this section, we do norm computations.

  return 0;
}


  //Initialize matrices on the host
 //std::vector<double> h_A = generateRandomMatrix(m, n, -5.0, 5.0);
  //std::vector<double> h_A = {1,2,3,4,5,6,7,8,9,10};
  //std::vector<float> h_Af = {1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,0.0};
//std::vector<float> h_D= {2,1,1,1,1}; //creates the diagonal matrix (as a float)
  //std::vector<float> h_DA(m);//test for diagonal
/*
//In this section, we do norm computations.
MatrixXd A(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          A(i,j)=h_A[i+j*m]; // Column-major order
        }
    }


MatrixXd Q(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          Q(i,j)=h_Q[i+j*m]; // Column-major order
        }
    }
//std::cout<<"Computed Q:\n"<<Q;
MatrixXd R(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          R(i,j)=h_R[i+j*n]; // Column-major order
        }
    }
//std::cout<<"Computed R:\n"<<R;

auto I = Eigen::MatrixXd::Identity(n,n);
MatrixXd QR= Q*R;
//std::cout<<"Reconstructed A\n"<<QR<<endl;

MatrixXd DEV= (Q.transpose()*Q)-I;
MatrixXd RESID= A-QR;

double relresid = RESID.lpNorm<2>()/A.lpNorm<2>();
std::cout<<"The Deviation from orthonormality is "<<DEV.lpNorm<2>()<<endl;
cout<<"The Relative Residual is "<<relresid<<endl;*/










//init_random_gpu_rad(h_DA.data(), m);
//init_random_gpu_samp(sampled_rows.data(), m);

//gpu_diag_multiply(h_Af.data(), m, n, h_D.data(), h_DA.data());


//this is the right way to generate the nice matrix
  /*MatrixXd A(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          A(i,j)=h_DA[i+j*m]; // Column-major order
        }
    
    }
   
std::cout << "Matrix DA:\n" << A << "\n\n";*/

//gpu_fft_w_sampling(h_Af.data(),m,n,sampled_rows.data(),3);  
//gpu_fft_w_sampling_double(h_A.data(),m,n,sampled_rows.data(),3);  
/*MatrixXd D(m, m);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < m; ++j) {
            D(i, j) = h_D[j * m + i]; 
    }
std::cout << "Matrix D:\n" << D << "\n\n";


std::vector<double> times(trials); //to hold times taken 
double time=0.0; //to hold the timer
 
  std::vector<double> h_G(n*n);
  std::vector<double> h_R(n*n); // To hold R
  std:: vector<double> h_Q(m*n); // To hold Q

  std:: vector<double> h_tau(n); // To hold tau

  std::vector<float> h_AS(h_A.size());
  std::transform(h_A.begin(), h_A.end(), h_AS.begin(), [](double val) {
  return static_cast<float>(val);
});

  std::vector<float> h_RS(m*n); // To hold R
  std:: vector<float> h_QTS(m*n); // To hold Q



  /*
  //TIME REGULAR QR
  for(int i=0;i<trials;++i){
    gpu_QRD(h_A.data(), h_tau.data(), h_R.data(),h_Q.data(), m, n,time );
    times[i]=time;
  }
  double averagetime= accumulate(times.begin(),times.end(),0.0)/trials;
  cout<<" The average time taken over "<<trials<< " trials of Double Precision QR was "<< averagetime<< "\n";


  for(int i=0;i<trials;++i){
    gpu_CholQR(h_A.data(), h_R.data(), n, h_Q.data(),m, time);
    times[i]=time;
  }
  averagetime= accumulate(times.begin(),times.end(),0.0)/trials;
  cout<<" The average time taken over "<<trials<< " trials of CholeskyQR was "<< averagetime<< "\n";

  for(int i=0;i<trials;++i){
    gpu_CholQR2(h_A.data(), h_R.data(), n, h_Q.data(),m, time);
    times[i]=time;
  }
  averagetime= accumulate(times.begin(),times.end(),0.0)/trials;
  cout<<" The average time taken over "<<trials<< " trials of CholeskyQR2 was "<< averagetime<< "\n";

*/

  /*for(int i=0;i<m*n;++i){
  cout<<h_Q[i]<<"\n";
  }
   for(int i=0;i<n*n;++i){
  cout<<h_R[i]<<"\n";
  }
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            std::cout << h_A[i + j * m] << " "; // Column-major order
        }
        std::cout << "\n";
    }

   for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            std::cout << h_Q[j + i * m] << " "; // Column-major order
        }
        std::cout << "\n";
    }
*/

/*
  MatrixXd Q(m, n);
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
          Q(i,j)=h_Q[i+j*m]; // Column-major order
        }
    
    }
   
std::cout << "Matrix Q:\n" << Q << "\n\n";


  MatrixXd Rtemp(n, n);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            Rtemp(i, j) = h_R[j * n + i]; // Converting from column-major to Eigen's row-major
        }
    }
    MatrixXd R = Rtemp.topRows(n).triangularView<Upper>();  // Only take upper triangular part
    std::cout << "Matrix R:\n" << R << "\n\n";
  MatrixXd QR=Q*R;
    std::cout <<"Matrix A after factorization:\n"<<QR<<"\n";*/

    /*
  double peakFLOPS = (976000000000)*.7; //geforce a10
  double peakFLOPS2= (31000000000000)*.005;
  double QRFlops =2*m*(n^2)-(2/3)*n^3; 
  double GramFlops=2*(m)*n*n;
  double solveflops=n*n*m;
  double cholflops=n^3;

  cout<< "ApproxPeak Gram Time "<< GramFlops/peakFLOPS<<"\n";  
  cout<< "ApproxPeak Chol Time "<< cholflops/peakFLOPS<<"\n";  
  cout<< "ApproxPeak solve Time "<< solveflops/peakFLOPS<<"\n";  
  cout<< "AprroxPeak QR Time "<< QRFlops/peakFLOPS2<<"\n";  */
 /// return 0;
//}
