#include <bits/stdc++.h>

#include <iostream>
#include <string>

#include "cutlass/cutlass.h"
#include "cutlass/epilogue/thread/linear_combination_relu.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/reference/device/gemm.h"
#include "cutlass/util/reference/host/tensor_compare.h"
#include "cutlass/util/reference/host/tensor_copy.h"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/tensor_view_io.h"
#include "helper.h"

// The code section below describes datatype for input, output matrices and
// computation between elements in input matrices.
using ElementAccumulator = float;  // <- data type of accumulator
using ElementComputeEpilogue =
    ElementAccumulator;       // <- data type of epilogue operations
using ElementInputA = float;  // cutlass::half_t;              // <- data type
                              // of elements in input matrix A
using ElementInputB = float;  // cutlass::half_t;              // <- data type
                              // of elements in input matrix B
using ElementOutput = float;  // <- data type of elements in output matrix D

// Note that if the output is column major, the bias has to be per row. i.e.
// every row has different bias. If the output is row major, the bias has to be
// per column, i.e. every column has different bias. Below list some other
// notices:
//
// Note this example only works for ColumnMajor output because
//   1) we only have row major epilogue.
//   2) we swap A and B if the output is column major then we can still use the
//      row major epilogue.
//   3) Mx1 bias vector becomes 1xM after the swapping/transposing.
//   4) we can use the existing OutputIterator to load 1xM bias vector.

using LayoutInputA = cutlass::layout::RowMajor;
using LayoutInputB = cutlass::layout::RowMajor;
using LayoutOutput = cutlass::layout::RowMajor;

// This code section describes whether you want to use tensor cores or regular
// SIMT cores on GPU SM
using MMAOp = cutlass::arch::OpClassTensorOp;

// This code section describes CUDA SM architecture number
using SmArch = cutlass::arch::Sm80;

// This code section describes the tile size a thread block will compute
using ShapeMMAThreadBlock =
    cutlass::gemm::GemmShape<128, 128, 32>;  // <- threadblock tile M = 128, N =
                                             // 128, K = 32
// This code section describes tile size a warp will compute
using ShapeMMAWarp =
    cutlass::gemm::GemmShape<64, 64,
                             32>;  // <- warp tile M = 64, N = 64, K = 32
// This code section describes the size of MMA op
using ShapeMMAOp =
    cutlass::gemm::GemmShape<16, 8, 8>;  // <- MMA Op tile M = 16, N = 8, K = 8

// This code section describes how threadblocks are scheduled on GPU
using SwizzleThreadBlock =
    cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>;  // <- ??

// Define the epilogue operation as LinearCombinationRelu. This is approximately
// equal to
//
//    d_ij = max(0, alpha * sum_k(a_ik * b_kj) + c_ij )
//
using EpilogueOpRelu = cutlass::epilogue::thread::LinearCombinationRelu<
    ElementOutput,  // <- data type of output matrix
    128 / cutlass::sizeof_bits<
              ElementOutput>::value,  // <- this is the number of elements per
                                      // vectorized memory access. For half
                                      // precision, it's 8 elements. This
                                      // becomes the vector width of math
                                      // instructions in epilogue too
    ElementAccumulator,      // <- data type of accumulator
    ElementComputeEpilogue,  // <- data type for alpha in linear combination
                             // function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;  // <- alpha x C +
                                                           // bias

using EpilogueOpGelu = cutlass::epilogue::thread::LinearCombinationGELU<
    ElementOutput,  // <- data type of output matrix
    128 / cutlass::sizeof_bits<
              ElementOutput>::value,  // <- this is the number of elements per
                                      // vectorized memory access. For half
                                      // precision, it's 8 elements. This
                                      // becomes the vector width of math
                                      // instructions in epilogue too
    ElementAccumulator,      // <- data type of accumulator
    ElementComputeEpilogue,  // <- data type for alpha in linear combination
                             // function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;  // <- alpha x C +
                                                           // bias

using EpilogueOpSwish = cutlass::epilogue::thread::LinearCombinationHardSwish<
    ElementOutput,  // <- data type of output matrix
    128 / cutlass::sizeof_bits<
              ElementOutput>::value,  // <- this is the number of elements per
                                      // vectorized memory access. For half
                                      // precision, it's 8 elements. This
                                      // becomes the vector width of math
                                      // instructions in epilogue too
    ElementAccumulator,      // <- data type of accumulator
    ElementComputeEpilogue,  // <- data type for alpha in linear combination
                             // function
    cutlass::epilogue::thread::ScaleType::NoBetaScaling>;  // <- alpha x C +
                                                           // bias

template <typename ElementOutput, int VectorSize, typename ElementAccumulator,
          typename ElementComputeEpilogue,
          //   cutlass::epilogue::thread::ScaleType ScaleType,
          bool UseGELU = false>
using EpilogueOpSelector = std::conditional_t<
    UseGELU,
    cutlass::epilogue::thread::LinearCombinationGELU<
        ElementOutput, VectorSize, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>,
    cutlass::epilogue::thread::LinearCombinationRelu<
        ElementOutput, VectorSize, ElementAccumulator, ElementComputeEpilogue,
        cutlass::epilogue::thread::ScaleType::NoBetaScaling>>;
// Number of pipelines you want to use
constexpr int NumStages = 2;

void generate_tensor_2D(float *ptr, int i_M, int i_N) {
  std::default_random_engine gen;
  std::uniform_real_distribution<float> distribution(-1.0, 1.0);
  for (int i = 0; i < i_M; i++) {
    for (int j = 0; j < i_N; j++) {
      *(ptr + i * i_N + j) = distribution(gen);
    }
  }
}

void print(float *ptr, int i_M, int i_N, char name) {
  std::cout << "matrix " << name << std::endl;
  for (int i = 0; i < i_M; i++) {
    for (int j = 0; j < i_N; j++) {
      std::cout << ptr[i * i_N + j] << " ";
    }
    std::cout << std::endl;
  }
}
int main(int argc, const char *argv[]) {
  int M = 1024;  // M
  int N = 1024;  // N
  int K = 1024;  // K
  // std::string act = "Relu";

  if (argc > 3) {
    M = atoi(argv[1]);
    N = atoi(argv[2]);
    K = atoi(argv[3]);
    // act = argv[4];
  }

  using EpilogueOp = EpilogueOpSwish;
  using CutlassGemm = cutlass::gemm::device::Gemm<
      ElementInputA, LayoutInputA, ElementInputB, LayoutInputB, ElementOutput,
      LayoutOutput, ElementAccumulator, MMAOp, SmArch, ShapeMMAThreadBlock,
      ShapeMMAWarp, ShapeMMAOp, EpilogueOp, SwizzleThreadBlock, NumStages>;
  int lda = K;
  int ldb = K;
  int ldc = N;
  int ldd = N;

  float alpha = 1.0;  // alpha
  float beta = 1.0;   // beta

  float *A;
  float *B;
  float *C;
  float *D;

  size_t A_mem_size =
      sizeof(float) * M * K;  // memory size of matrix A = M * K * sizeof(float)
  size_t B_mem_size =
      sizeof(float) * K * N;  // memory size of matrix B = K * N * sizeof(float)
  size_t C_mem_size =
      sizeof(float) * M * N;  // memory size of matrix C = M * N * sizeof(float)
  size_t D_mem_size =
      sizeof(float) * M * N;  // memory size of matrix C = M * N * sizeof(float)

  A = (float *)malloc(A_mem_size);
  B = (float *)malloc(B_mem_size);
  C = (float *)malloc(C_mem_size);
  D = (float *)malloc(D_mem_size);

  generate_tensor_2D(A, M, K);
  generate_tensor_2D(B, K, N);
  generate_tensor_2D(C, M, N);

  // print(A, M, K, 'A');
  // print(B, K, N, 'B');
  // print(C, M, N, 'C');

  float *d_A;
  float *d_B;
  float *d_C;
  float *d_D;

  cudaMalloc((void **)&d_A, A_mem_size);
  cudaMalloc((void **)&d_B, B_mem_size);
  cudaMalloc((void **)&d_C, C_mem_size);
  cudaMalloc((void **)&d_D, D_mem_size);

  cudaMemcpy(d_A, A, A_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_B, B, B_mem_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_C, C, C_mem_size, cudaMemcpyHostToDevice);

  CutlassGemm gemm_operator;
  CutlassGemm::Arguments args({M, N, K},       // Gemm Problem dimensions
                              {d_A, lda},      // source matrix A
                              {d_B, ldb},      // source matrix B
                              {d_C, ldc},      // source matrix C
                              {d_D, ldd},      // destination matrix D
                              {alpha, beta});  // alpha & beta
  cutlass::Status status;

  for (int i = 0; i < 5; ++i) {
    status = gemm_operator(args);
  }
  int iters = 1;
  // GpuTimer timer;
  // timer.start();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  cudaEventRecord(start);
  for (int i = 0; i < iters; ++i) {
    status = gemm_operator(args);
  }
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);
  printf("LOG >>> Execution Time (ms): %f\n", milliseconds / iters);

  // timer.stop();
  // printf("LOG >>> Execution Time (ms): %f\n", timer.elapsed_millis() /
  // iters); std::cout << "LOG >>> Execution Time(ms): "<<
  // timer.elapsed_millis() / iters<< std::endl;
  //
  // Return a cudaError_t if the CUTLASS GEMM operator returned an error code.
  //
  CUTLASS_CHECK(status);

  cudaMemcpy(D, d_D, D_mem_size, cudaMemcpyDeviceToHost);
  // cudaMemcpy(C, d_C, C_mem_size, cudaMemcpyDeviceToHost);
  // std::cout << D[0] << std::endl;
  // std::cout << D[M * N - 1] << std::endl;
  // print(D, M, N, 'D');
  return 0;
}
