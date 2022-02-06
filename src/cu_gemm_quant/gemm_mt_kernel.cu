#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>
#include <vector>
#include <assert.h>

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))
#define DIVCEIL(a,b) (((a)%(b)!=0)?(a/b+1):(a/b))
#define IS_ROUND_UP(a, is_rnd, rnd_bit) (((a & is_rnd) == is_rnd) || ((a & ((a & rnd_bit) - ((a & rnd_bit) != 0))) != 0))

inline unsigned int intDivCeil(const unsigned int &a, const unsigned int &b) { return ( a%b != 0 ) ? (a/b+1) : (a/b); }


template <typename scalar_t>
__global__ void gemm_48_opt(
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> A_SL,
    const torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> B_SL,
    torch::PackedTensorAccessor32<scalar_t, 2, torch::RestrictPtrTraits> C,
    const bool is_round) {

    int bx = blockIdx.x;
    int by = blockIdx.y;

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int row = blockDim.x * bx + tx;
    int col = blockDim.y * by + ty;

    scalar_t psum = 0;

    if (row < C.size(0) && col < C.size(1)) {
        for (int k = 0; k < A.size(1); k+=2) {

            float a1 = (k < A.size(1)) ? A[row][k] : 0;
            float b1 = (k < A.size(1)) ? B[col][k] : 0;
            float a1_sl = (k < A.size(1)) ? A_SL[row][k] : 0;
            float b1_sl = (k < A.size(1)) ? B_SL[col][k] : 0;

            float a2 = (k+1 < A.size(1)) ? A[row][k+1] : 0;
            float b2 = (k+1 < A.size(1)) ? B[col][k+1] : 0;
            float a2_sl = (k+1 < A.size(1)) ? A_SL[row][k+1] : 0;
            float b2_sl = (k+1 < A.size(1)) ? B_SL[col][k+1] : 0;

            //assert((a < 256) && (a >= 0));
            //assert((b <= 127) && (b >= -128));

            if (is_round) {
                b1 = std::nearbyint(b1 / b1_sl) * b1_sl;
                b2 = std::nearbyint(b2 / b2_sl) * b2_sl;
            }
            else {
                b1 = floor(b1 / b1_sl) * b1_sl;
                b2 = floor(b2 / b2_sl) * b2_sl;
            }

            if (a1 == 0) {
                psum += a2 * b2;
            }
            else if (a2 == 0) {
                psum += a1 * b1;
            }
            else {
                if (is_round) {
                    a1 = (a1_sl != 0) ? std::nearbyint(a1 / a1_sl) * a1_sl : 0;
                    a2 = (a2_sl != 0) ? std::nearbyint(a2 / a2_sl) * a2_sl : 0;
                }
                else {
                    a1 = (a1_sl != 0) ? floor(a1 / a1_sl) * a1_sl : 0;
                    a2 = (a2_sl != 0) ? floor(a2 / a2_sl) * a2_sl : 0;
                }

                psum += a1 * b1 + a2 * b2;
            }
        }

        C[row][col] = psum;
    }
}


std::vector<torch::Tensor> gemm_sparq_cuda(
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor a_sl,
	torch::Tensor b_sl,
    const bool is_round) {

    torch::Device device = torch::kCUDA;

    auto output = torch::zeros({a.size(0), b.size(0)}, device);
    auto stats = torch::zeros({8, a.size(0), b.size(0)}, device);

    const int block_size = 16;
    const dim3 threads(block_size, block_size);
    const dim3 grid(DIVCEIL(output.size(0), threads.x), DIVCEIL(output.size(1), threads.y));

    AT_DISPATCH_FLOATING_TYPES(a.scalar_type(), "_gemm", ([&] {
        gemm_48_opt<scalar_t><<< grid, threads >>>(
          a.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          b.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          a_sl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          b_sl.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          output.packed_accessor32<scalar_t, 2, torch::RestrictPtrTraits>(),
          is_round
        );
    }));

    cudaError_t code = cudaGetLastError();
    if (code != cudaSuccess) {
        const char * errorMessage = cudaGetErrorString(code);
        fprintf(stderr, "CUDA error: (%d) %s\n", code, errorMessage);
    }

    return {output, stats};
}
