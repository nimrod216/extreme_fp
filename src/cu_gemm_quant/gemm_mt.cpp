#include <torch/extension.h>

#include <vector>
#include <assert.h>

// CUDA forward declerations

std::vector<torch::Tensor> gemm_sparq_cuda(
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor a_sl,
	torch::Tensor b_sl,
    const bool is_round);

// C++ interface

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)


std::vector<torch::Tensor> gemm_sparq_aux(
	torch::Tensor a,
	torch::Tensor b,
	torch::Tensor a_sl,
	torch::Tensor b_sl,
    const bool is_round) {

    CHECK_INPUT(a);
    CHECK_INPUT(b);
    CHECK_INPUT(a_sl);
    CHECK_INPUT(b_sl);

    return gemm_sparq_cuda(a, b, a_sl, b_sl, is_round);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("run", &gemm_sparq_aux, "GEMM 2x4b-8b simulation");
}

