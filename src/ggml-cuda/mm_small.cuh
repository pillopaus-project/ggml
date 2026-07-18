#pragma once
#include "common.cuh"

// Thresholds for the small-matmul custom kernel.
// When total output elements <= SMALL_MM_MAX_ELTS and K <= SMALL_MM_MAX_K
// the kernel is used instead of cuBLAS to avoid launch overhead.
#define SMALL_MM_MAX_ELTS 4096
#define SMALL_MM_MAX_K    512

bool ggml_cuda_should_use_small_mm(const ggml_tensor * src0, const ggml_tensor * src1);

void ggml_cuda_op_mul_mat_small(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);
