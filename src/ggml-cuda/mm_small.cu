#include "mm_small.cuh"

// One thread per output element; F16 weights converted on-the-fly.
// Avoids cuBLAS launch overhead for very small matrices.
static __global__ void small_mm_f16_f32(
        const half * __restrict__ w,
        const float * __restrict__ a,
        float * __restrict__ dst,
        const int M, const int N, const int K) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= M * N) {
        return;
    }

    const int m = idx / N;
    const int n = idx % N;

    float sum = 0.0f;
    const half * w_row = w + (int64_t)m * K;
    const float * a_col = a + n;
    for (int k = 0; k < K; k++) {
        sum += __half2float(w_row[k]) * a_col[(int64_t)k * N];
    }
    dst[idx] = sum;
}

bool ggml_cuda_should_use_small_mm(const ggml_tensor * src0, const ggml_tensor * src1) {
    if (src0->type != GGML_TYPE_F16 || src1->type != GGML_TYPE_F32) {
        return false;
    }
    const int64_t M = src0->ne[1];
    const int64_t N = src1->ne[1];
    const int64_t K = src0->ne[0];

    return M * N <= SMALL_MM_MAX_ELTS && K <= SMALL_MM_MAX_K && ggml_is_contiguous(src0);
}

void ggml_cuda_op_mul_mat_small(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst,
    const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i,
    const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream) {

    GGML_ASSERT(src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    const int64_t K = src0->ne[0];
    const int64_t M = row_high - row_low;
    const int64_t N = src1_ncols;

    const int total = M * N;
    const int block_size = 256;
    const int num_blocks = (total + block_size - 1) / block_size;

    small_mm_f16_f32<<<num_blocks, block_size, 0, stream>>>(
        (const half *)src0_dd_i, src1_ddf_i, dst_dd_i, M, N, K);

    GGML_UNUSED_VARS(ctx, dst, src1_ddq_i, src1_padded_row_size);
}
