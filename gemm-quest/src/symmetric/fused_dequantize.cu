#include "int4.h"
#include "symmetric/gemm/device/gemm_dequant.h"
#include "symmetric/symmetric_internal.h"
#include "util.h"
#include <c10/cuda/CUDAStream.h>

namespace QUIK::symmetric {

torch::Tensor int4FusedDequantizeCUDA(const torch::Tensor &A,
                                      const torch::Tensor &B,
                                      const torch::Tensor &scale_row,
                                      const torch::Tensor &scale_col,
                                      const torch::Tensor &vec_a_add,
                                      const torch::Tensor &vec_b_add,
                                      const torch::Tensor &y) {
  torch::checkAllSameGPU("int4FusedDequantize", {{A, "A", 0},
                                                 {B, "B", 1},
                                                 {scale_row, "scale_row", 2},
                                                 {scale_col, "scale_col", 3},
                                                 {vec_a_add, "vec_a_add", 4},
                                                 {vec_b_add, "vec_b_add", 5},
                                                 {y, "y", 6}});
  auto M = A.size(0);
  auto N = B.size(0);
  auto K = A.size(1) * kElementsPerVector;
  auto D = torch::empty({0}, torch::dtype(torch::kF16).device(A.device()));

  using Gemm = cutlass::gemm::device::symmetric::GemmDequant<
      cutlass::int4b_t,                // ElementA
      cutlass::layout::RowMajor,       // LayoutA
      cutlass::int4b_t,                // ElementB
      cutlass::layout::ColumnMajor,    // LayoutB
      cutlass::half_t,                 // ElementOutput
      cutlass::layout::RowMajor,       // LayoutOutput
      int32_t,                         // ElementAccumulator
      cutlass::arch::OpClassTensorOp,  // tag indicating Tensor Cores
      cutlass::arch::Sm80,  // tag indicating target GPU compute architecture
      cutlass::gemm::GemmShape<128, 128, 128>,
      cutlass::gemm::GemmShape<64, 64, 128>
      /* cutlass::gemm::GemmShape<16, 8, 64>,
      cutlass::epilogue::thread::symmetric::LinearCombinationDequant<
            cutlass::half_t,
            128 / cutlass::sizeof_bits<cutlass::half_t>::value,
            int32_t,
            cutlass::half_t,
            cutlass::epilogue::thread::symmetric::MyScaleType::Dequantize,
            cutlass::FloatRoundStyle::round_to_nearest, cutlass::half_t>,
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
      2 */
      //16x8x128
      >;
  /* cutlass::gemm::GemmShape<128, 128, 128>,
  cutlass::gemm::GemmShape<64, 64, 128>, */
  Gemm gemmOp;

  using GemmCoord = cutlass::gemm::GemmCoord;

  typename Gemm::Arguments arguments{
      {static_cast<GemmCoord::Index>(M), static_cast<GemmCoord::Index>(N),
       static_cast<GemmCoord::Index>(K)},
      {(cutlass::int4b_t *)A.data_ptr<uint8_t>(), K},
      {(cutlass::int4b_t *)B.data_ptr<uint8_t>(), K},
      {(cutlass::half_t *)D.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)y.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_col.data_ptr<torch::Half>(), N},
      {(cutlass::half_t *)scale_row.data_ptr<torch::Half>(), M},
      //{(cutlass::half_t *)vec_a_add.data_ptr<torch::Half>(), N},
      //{(cutlass::half_t *)vec_b_add.data_ptr<torch::Half>(), M},
      {(float *)vec_a_add.data_ptr<float>(), N},
      {(float *)vec_b_add.data_ptr<float>(), M},
      Gemm::ElementC(0)};

  cudaStream_t stream = at::cuda::getCurrentCUDAStream(A.device().index());
  auto status = gemmOp(arguments, nullptr, stream);

  TORCH_CHECK(status == cutlass::Status::kSuccess,
              cutlassGetStatusString(status))

  return D;
}
}  // namespace QUIK::symmetric