import torch

def int4FusedDequantize(A: torch.Tensor, B: torch.Tensor,
                        scale_row: torch.Tensor, scale_col: torch.Tensor,
                        vec_a_add: torch.Tensor, vec_b_add: torch.Tensor,
                        y: torch.Tensor) -> torch.Tensor: ...