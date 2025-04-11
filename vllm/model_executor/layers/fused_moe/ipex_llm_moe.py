# SPDX-License-Identifier: Apache-2.0

from abc import abstractmethod
from enum import Enum
from typing import Callable, List, Optional, Tuple
import os

import torch
import torch.nn.functional as F
from torch.nn.parameter import UninitializedParameter

import vllm.envs as envs
from vllm.config import get_current_vllm_config
from vllm.distributed import (get_dp_group, get_tensor_model_parallel_rank,
                              get_tensor_model_parallel_world_size,
                              tensor_model_parallel_all_reduce)
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.logger import init_logger
from vllm.model_executor.custom_op import CustomOp
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig, QuantizeMethodBase)
from vllm.model_executor.utils import set_weight_attrs
from vllm.platforms import current_platform
from vllm.platforms.interface import CpuArchEnum
from vllm.utils import direct_register_custom_op

fused_experts = None  # type: ignore
fused_moe_pallas = None  # type: ignore
if current_platform.is_xpu():
    from .moe_pallas import fused_moe_xpu
else:
    fused_moe_xpu = None  # type: ignore
logger = init_logger(__name__)

from ipex_llm.ggml.quantize import ggml_tensor_qtype, gguf_mixed_qtype
import ipex_llm.ggml.model.llama.llama_cpp as ggml
from ipex_llm.transformers.low_bit_linear import LowBitLinear, FP4Params, \
        FP16Linear, BF16Linear, ggml_convert_qtype, ggml_int4_convert_fp32

from vllm.model_executor.layers.fused_moe import FusedMoEMethodBase
from vllm.model_executor.layers.quantization.gguf import GGUFUninitializedParameter
from torch.nn.parameter import Parameter, UninitializedParameter
from vllm.model_executor.layers.activation import SiluAndMul

# def _fuse_mul_mat(x: torch.Tensor, qweight: torch.Tensor,
#                   qweight_type: int) -> torch.Tensor:
#     # HACK: when doing chunked prefill we don't generate output tokens
#     # so input to logits generator is empty which causes invalid parameter
#     if x.shape[0] == 0:
#         return torch.empty(x.shape[0],
#                            qweight.shape[0],
#                            dtype=x.dtype,
#                            device=x.device)
#     # there is no need to call any kernel for fp16/bf16
#     if qweight_type in UNQUANTIZED_TYPES:
#         return x @ qweight.T
#     # enable MMVQ in contiguous batching with batch_size=1
#     if x.shape[0] == 1 and qweight_type in MMVQ_QUANT_TYPES:
#         y = ops.ggml_mul_mat_vec_a8(qweight, x, qweight_type, qweight.shape[0])
#     # Use MMQ Kernel if it's available (standard + k-quants)
#     elif qweight_type in MMQ_QUANT_TYPES:
#         y = ops.ggml_mul_mat_a8(qweight, x, qweight_type, qweight.shape[0])
#     # If there is no available MMQ kernel, fallback to dequantize
#     elif qweight_type in DEQUANT_TYPES:
#         block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
#         shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
#         weight = ops.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
#         y = x @ weight.T
#     else:
#         # Raise an error if the quantization type is not supported.
#         # Might be useful if llama.cpp adds a new quantization type.
#         # Wrap to GGMLQuantizationType IntEnum to make sure it's a valid type.
#         qweight_type = WeightType(qweight_type)
#         raise NotImplementedError(
#             f"Unsupported GGUF quantization type: {qweight_type}")
#     return y

import gguf
from gguf import GGMLQuantizationType as WeightType
import ipex_llm.ggml.model.llama.llama_cpp as ggml
from ipex_llm.ggml.quantize import ggml_tensor_qtype

def _fuse_mul_mat(x: torch.Tensor, qweight: torch.Tensor,
                   qweight_type: int) -> torch.Tensor:
    # result = ggml_matmul_src1_x_src0_t(qweight, x, qweight_shape,qweight_type)
    # block_size, type_size = gguf.GGML_QUANT_SIZES[qweight_type]
    block_size = ggml.ggml_qk_size(ggml_tensor_qtype[qweight_type])
    type_size = ggml.ggml_type_size(qweight_type)
    shape = (qweight.shape[0], qweight.shape[1] // type_size * block_size)
    weight = ggml.ggml_dequantize(qweight, qweight_type, *shape, x.dtype)
    y = x @ weight.T
    return y


def _fused_moe_gguf(
    x: torch.Tensor,
    w1: torch.Tensor,
    w2: torch.Tensor,
    topk_weights: torch.Tensor,
    topk_ids: torch.Tensor,
    qweight_type: int,
    qweight_type2: int,
    act,
) -> torch.Tensor:
    # lazy import to avoid triggering triton import in CPU backend
    from vllm.model_executor.layers.fused_moe.fused_moe import (
        moe_align_block_size)

    out_hidden_states = torch.empty_like(x)

    logger.warning_once("There is no support for fast MoE kernel "
                        "for current quantization method. "
                        "Falling back to slow implementation. ")
    for tok, (w, idx) in enumerate(zip(topk_weights, topk_ids)):
        inp = x[tok].reshape((1, ) + x.shape[1:])
        current_hidden_state = None
        for ww, ii in zip(w, idx):
            expert_up = w1[ii]

            out = _fuse_mul_mat(inp, expert_up, qweight_type)
            out = act(out)

            expert_down = w2[ii]
            current_state = _fuse_mul_mat(out, expert_down,
                                            qweight_type2).mul_(ww)
            if current_hidden_state is None:
                current_hidden_state = current_state
            else:
                current_hidden_state.add_(current_state)
        out_hidden_states[tok] = current_hidden_state
    return out_hidden_states


# @CustomOp.register("unquantized_fused_moe")
class IPEXLLMFusedMoEMethod(FusedMoEMethodBase):
    """MoE method without quantization."""

    def create_weights(self, layer: torch.nn.Module, num_experts: int,
                       hidden_size: int, intermediate_size_per_partition: int,
                       params_dtype: torch.dtype, **extra_weight_attrs):

        # Fused gate_up_proj (column parallel)
        w13_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_size,
            dtype=params_dtype),
                                        requires_grad=False)
        layer.register_parameter("w13_weight", w13_weight)
        set_weight_attrs(w13_weight, extra_weight_attrs)

        # down_proj (row parallel)
        w2_weight = torch.nn.Parameter(torch.empty(
            num_experts,
            hidden_size,
            intermediate_size_per_partition,
            dtype=params_dtype),
                                       requires_grad=False)
        layer.register_parameter("w2_weight", w2_weight)
        set_weight_attrs(w2_weight, extra_weight_attrs)


    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        # w1: [num_experts, intermediate_size * 2, hidden_size]
        # w2: [num_experts, hidden_size, intermediate_size]
        self.num_experts = layer.w13_weight.data.shape[0]
        self.intermediate_size = layer.w2_weight.data.shape[2]
        self.hidden_size = layer.w13_weight.data.shape[2]

        # super().process_weights_after_loading(layer)
        
        device = layer.w13_weight.data.device
        lowbit = os.getenv("IPEX_LLM_LOWBIT", "sym_int4")
        qtype = ggml_tensor_qtype[lowbit]
        w13_params = FP4Params(data=layer.w13_weight.data,
                                requires_grad=False,
                                quantized=False,
                                _shape=None,
                                convert_shape_only=False,
                                qtype=qtype).to(device)
        layer._parameters['w13_weight'] = w13_params

        w2_params = FP4Params(data=layer.w2_weight.data,
                                requires_grad=False,
                                quantized=False,
                                _shape=None,
                                convert_shape_only=False,
                                qtype=qtype).to(device)
        layer._parameters['w2_weight'] = w2_params

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        router_logits: torch.Tensor,
        top_k: int,
        renormalize: bool,
        use_grouped_topk: bool = False,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        apply_router_weight_on_input: bool = False,
        activation: str = "silu",
    ) -> torch.Tensor:
        # topk_weights, topk_ids = FusedMoE.select_experts(
        #     hidden_states=x,
        #     router_logits=router_logits,
        #     use_grouped_topk=use_grouped_topk,
        #     top_k=top_k,
        #     renormalize=renormalize,
        #     topk_group=topk_group,
        #     num_expert_group=num_expert_group,
        #     custom_routing_function=custom_routing_function,
        #     scoring_func=scoring_func,
        #     e_score_correction_bias=e_score_correction_bias)
        # return _fused_moe_gguf(x, layer.w13_qweight, layer.w2_qweight,
        #                        topk_weights, topk_ids,
        #                        layer.w13_qweight_type.weight_type,
        #                        layer.w2_qweight_type.weight_type, self.act)

        return self.forward_xpu(
            x=x,
            layer=layer,
            router_logits=router_logits,
            top_k=top_k,
            renormalize=renormalize,
            use_grouped_topk=use_grouped_topk,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            global_num_experts=global_num_experts,
            expert_map=expert_map,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            activation=activation,
            apply_router_weight_on_input=apply_router_weight_on_input)


    def forward_xpu(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        use_grouped_topk: bool,
        top_k: int,
        router_logits: torch.Tensor,
        renormalize: bool,
        topk_group: Optional[int] = None,
        num_expert_group: Optional[int] = None,
        global_num_experts: int = -1,
        expert_map: Optional[torch.Tensor] = None,
        custom_routing_function: Optional[Callable] = None,
        scoring_func: str = "softmax",
        e_score_correction_bias: Optional[torch.Tensor] = None,
        activation: str = "silu",
        apply_router_weight_on_input: bool = False,
        **kwargs,
    ):
        return self.fused_moe_xpu(hidden_states=x,
                                w1=layer.w13_weight,
                                w2=layer.w2_weight,
                                topk=top_k,
                                gating_output=router_logits,
                                renormalize=renormalize)


    def fused_moe_xpu(
        self,
        hidden_states: torch.Tensor,
        w1: torch.Tensor,
        w2: torch.Tensor,
        gating_output: torch.Tensor,
        topk: int,
        renormalize: bool,
    ) -> torch.Tensor:
        """
        Args:
            hidden_states: [*, hidden_size]
            w1: [num_experts, intermediate_size * 2, hidden_size]
            w2: [num_experts, hidden_size, intermediate_size]
            gating_output: [*, num_experts]
        """
        orig_shape = hidden_states.shape
        hidden_size = hidden_states.shape[-1]
        num_tokens = hidden_states.shape[:-1].numel()
        num_experts = self.num_experts
        intermediate_size = self.intermediate_size

        device = hidden_states.device
        dtype = hidden_states.dtype
        hidden_states = hidden_states.view(num_tokens, hidden_size)
        gating_output = gating_output.view(num_tokens, num_experts)
        topk_weights, topk_indices = F.softmax(gating_output, dim=-1, dtype=torch.float).topk(topk, dim=-1)
        if renormalize:
            topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)
        topk_weights = topk_weights.to(dtype)

        topk_indices = topk_indices.flatten()
        topk_argsort_indices = topk_indices.argsort()
        topk_argsort_revert_indices = topk_argsort_indices.argsort()
        token_indices = torch.arange(num_tokens, device=device).repeat_interleave(topk)
        token_indices = token_indices[topk_argsort_indices]
        group_sizes = custom_histogram(topk_indices.to(torch.int32), 0, num_experts - 1)
        
        x = hidden_states[token_indices]

        lowbit = os.getenv("IPEX_LLM_LOWBIT", "sym_int4")
        qtype = ggml_tensor_qtype[lowbit]
        w1 = xe_linear.dequant(x, w1.contiguous(), qtype)
        w2 = xe_linear.dequant(x, w2.contiguous(), qtype)

        w1 = w1.view(num_experts, intermediate_size * 2, hidden_size)
        w2 = w2.view(num_experts, hidden_size, intermediate_size)

        w1 = w1.transpose(1, 2)
        w2 = w2.transpose(1, 2)
        
        x = custom_gmm(x, w1, group_sizes)
        x = F.silu(x[..., :intermediate_size]) * x[..., intermediate_size:]
        x = custom_gmm(x, w2, group_sizes)
        x = x[topk_argsort_revert_indices].reshape(-1, topk, hidden_size)

        x = x * topk_weights.unsqueeze_(dim=-1)
        x = x.sum(dim=-2)
        x = x.reshape(orig_shape)
        return x


def custom_histogram(indices, min, max):
    bin_counts = torch.histc(indices, bins=max - min + 1, min=min, max=max).to(torch.int32)
    return bin_counts


from ipex_llm.transformers.low_bit_linear import MatMulLowBit
import xe_linear

def custom_gmm(x, w, group_sizes):
    result = torch.zeros(
            (x.shape[0], w.shape[-1]),
            dtype=x.dtype,
            device=x.device
        )
    start = 0
    i = 0
    for end_index in group_sizes.tolist():
        if end_index > 0:
            end = start + end_index
            result[start:end] = torch.matmul(x[start:end], w[i])
            start = end
        i += 1
    return result
