# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in routed GELU expert fusion for Gemma4 XPU decode."""

import operator

import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized

from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.utils import Range
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner

from ..vllm_inductor_pass import VllmInductorPass

logger = init_logger(__name__)


class XpuGemmaMoEFusionPass(VllmInductorPass):
    """Fuse Gemma routing and routed experts; retain dense branch and norms."""

    def __init__(self, config: VllmConfig) -> None:
        super().__init__(config)
        from vllm._xpu_ops import xpu_gemma_moe_unsupported_reason

        self.max_tokens = (
            config.compilation_config.pass_config.xpu_gemma_moe_fusion_max_token_num
        )
        self.enabled = (
            hasattr(torch.ops.vllm, "xpu_gemma_moe_fused")
            and not config.parallel_config.use_ubatching
        )
        self.matched_count = 0
        if not self.enabled:
            return
        runners = get_layers_from_vllm_config(config, MoERunner)
        self.enabled = self.enabled and bool(runners)
        for name, runner in runners.items():
            reason = xpu_gemma_moe_unsupported_reason(runner)
            if reason is not None:
                logger.debug("Gemma expert fusion skipped for %s: %s", name, reason)
                self.enabled = False
                break
        self.matched_count = 0

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return self.enabled and compile_range.end <= self.max_tokens

    @VllmInductorPass.time_and_log
    def __call__(self, graph: fx.Graph) -> None:
        self.matched_count = 0
        if not self.enabled:
            return
        for node in list(graph.nodes):
            if (
                node.op != "call_function"
                or node.target is not auto_functionalized
                or not node.args
                or node.args[0] is not torch.ops.vllm.moe_forward.default
                or node.kwargs.get("shared_experts_input") is not None
                or node.kwargs.get("input_ids") is not None
                or node.kwargs.get("hidden_dim_unpadded", 0) != 0
                or len(node.users) != 1
            ):
                continue
            routed = next(iter(node.users))
            if (
                routed.op != "call_function"
                or routed.target is not operator.getitem
                or routed.args != (node, 0)
            ):
                continue
            hidden, logits, layer = (
                node.kwargs.get(n)
                for n in ("hidden_states", "router_logits", "layer_name")
            )
            val = hidden.meta.get("val") if isinstance(hidden, fx.Node) else None
            if (
                not isinstance(val, torch.Tensor)
                or val.ndim != 2
                or val.shape[1] != 2816
                or not val.is_contiguous()
                or val.dtype not in (torch.float16, torch.bfloat16)
                or val.device.type != "xpu"
                or logits is None
                or layer is None
            ):
                continue
            with graph.inserting_before(node):
                fused = graph.call_function(
                    torch.ops.vllm.xpu_gemma_moe_fused.default, (hidden, logits, layer)
                )
            fused.meta = dict(routed.meta)
            routed.replace_all_uses_with(fused)
            graph.erase_node(routed)
            graph.erase_node(node)
            self.matched_count += 1
        logger.info(
            "XpuGemmaMoEFusionPass replaced %d routed expert layers", self.matched_count
        )

    def uuid(self) -> str:
        from vllm._xpu_ops import xpu_gemma_moe_unsupported_reason

        return self.hash_source(self, xpu_gemma_moe_unsupported_reason)
