# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import operator

import pytest
import torch
from torch import fx
from torch._higher_order_ops.auto_functionalize import auto_functionalized
from torch._subclasses.fake_tensor import FakeTensorMode

import vllm._xpu_ops as xpu_ops
import vllm.model_executor.layers.fused_moe.runner.moe_runner  # noqa: F401
from vllm.compilation.passes.fusion import xpu_gemma_fusion as fusion
from vllm.config import VllmConfig
from vllm.config.compilation import PassConfig
from vllm.config.utils import Range


@pytest.fixture(scope="module", autouse=True)
def register_test_op():
    lib = None
    if not hasattr(torch.ops.vllm, "xpu_gemma_moe_fused"):
        lib = torch.library.Library("vllm", "FRAGMENT")
        lib.define(
            "xpu_gemma_moe_fused(Tensor hidden_states, Tensor router_logits, "
            "str layer_name) -> Tensor"
        )
    yield
    del lib


@pytest.fixture
def gemma_pass(monkeypatch):
    monkeypatch.setattr(
        fusion, "get_layers_from_vllm_config", lambda *_: {"layer": object()}
    )
    monkeypatch.setattr(xpu_ops, "xpu_gemma_moe_unsupported_reason", lambda _: None)
    return fusion.XpuGemmaMoEFusionPass(VllmConfig())


def graph(
    dtype=torch.float16,
    hidden=2816,
    other_output=False,
    shared=False,
    padded=False,
    strided=False,
):
    g = fx.Graph()
    x, logits = g.placeholder("hidden"), g.placeholder("logits")
    with FakeTensorMode():
        xv = torch.empty(
            2, hidden * 2 if strided else hidden, dtype=dtype, device="xpu"
        )
        if strided:
            xv = xv[:, ::2]
        lv = torch.empty(2, 128, dtype=torch.float32, device="xpu")
    x.meta["val"], logits.meta["val"] = xv, lv
    node = g.call_function(
        auto_functionalized,
        (torch.ops.vllm.moe_forward.default,),
        {
            "hidden_states": x,
            "router_logits": logits,
            "layer_name": "layer",
            "shared_experts_input": x if shared else None,
            "input_ids": None,
            "hidden_dim_unpadded": hidden if padded else 0,
        },
    )
    y = g.call_function(operator.getitem, (node, 0))
    y.meta["val"] = xv
    if other_output:
        modified = g.call_function(operator.getitem, (node, 1))
        g.output((y, modified))
    else:
        g.output((y,))
    return g


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_rewrites_routed_output_without_changing_callers(gemma_pass, dtype):
    g = graph(dtype)
    gemma_pass(g)
    g.lint()
    assert gemma_pass.matched_count == 1
    targets = [n.target for n in g.nodes]
    assert torch.ops.vllm.xpu_gemma_moe_fused.default in targets
    assert auto_functionalized not in targets


@pytest.mark.parametrize(
    "kwargs",
    [
        {"dtype": torch.float32},
        {"hidden": 5376},
        {"other_output": True},
        {"shared": True},
        {"padded": True},
        {"strided": True},
    ],
)
def test_unsupported_graph_retains_original_moe(gemma_pass, kwargs):
    g = graph(**kwargs)
    gemma_pass(g)
    assert gemma_pass.matched_count == 0
    assert any(n.target is auto_functionalized for n in g.nodes)


def test_prefill_ranges_skip_expert_fusion(gemma_pass):
    assert gemma_pass.is_applicable_for_range(Range(1, 8))
    assert not gemma_pass.is_applicable_for_range(Range(9, 1024))


def test_any_unsupported_layer_keeps_original_graph(monkeypatch):
    monkeypatch.setattr(
        fusion, "get_layers_from_vllm_config", lambda *_: {"layer": object()}
    )
    monkeypatch.setattr(xpu_ops, "xpu_gemma_moe_unsupported_reason", lambda _: "layout")
    p = fusion.XpuGemmaMoEFusionPass(VllmConfig())
    g = graph()
    p(g)
    assert p.matched_count == 0


@pytest.mark.parametrize("n", [0, 9])
def test_max_token_range_rejects_unsupported_limit(n):
    with pytest.raises(ValueError):
        PassConfig(xpu_gemma_moe_fusion_max_token_num=n)


def test_default_off():
    assert not PassConfig().fuse_xpu_gemma_moe


def test_companion_module_missing_disables_availability(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "vllm_xpu_kernels.gemma_moe_fused_interface", None)
    assert not xpu_ops.xpu_gemma_moe_fused_available()


def test_missing_native_reason_returns_before_optional_import(monkeypatch):
    monkeypatch.setattr(xpu_ops, "xpu_gemma_moe_fused_available", lambda: False)
    reason = xpu_ops.xpu_gemma_moe_unsupported_reason(object())
    assert reason is not None and "unavailable" in reason


@pytest.mark.parametrize("kind", ["gemma", "other", "override", "eplb"])
def test_native_routing_requires_exact_gemma_semantics(kind):
    from types import SimpleNamespace

    from vllm.model_executor.models.gemma4 import Gemma4Router

    owner = Gemma4Router.__new__(Gemma4Router)
    torch.nn.Module.__init__(owner)
    fn = owner.routing_function
    if kind == "other":
        fn = lambda **kwargs: None
    if kind == "override":

        def replacement(self, **kwargs):
            return None

        fn = replacement.__get__(owner, Gemma4Router)
    router = SimpleNamespace(
        custom_routing_function=fn,
        top_k=8,
        global_num_experts=128,
        eplb_state=object() if kind == "eplb" else None,
    )
    runner = SimpleNamespace(router=router)
    result = xpu_ops._xpu_gemma_routing_owner(runner)
    assert (result is owner) if kind == "gemma" else result is None


@pytest.mark.parametrize("matrix", [None, "w13", "w2"])
@pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")
def test_offset_expert_weight_retains_unfused_graph(monkeypatch, matrix):
    from types import SimpleNamespace as NS

    from vllm.model_executor.layers.fused_moe.activation import MoEActivation
    from vllm.model_executor.layers.fused_moe.experts.xpu_moe import XPUExpertsFp8
    from vllm.model_executor.models.gemma4 import Gemma4Router

    def weight(shape, offset):
        n = 1
        for d in shape:
            n *= d
        storage = torch.empty(n + 64, device="xpu:0", dtype=torch.float8_e4m3fn)
        return storage[offset : offset + n].view(shape)

    w13 = weight((128, 2816, 704), int(matrix == "w13"))
    w2 = weight((128, 352, 2816), int(matrix == "w2"))
    experts = XPUExpertsFp8.__new__(XPUExpertsFp8)
    experts.quant_config = NS(
        w1_scale=torch.ones(128, device="xpu:0"),
        w2_scale=torch.ones(128, device="xpu:0"),
    )
    experts.gemm1_clamp_limit = None
    owner = Gemma4Router.__new__(Gemma4Router)
    torch.nn.Module.__init__(owner)
    owner.per_expert_scale = torch.nn.Parameter(torch.ones(128, device="xpu:0"))
    routed = NS(
        w13_weight=w13,
        w2_weight=w2,
        apply_router_weight_on_input=False,
        quant_method=NS(
            moe_kernel=NS(fused_experts=experts), topk_indices_dtype=torch.int32
        ),
    )
    parallel = NS(
        use_ep=False,
        ep_size=1,
        enable_eplb=False,
        dp_size=1,
        pcp_size=1,
        is_sequence_parallel=False,
    )
    runner = NS(
        routed_experts=routed,
        activation=MoEActivation.GELU_TANH,
        shared_experts=None,
        routed_scaling_factor=1.0,
        expert_map=None,
        moe_config=NS(
            has_bias=False,
            is_lora_enabled=False,
            moe_parallel_config=parallel,
            in_dtype=torch.float16,
            experts_per_token=8,
        ),
        router=NS(
            custom_routing_function=owner.routing_function,
            top_k=8,
            global_num_experts=128,
            eplb_state=None,
        ),
    )
    monkeypatch.setattr(
        fusion, "get_layers_from_vllm_config", lambda *_: {"layer": runner}
    )
    p = fusion.XpuGemmaMoEFusionPass(VllmConfig())
    g = graph()
    p(g)
    g.lint()
    assert p.matched_count == (1 if matrix is None else 0)
    assert any(n.target is auto_functionalized for n in g.nodes) == (matrix is not None)


@pytest.mark.parametrize("maximum", [4, 8, 16])
def test_gemma_endpoint_respects_scheduler_upper_bound(maximum):
    from vllm.config.compilation import CompilationConfig
    from vllm.config.scheduler import SchedulerConfig

    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=maximum,
            max_model_len=maximum,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(),
    )
    config.compilation_config.pass_config.fuse_xpu_gemma_moe = True
    config._set_compile_ranges()
    ranges = config.compilation_config.get_compile_ranges()
    assert ranges[-1].end == maximum
    assert all(r.end <= maximum for r in ranges)
    if maximum > 8:
        assert ranges[0] == Range(1, 8)


def test_configured_smaller_limit_restricts_fusion_ranges(gemma_pass):
    from vllm.config.compilation import CompilationConfig
    from vllm.config.scheduler import SchedulerConfig

    config = VllmConfig(
        scheduler_config=SchedulerConfig(
            max_num_seqs=1,
            max_num_batched_tokens=16,
            max_model_len=16,
            is_encoder_decoder=False,
        ),
        compilation_config=CompilationConfig(),
    )
    config.compilation_config.pass_config.fuse_xpu_gemma_moe = True
    config.compilation_config.pass_config.xpu_gemma_moe_fusion_max_token_num = 4
    config._set_compile_ranges()
    p = fusion.XpuGemmaMoEFusionPass(config)
    ranges = config.compilation_config.get_compile_ranges()
    assert ranges == [Range(1, 4), Range(5, 16)]
    assert p.is_applicable_for_range(ranges[0])
    assert not p.is_applicable_for_range(ranges[1])


@pytest.mark.parametrize(
    "enable_dbo,ubatch_size", [(False, 0), (False, 1), (False, 2), (True, 0)]
)
def test_microbatching_keeps_original_gemma_graph(gemma_pass, enable_dbo, ubatch_size):
    from vllm.config.parallel import ParallelConfig

    config = VllmConfig()
    config.parallel_config = ParallelConfig(
        enable_dbo=enable_dbo, ubatch_size=ubatch_size
    )
    pass_ = fusion.XpuGemmaMoEFusionPass(config)
    expected = not config.parallel_config.use_ubatching
    assert pass_.is_applicable_for_range(Range(1, 8)) == expected
    g = graph()
    pass_(g)
    g.lint()
    assert pass_.matched_count == (1 if expected else 0)
    assert any(node.target is auto_functionalized for node in g.nodes) == (not expected)
