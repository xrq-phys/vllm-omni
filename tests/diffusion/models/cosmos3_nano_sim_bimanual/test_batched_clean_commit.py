# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Compare real clean-refresh layer math and managed caches across window rolls."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from tests.diffusion.models.cosmos3_nano_sim_bimanual.test_frame_causal_attention import fa4_config as fa4_config
from tests.helpers.mark import hardware_test
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.frame_causal_attention import FrameCausalAttentionState
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.transformer_cosmos3_nano_sim_bimanual import (
    Cosmos3NanoSimBimanualGenDecoderLayer,
    Cosmos3NanoSimBimanualJointAttention,
    Cosmos3NanoSimBimanualTransformer,
)
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache import ARDiffusionKVCache, ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]
BLOCK, WIDTH, HEAD, LAYERS = 16, 64, 64, 3


@pytest.fixture(autouse=True)
def cpu_attention_oracle(monkeypatch, request):
    if request.node.get_closest_marker("cpu") is None:
        return
    from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual import transformer_cosmos3_nano_sim_bimanual as module

    def oracle(inputs, q, k, v, tk, tv, scale):
        assert q.device.type == "cpu"
        size = inputs.block_size
        for pool, values, text in [(inputs.key_pool, k, tk), (inputs.value_pool, v, tv)]:
            pool.index_copy_(0, inputs.video_slots, values)
            pool.index_copy_(0, inputs.action_slots, text)
        result = []
        for row, length in enumerate(inputs.seq_lens.tolist()):
            pages = inputs.block_table[row, : (length + size - 1) // size].long()
            key = inputs.key_pool.unflatten(0, (-1, size))[pages].flatten(0, 1)[:length]
            value = inputs.value_pool.unflatten(0, (-1, size))[pages].flatten(0, 1)[:length]
            start, end = inputs.query_start_loc[row : row + 2].tolist()
            result.append(
                torch.nn.functional.scaled_dot_product_attention(
                    q[start:end].transpose(0, 1),
                    key.transpose(0, 1),
                    value.transpose(0, 1),
                    scale=scale,
                    enable_gqa=True,
                ).transpose(0, 1)
            )
        return torch.cat(result)

    monkeypatch.setattr(module, "paged_write_attn", oracle)


def layers(device, dtype):
    """Use production forward methods with small weights; no distributed init."""
    result = nn.ModuleList()
    for _ in range(LAYERS):
        attn = Cosmos3NanoSimBimanualJointAttention.__new__(Cosmos3NanoSimBimanualJointAttention)
        nn.Module.__init__(attn)
        attn.num_heads_local, attn.num_kv_heads_local, attn.head_dim = 4, 2, HEAD
        attn.qk_norm = True
        attn.to_q, attn.to_k, attn.to_v = [nn.Linear(WIDTH, n * HEAD, bias=False) for n in (4, 2, 2)]
        attn.to_out = nn.Linear(4 * HEAD, WIDTH, bias=False)
        attn.norm_q, attn.norm_k = nn.RMSNorm(HEAD), nn.RMSNorm(HEAD)
        attn.norm_q.variance_epsilon = attn.norm_k.variance_epsilon = 1e-6
        layer = Cosmos3NanoSimBimanualGenDecoderLayer.__new__(Cosmos3NanoSimBimanualGenDecoderLayer)
        nn.Module.__init__(layer)
        layer.cross_attention = attn
        layer.input_layernorm = nn.RMSNorm(WIDTH, eps=1e-6)
        layer.post_attention_layernorm = nn.RMSNorm(WIDTH, eps=1e-6)
        layer.mlp = nn.Sequential(nn.Linear(WIDTH, 2 * WIDTH), nn.SiLU(), nn.Linear(2 * WIDTH, WIDTH))
        result.append(layer)
    return result.to(device=device, dtype=dtype).eval()


def state(device, dtype, window, sink, page_size=None):
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=BLOCK, window_chunks=window, sink_chunks=sink),
        num_layers=LAYERS,
        num_kv_heads=2,
        head_size=HEAD,
        dtype=dtype,
        block_size=BLOCK if page_size is None else page_size,
        max_model_len=4096,
        available_bytes=1 << 25,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        frames_per_block=4,
        max_scratch_tokens_per_branch=BLOCK,
        device=device,
    )
    return FrameCausalAttentionState(
        ARDiffusionKVState(cache, "test", {"main": cache.begin_request("main")}, num_layers=LAYERS)
    )


@torch.no_grad()
def run_forward(net, h, text, start, *, paged, batched=False, window=5, sink=0, commit=True):
    frames = h.shape[1] // BLOCK
    nulls = tuple(i for i in range(frames) if (start + i) % 3 == 0)
    positions = torch.arange(start * BLOCK, (start + frames) * BLOCK, device=h.device).float()
    phases = positions.view(1, -1, 1, 1) * torch.arange(1, HEAD + 1, device=h.device).view(1, 1, 1, -1) * 0.001
    cos, sin = phases.cos().to(h.dtype), phases.sin().to(h.dtype)
    contexts = paged.get_kv_caches("main", frames * BLOCK, commit_current=commit, frame_causal=batched)
    contexts[0].forward_ctx.prepare(device=h.device, action_len=text[0][0].shape[1], query_len=h.shape[1])
    for index, layer in enumerate(net):
        h, _, _ = layer(
            h,
            text_k=text[index][0],
            text_v=text[index][1],
            real_text_kv_len=text[index][0].shape[1],
            freqs_cos=cos,
            freqs_sin=sin,
            paged_context=contexts[index].to_layer_inputs(),
            num_frames=frames,
            tokens_per_frame=BLOCK,
            action_tokens_per_frame=2,
            null_action_frame_indexes=nulls,
            clean_history_window=(sink, window) if batched else None,
        )
    paged.commit_paged_context("main")
    return h


def compare(*, device, dtype, history, window, sink, frames, page_size=None):
    torch.manual_seed(123)
    net = layers(device, dtype)
    text = [
        (torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype), torch.randn(1, 5, 2, HEAD, device=device, dtype=dtype))
        for _ in net
    ]
    inputs = torch.randn(1, (history + frames + 1) * BLOCK, WIDTH, device=device, dtype=dtype)
    a, b = [state(device, dtype, window, sink, page_size) for _ in range(2)]
    tolerance = 5e-2 if dtype == torch.bfloat16 else 2e-5
    try:
        for i in range(history):
            x = inputs[:, i * BLOCK : (i + 1) * BLOCK]
            for cache in (a, b):
                run_forward(net, x, text, i, paged=cache, window=window, sink=sink)
        sequential = [
            run_forward(net, inputs[:, i * BLOCK : (i + 1) * BLOCK], text, i, paged=a, window=window, sink=sink)
            for i in range(history, history + frames)
        ]
        actual = run_forward(
            net,
            inputs[:, history * BLOCK : (history + frames) * BLOCK],
            text,
            history,
            paged=b,
            batched=True,
            window=window,
            sink=sink,
        )
        torch.testing.assert_close(actual, torch.cat(sequential, dim=1), atol=tolerance, rtol=tolerance)
        assert a.adapter("main").completed_chunks == b.adapter("main").completed_chunks == history + frames
        ia, ib = [cache.kv_cache.window_block_ids(cache.adapter("main")) for cache in (a, b)]
        assert len(ia) == len(ib)
        for layer in range(LAYERS):
            for accessor in ("key_cache", "value_cache"):
                ka = getattr(a.kv_cache, accessor)(layer)[ia]
                kb = getattr(b.kv_cache, accessor)(layer)[ib]
                torch.testing.assert_close(ka, kb, atol=tolerance, rtol=tolerance)
        # The next denoising call consumes the newly committed history.
        outputs = [
            run_forward(
                net, inputs[:, -BLOCK:], text, history + frames, paged=cache, window=window, sink=sink, commit=False
            )
            for cache in (a, b)
        ]
        torch.testing.assert_close(*outputs, atol=tolerance, rtol=tolerance)
    finally:
        a.close()
        b.close()


@pytest.mark.cpu
@pytest.mark.parametrize("frames", [1, 3, 4])
@pytest.mark.parametrize(
    "history,window,sink", [(0, 5, 0), (2, 5, 0), (4, 5, 0), (8, 5, 0), (8, 2, 0), (8, 2, 1), (0, 1, 2)]
)
def test_batched_clean_cache_matches_sequential_cpu(history, window, sink, frames):
    compare(
        device=torch.device("cpu"),
        dtype=torch.float32,
        history=history,
        window=window,
        sink=sink,
        frames=frames,
    )


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged FlashAttention")
@pytest.mark.parametrize("history,window,sink,frames", [(0, 5, 0, 4), (4, 5, 0, 4), (8, 2, 0, 3), (8, 2, 1, 4)])
@pytest.mark.usefixtures("fa4_config")
def test_batched_clean_cache_matches_sequential_cuda(history, window, sink, frames):
    compare(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        history=history,
        window=window,
        sink=sink,
        frames=frames,
    )


@pytest.mark.cpu
@pytest.mark.parametrize("nulls", [(), (0,), (1, 3), (0, 1, 2, 3)])
@pytest.mark.parametrize("fps", [15.0, 29.97])
def test_clean_rope_matches_individual_frames(nulls, fps):
    # Return the position IDs themselves to inspect the real packing code.
    def rotary(hidden, *, position_ids):
        ids = position_ids[:, 0].T.unsqueeze(0)
        return ids, torch.zeros_like(ids)

    model = SimpleNamespace(
        manifest=SimpleNamespace(action_tokens_per_frame=4),
        temporal_modality_margin=2,
        base_fps=24.0,
        temporal_compression_factor=4,
        enable_fps_modulation=True,
        language_model=SimpleNamespace(rotary_emb=rotary),
    )
    common = dict(grid_h=2, grid_w=2, real_text_kv_len=5, fps=fps)
    hidden = torch.empty(1, 32, WIDTH)
    batched, _ = Cosmos3NanoSimBimanualTransformer._current_rope(
        model,
        hidden,
        frame_start=7,
        num_frames=4,
        null_action_frame_indexes=nulls,
        frame_causal=True,
        **common,
    )
    sequential = [
        Cosmos3NanoSimBimanualTransformer._current_rope(
            model,
            hidden[:, :8],
            frame_start=7 + i,
            num_frames=1,
            null_action_frame_indexes=(0,) if i in nulls else (),
            **common,
        )[0]
        for i in range(4)
    ]
    torch.testing.assert_close(batched, torch.cat(sequential, dim=1), rtol=0, atol=0)


@pytest.mark.cpu
def test_non_multiple_of_16_frame_geometry_cpu(monkeypatch):
    monkeypatch.setitem(globals(), "BLOCK", 394)
    compare(
        device=torch.device("cpu"),
        dtype=torch.float32,
        history=1,
        window=5,
        sink=0,
        frames=2,
        page_size=394,
    )


@hardware_test(res={"cuda": ["H100", "B200"]}, num_cards=1)
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA paged attention")
@pytest.mark.parametrize("tokens_per_frame", [394, 924])
@pytest.mark.parametrize("frames", [3, 4])
@pytest.mark.usefixtures("fa4_config")
def test_real_frame_geometry_cuda(monkeypatch, tokens_per_frame, frames):
    monkeypatch.setitem(globals(), "BLOCK", tokens_per_frame)
    compare(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
        history=4,
        window=2,
        sink=1,
        frames=frames,
        page_size=tokens_per_frame,
    )
