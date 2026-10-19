"""Direct FA4 frame-page experiment on a CUDA device."""

from importlib.metadata import version

import pytest
import torch

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.geometry import Cosmos3NanoSimBimanualGeometry
from vllm_omni.experimental.ar_diffusion.capability import ARDiffusionKVBranchSpec
from vllm_omni.experimental.ar_diffusion.kv_cache.config import ARDiffusionKVConfig
from vllm_omni.experimental.ar_diffusion.kv_cache.manager import ARDiffusionKVCache
from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import ARDiffusionPagedLayerInputs, paged_write_attn
from vllm_omni.experimental.ar_diffusion.kv_cache.state import ARDiffusionKVState

pytestmark = [pytest.mark.core_model, pytest.mark.diffusion]


@pytest.mark.cuda
def test_fa4_discontiguous_frame_pages_and_real_text_length(monkeypatch):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is required")
    interface = pytest.importorskip("flash_attn.cute.interface")

    monkeypatch.setenv("VLLM_OMNI_AR_DIFFUSION_KV_GATHER", "1")

    page = Cosmos3NanoSimBimanualGeometry(720, 1280).tokens_per_frame(4)
    assert page == 23 * 40 + 4 == 924
    assert version("flash-attn-4").startswith("4.")
    if interface._get_device_arch() // 10 not in (9, 10, 11):
        pytest.skip("Installed FA4 paged K/V requires SM90/SM100/SM110")

    torch.manual_seed(7)
    query = torch.randn(page, 32, 128, device="cuda", dtype=torch.bfloat16)
    key = torch.randn(9, page, 8, 128, device="cuda", dtype=torch.bfloat16)
    value = torch.randn_like(key)
    ids = [7, 2, 5, 1]
    text_len = 37
    current_key, current_value = key[ids[2]].clone(), value[ids[2]].clone()
    text_key, text_value = key[ids[3], :text_len].clone(), value[ids[3], :text_len].clone()
    key[ids[2]].zero_()
    value[ids[2]].zero_()
    key[ids[-1], text_len:] = float("nan")
    value[ids[-1], text_len:] = float("nan")
    length = 3 * page + text_len
    table = torch.tensor([ids], device="cuda", dtype=torch.int32)
    qstart = torch.tensor([0, page], device="cuda", dtype=torch.int32)
    seqlens = torch.tensor([length], device="cuda", dtype=torch.int32)
    inputs = ARDiffusionPagedLayerInputs(
        layer_idx=torch.tensor(0),
        key_pool=key.flatten(0, 1),
        value_pool=value.flatten(0, 1),
        block_size=page,
        seq_len=page,
        video_slots=ids[2] * page + torch.arange(page, device="cuda"),
        action_slots=ids[3] * page + torch.arange(text_len, device="cuda"),
        block_table=table,
        query_start_loc=qstart,
        seq_lens=seqlens,
        max_query_len=page,
        max_seq_len=4 * page,
        fa4_frame_pages=True,
    )

    def attend():
        return paged_write_attn(inputs, query, current_key, current_value, text_key, text_value, 128**-0.5)

    actual = attend()
    torch.accelerator.synchronize()
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
    ) as trace:
        attend()
        torch.accelerator.synchronize()
    kernels = sorted({event.name for event in trace.events() if str(event.device_type) == "DeviceType.CUDA"})
    print(f"FA4 {version('flash-attn-4')} SM{interface._get_device_arch()} kernels={kernels}")
    assert any("flash" in name.lower() for name in kernels)
    ordered_key = torch.cat([key[index] for index in ids[:3]] + [key[ids[3], :text_len]], dim=0)
    ordered_value = torch.cat([value[index] for index in ids[:3]] + [value[ids[3], :text_len]], dim=0)
    expected = torch.nn.functional.scaled_dot_product_attention(
        query.transpose(0, 1).float(),
        ordered_key.transpose(0, 1).float(),
        ordered_value.transpose(0, 1).float(),
        scale=128**-0.5,
        enable_gqa=True,
    ).transpose(0, 1)
    torch.testing.assert_close(actual.float(), expected, atol=0.012, rtol=0.012)


@pytest.mark.cpu
def test_frame_pages_retain_hidden_frames_and_mask_text_padding():
    from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager

    page = 924
    cache = ARDiffusionKVCache(
        ARDiffusionKVConfig(enable=True, chunk_size=page, window_chunks=2, sink_chunks=1),
        num_layers=1,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float32,
        block_size=page,
        max_model_len=1 << 20,
        available_bytes=1 << 26,
        kv_branches=(ARDiffusionKVBranchSpec("main", 0),),
        session_capacity=1,
        max_scratch_frames_per_branch=2,
        max_scratch_tokens_per_branch=37,
        device=torch.device("cpu"),
        fa4_frame_pages=True,
    )
    assert type(cache.manager.coordinator.single_type_managers[0]) is FullAttentionManager
    initial_free = cache.manager.block_pool.get_num_free_blocks()
    adapter = cache.begin_request("frame-page-test")
    state = ARDiffusionKVState(cache, "session", {"main": adapter}, num_layers=1)
    for _ in range(8):
        cache.allocate_chunk(adapter)
        cache.commit_chunk(adapter)
    history = cache.block_table(adapter)
    assert len(history) == 8
    assert cache.null_block_id not in history
    assert adapter.compacted_tokens == 0
    context = state.get_kv_caches("main", 2 * page, extra_visible_tokens=2 * page)[0].forward_ctx
    context.prepare(torch.device("cpu"), action_len=37, query_len=2 * page)
    current = context.current_video_block_ids
    text = context.action_scratch_block_ids
    assert context.block_table.tolist() == [[history[0], *history[-2:], *current, *text]]
    assert context.seq_lens.tolist() == [5 * page + 37]
    assert context.current_video_slot_mapping[::page].remainder(page).tolist() == [0, 0]
    assert context.action_slot_mapping.remainder(page).tolist() == list(range(37))

    clean = state.get_kv_caches(
        "main",
        2 * page,
        commit_current=True,
        extra_visible_tokens=page,
        frame_causal=True,
    )[0].forward_ctx
    clean.prepare(torch.device("cpu"), action_len=37, query_len=2 * page)
    current, text = clean.current_video_block_ids, clean.action_scratch_block_ids
    assert clean.block_table.tolist() == [
        [history[0], *history[-2:], current[0], *text],
        [history[0], history[-1], *current, *text],
    ]
    assert clean.seq_lens.tolist() == [4 * page + 37, 4 * page + 37]
    state.close()
    assert cache.manager.block_pool.get_num_free_blocks() == initial_free
