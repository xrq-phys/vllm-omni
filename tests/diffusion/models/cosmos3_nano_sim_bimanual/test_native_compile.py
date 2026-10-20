# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Admission contracts for the native Inductor path and Sim parallel execution."""

from types import SimpleNamespace
from typing import Any

import pytest

from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import (
    resolve_ar_compile_mode,
    validate_sim_parallel_config,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("source", ["direct", "model_config", "custom_pipeline_args"])
@pytest.mark.parametrize("enabled,mode", [(False, "default"), (True, "reduce-overhead")])
def test_cuda_graph_flag_selects_compile_mode(source, enabled, mode):
    options = {"use_cuda_graphs": enabled}
    config = SimpleNamespace(**(options if source == "direct" else {source: options}))
    assert resolve_ar_compile_mode(config) == mode
    assert resolve_ar_compile_mode(SimpleNamespace()) == "default"


@pytest.mark.parametrize("override", [dict(enforce_eager=True), dict(diffusion_compile_granularity="full")])
def test_native_capture_rejects_inactive_or_unvalidated_compilation(override):
    with pytest.raises(ValueError):
        resolve_ar_compile_mode(SimpleNamespace(model_config={"use_cuda_graphs": True}, **override))


@pytest.mark.parametrize("value", ["true", "false", 0, 1])
def test_cuda_graph_flag_requires_boolean(value):
    with pytest.raises(ValueError, match="must be a boolean"):
        resolve_ar_compile_mode(SimpleNamespace(model_config={"use_cuda_graphs": value}))


@pytest.mark.parametrize("mode", ["default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"])
def test_old_mode_selector_fails_with_migration_message(mode):
    with pytest.raises(ValueError, match="replaced by the boolean use_cuda_graphs"):
        resolve_ar_compile_mode(SimpleNamespace(model_config={"ar_compile_mode": mode}))


@pytest.mark.parametrize("source", ["direct", "model_config", "custom_pipeline_args"])
def test_hsdp_rejects_cuda_graphs(source):
    options: dict[str, Any] = {"use_cuda_graphs": True}
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(use_hsdp=True, hsdp_shard_size=2),
        **(options if source == "direct" else {source: options}),
    )
    with pytest.raises(ValueError, match="HSDP is currently incompatible with CUDA graphs"):
        resolve_ar_compile_mode(config)


@pytest.mark.parametrize("options", [{}, {"use_cuda_graphs": False}])
def test_hsdp_allows_compilation_without_cuda_graphs(options):
    config = SimpleNamespace(
        model_config=options,
        parallel_config=SimpleNamespace(use_hsdp=True, hsdp_shard_size=2),
    )
    assert resolve_ar_compile_mode(config) == "default"


def test_tensor_parallel_allows_cuda_graphs():
    config = SimpleNamespace(
        model_config={"use_cuda_graphs": True},
        parallel_config=SimpleNamespace(use_hsdp=False, tensor_parallel_size=2),
    )
    assert resolve_ar_compile_mode(config) == "reduce-overhead"


@pytest.mark.parametrize(
    "field",
    [
        "sequence_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "text_encoder_tp_size",
        "vae_patch_parallel_size",
    ],
)
def test_unsupported_parallel_paths_fail_before_loading_weights(field):
    config = SimpleNamespace(parallel_config=SimpleNamespace(**{field: 2}))
    with pytest.raises(ValueError, match=field):
        validate_sim_parallel_config(config)


@pytest.mark.parametrize(
    "parallel",
    [
        dict(tensor_parallel_size=2),
        dict(tensor_parallel_size=4),
        dict(use_hsdp=True, hsdp_shard_size=2),
        dict(data_parallel_size=None),
    ],
)
def test_model_parallel_paths_are_not_blanket_disabled(parallel):
    validate_sim_parallel_config(SimpleNamespace(parallel_config=SimpleNamespace(**parallel)))


def test_dense_model_does_not_claim_expert_parallelism():
    with pytest.raises(ValueError, match="no experts"):
        validate_sim_parallel_config(SimpleNamespace(parallel_config=SimpleNamespace(enable_expert_parallel=True)))


@pytest.mark.parametrize("frame_causal", [False, True])
def test_committed_history_survives_reuse_of_graph_output_storage(frame_causal):
    import torch

    from tests.diffusion.models.cosmos3_nano_sim_bimanual.test_frame_causal_attention import make_state

    state = make_state()
    cache = state.kv_cache
    try:
        for value in (1, 3):
            ctx = state.get_kv_caches("main", cache.block_size, commit_current=True, frame_causal=frame_causal)[
                0
            ].forward_ctx
            scratch = ctx.current_video_block_ids[0]
            for layer in range(state.num_layers):
                cache.key_cache(layer)[scratch].fill_(value)
                cache.value_cache(layer)[scratch].fill_(value + 1)
            state.commit_paged_context("main")
            # Simulate graph replay overwriting every reusable current buffer.
            for layer in range(state.num_layers):
                cache.key_cache(layer)[scratch].fill_(99)
                cache.value_cache(layer)[scratch].fill_(99)
            pages = cache.window_block_ids(state.adapter("main"))
            for layer in range(state.num_layers):
                expected = torch.tensor([1] if value == 1 else [1, 3], dtype=torch.float32)
                torch.testing.assert_close(cache.key_cache(layer)[pages, 0, 0, 0], expected)
                torch.testing.assert_close(cache.value_cache(layer)[pages, 0, 0, 0], expected + 1)
    finally:
        state.close()
