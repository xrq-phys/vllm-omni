# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Reference cache capacity, checkpoint defaults, and session separation."""

from types import SimpleNamespace

import pytest
from test_action_schema import manifest

from tests.diffusion.models.cosmos3_nano_sim_bimanual.test_frame_causal_attention import make_state, publish
from vllm_omni.diffusion.models.cosmos3_nano_sim_bimanual.config import (
    COSMOS3_NANO_SIM_BIMANUAL_ARTIFACT_FIELDS,
    Cosmos3NanoSimBimanualManifest,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def config(**overrides):
    original = manifest()
    artifact = {
        key: getattr(original, key)
        for key in COSMOS3_NANO_SIM_BIMANUAL_ARTIFACT_FIELDS - {"conditioning", "fixed_step_sampler_config"}
    }
    artifact["conditioning"] = original.conditioning.model_dump(mode="json")
    artifact["fixed_step_sampler_config"] = {
        "sample_type": original.sample_type,
        "t_list": list(original.t_list),
        "num_train_timesteps": original.num_train_timesteps,
    }
    return SimpleNamespace(tf_model_config={"cosmos3_nano_sim_bimanual": artifact}, model_config=overrides)


@pytest.mark.parametrize(("cache_size", "sinks"), [(16, 0), (64, 8), (128, 32)])
def test_override_preserves_checkpoint_and_separates_sessions(cache_size, sinks):
    baseline = Cosmos3NanoSimBimanualManifest.from_od_config(config())
    options = config(kv_cache_inference_size=cache_size, attention_sink_size=sinks)
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(options)
    assert (effective.window_frames, effective.sink_frames) == (cache_size - sinks - 1, sinks)
    assert effective.digest != baseline.digest
    assert effective.chunk_size == baseline.chunk_size and effective.t_list == baseline.t_list
    assert options.tf_model_config["cosmos3_nano_sim_bimanual"]["window_frames"] == baseline.window_frames


def test_retained_history_matches_reference_frame_indices():
    effective = Cosmos3NanoSimBimanualManifest.from_od_config(config(kv_cache_inference_size=64, attention_sink_size=8))
    state = make_state(page=1, window=effective.window_frames, sink=effective.sink_frames)
    try:
        for frame in range(226):
            ctx = state.get_kv_caches("main", 1)[0].forward_ctx
            pages, _ = ctx.video_block_table()
            history = [
                int(state.kv_cache.key_cache(0)[p, 0, 0, 0]) for p in pages if p not in ctx.current_video_block_ids
            ]
            assert history == list(range(min(frame, 8))) + list(range(max(8, frame - 55), frame))
            publish(state, [frame])
    finally:
        state.close()


@pytest.mark.parametrize(
    "overrides",
    [
        {"attention_sink_size": 8},
        {"kv_cache_inference_size": True},
        {"kv_cache_inference_size": 1},
        {"kv_cache_inference_size": 64, "attention_sink_size": -1},
        {"kv_cache_inference_size": 64, "attention_sink_size": 63},
    ],
)
def test_invalid_override_rejected(overrides):
    with pytest.raises(ValueError):
        Cosmos3NanoSimBimanualManifest.from_od_config(config(**overrides))
