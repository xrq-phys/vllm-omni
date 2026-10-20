# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Request geometry admission with upstream allocation and backend page sizes."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from tests.diffusion.ar_diffusion.test_capability_runner import CapablePipeline, lingbot_like_spec, make_runner
from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner
from vllm_omni.experimental.ar_diffusion.capability import (
    ARDiffusionRequestKVSpec,
    ARDiffusionRequestRejectedError,
)

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


class DynamicPipeline(CapablePipeline):
    def __init__(self):
        super().__init__(replace(lingbot_like_spec(capacity=1), tokens_per_frame=5, supported_block_sizes=(5,)))

    def ar_diffusion_default_request_spec(self):
        return ARDiffusionRequestKVSpec(self.spec, 5)

    def ar_diffusion_request_spec(self, request):
        page = request.sampling_params.page
        if page <= 0:
            raise ValueError("invalid geometry")
        return ARDiffusionRequestKVSpec(replace(self.spec, tokens_per_frame=page, supported_block_sizes=(page,)), page)

    def ar_diffusion_worst_case_request_specs(self):
        yield ARDiffusionRequestKVSpec(replace(self.spec, tokens_per_frame=7, supported_block_sizes=(7,)), 7)

    def validate_ar_diffusion_effective_spec(self, spec):
        assert spec.frames_per_block == self.spec.frames_per_block


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.setattr(
        "vllm_omni.experimental.ar_diffusion.runner.current_omni_platform",
        SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
    )
    pipeline = DynamicPipeline()
    value = make_runner(pipeline)
    value.release_captured_graphs = lambda: None
    monkeypatch.setattr(DiffusionModelRunner, "execute_model", lambda self, req, **kw: self.pipeline.bound_state)
    yield value
    for session in tuple(value._sessions):
        value.close_session(session)


def request(page=5, **extra):
    return SimpleNamespace(sampling_params=SimpleNamespace(page=page, extra_args={"session_id": "world", **extra}))


def test_non_aligned_frame_page_and_reset_geometry_rebuild(runner):
    first = runner.execute_model(request())
    old_cache = runner.kv_cache
    assert old_cache.block_size == 5
    with pytest.raises(ARDiffusionRequestRejectedError, match="reset is required"):
        runner.execute_model(request(7))
    assert runner._sessions["world"] is first
    assert runner.kv_cache is old_cache
    second = runner.execute_model(request(7, reset=True))
    assert second is not first
    assert runner.kv_cache is not old_cache
    assert runner.kv_cache.block_size == 7
    assert runner._session_geometry_keys == {"world": 7}


def test_invalid_geometry_and_failed_capacity_admission_preserve_session(runner, monkeypatch):
    state = runner.execute_model(request())
    cache = runner.kv_cache
    with pytest.raises(ARDiffusionRequestRejectedError, match="invalid geometry"):
        runner.execute_model(request(0, reset=True))
    original = runner._build_kv_cache

    def no_capacity(spec, config, budget, device):
        if spec.tokens_per_frame == 7:
            raise ValueError("insufficient capacity")
        return original(spec, config, budget, device)

    monkeypatch.setattr(runner, "_build_kv_cache", no_capacity)
    with pytest.raises(ValueError, match="insufficient capacity"):
        runner.execute_model(request(7, reset=True))
    assert runner._sessions["world"] is state
    assert runner.kv_cache is cache


def test_admission_rejection_retains_history_execution_failure_releases(runner, monkeypatch):
    state = runner.execute_model(request())

    def reject(*args, **kwargs):
        raise ARDiffusionRequestRejectedError("unchanged input required")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", reject)
    with pytest.raises(ARDiffusionRequestRejectedError):
        runner.execute_model(request())
    assert runner._sessions["world"] is state

    def fail(*args, **kwargs):
        raise RuntimeError("failed forward")

    monkeypatch.setattr(DiffusionModelRunner, "execute_model", fail)
    with pytest.raises(RuntimeError, match="failed forward"):
        runner.execute_model(request())
    assert not runner._sessions
    assert not runner._session_geometry_keys
