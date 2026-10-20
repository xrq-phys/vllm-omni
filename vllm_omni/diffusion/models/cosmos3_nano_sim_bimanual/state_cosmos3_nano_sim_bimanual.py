# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Non-KV per-session state and fail-closed request fingerprinting."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass(frozen=True)
class Cosmos3NanoSimBimanualSessionFingerprint:
    prompt_hash: str
    real_text_kv_lengths: tuple[tuple[str, int], ...]
    height: int
    width: int
    fps: float
    domain_id: int
    embodiment: str
    action_contract_sha256: str
    checkpoint_id: str
    manifest_id: str
    sampler_id: str
    action_space: str = "normalized_unified_v1"
    inference_id: str = ""

    def __post_init__(self) -> None:
        if not self.prompt_hash:
            raise ValueError("Cosmos3-Nano-Sim-Bimanual session fingerprint requires prompt_hash")
        if self.height <= 0 or self.width <= 0 or self.fps <= 0:
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual fingerprint resolution/FPS must be "
                f"positive, got {self.height}x{self.width}@{self.fps}"
            )
        if self.domain_id < 0:
            raise ValueError(f"Cosmos3-Nano-Sim-Bimanual domain_id must be non-negative, got {self.domain_id}")
        if not self.embodiment:
            raise ValueError("Cosmos3-Nano-Sim-Bimanual session fingerprint requires an embodiment")
        if not self.action_contract_sha256:
            raise ValueError("Cosmos3-Nano-Sim-Bimanual session fingerprint requires action_contract_sha256")
        if not self.real_text_kv_lengths:
            raise ValueError("Cosmos3-Nano-Sim-Bimanual fingerprint requires at least one text KV branch")
        if any(length <= 0 for _, length in self.real_text_kv_lengths):
            raise ValueError(
                f"Cosmos3-Nano-Sim-Bimanual real text KV lengths must be positive, got {self.real_text_kv_lengths}"
            )

    def text_length(self, branch: str) -> int:
        try:
            return dict(self.real_text_kv_lengths)[branch]
        except KeyError as exc:
            raise KeyError(f"Cosmos3-Nano-Sim-Bimanual fingerprint has no text length for branch {branch!r}") from exc


@dataclass
class Cosmos3NanoSimBimanualSessionState:
    session_id: str
    fingerprint: Cosmos3NanoSimBimanualSessionFingerprint | None = None
    next_frame_idx: int = 0
    terminal: bool = False
    tick_output_type: str | None = None
    text_kv_by_branch: dict[str, list[tuple[torch.Tensor, torch.Tensor]]] = field(default_factory=dict)
    latents: list[torch.Tensor] = field(default_factory=list)
    vae_decoder_feat_cache: list[Any] | None = None
    vae_decoder_initialized: bool = False
    last_vae_decode_input_frames: int = 0
    max_vae_decode_input_frames: int = 0

    def initialize(
        self,
        fingerprint: Cosmos3NanoSimBimanualSessionFingerprint,
        *,
        next_frame_idx: int = 0,
    ) -> None:
        if self.fingerprint is not None:
            raise RuntimeError(
                f"Cosmos3-Nano-Sim-Bimanual session {self.session_id!r} is already initialized; session reset required"
            )
        if next_frame_idx < 0:
            raise ValueError(f"Cosmos3-Nano-Sim-Bimanual next_frame_idx must be non-negative, got {next_frame_idx}")
        self.fingerprint = fingerprint
        self.next_frame_idx = int(next_frame_idx)

    def validate_request(
        self,
        fingerprint: Cosmos3NanoSimBimanualSessionFingerprint,
        *,
        frame_idx: int,
    ) -> None:
        if self.fingerprint is None:
            raise RuntimeError(
                f"Cosmos3-Nano-Sim-Bimanual session {self.session_id!r} is not initialized; session reset required"
            )
        if self.terminal:
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual session "
                f"{self.session_id!r} already completed a full rollout; session reset required"
            )
        if fingerprint != self.fingerprint:
            changed = [
                field_name
                for field_name in self.fingerprint.__dataclass_fields__
                if getattr(self.fingerprint, field_name) != getattr(fingerprint, field_name)
            ]
            raise ValueError(
                f"Cosmos3-Nano-Sim-Bimanual session conditioning changed ({', '.join(changed)}); session reset required"
            )
        if frame_idx != self.next_frame_idx:
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual request is out of order: "
                f"expected latent frame {self.next_frame_idx}, got {frame_idx}; session reset required"
            )

    def append_chunk(
        self,
        chunk: torch.Tensor,
        *,
        frame_start: int,
        retain_latent: bool = True,
    ) -> None:
        if chunk.ndim != 5 or chunk.shape[0] != 1:
            raise ValueError(
                f"Cosmos3-Nano-Sim-Bimanual session chunks must have shape [1,C,T,H,W], got {tuple(chunk.shape)}"
            )
        if frame_start != self.next_frame_idx:
            raise ValueError(
                "Cosmos3-Nano-Sim-Bimanual cannot append an out-of-order chunk: "
                f"expected {self.next_frame_idx}, got {frame_start}; session reset required"
            )
        if retain_latent:
            self.latents.append(chunk.detach())
        self.next_frame_idx += int(chunk.shape[2])

    def record_incremental_decode(
        self,
        *,
        input_frames: int,
        feature_cache: list[Any],
    ) -> None:
        if input_frames <= 0:
            raise ValueError(f"Cosmos3-Nano-Sim-Bimanual decode input_frames must be positive, got {input_frames}")
        self.vae_decoder_feat_cache = feature_cache
        self.vae_decoder_initialized = True
        self.last_vae_decode_input_frames = int(input_frames)
        self.max_vae_decode_input_frames = max(self.max_vae_decode_input_frames, int(input_frames))

    def reset(self) -> None:
        self.fingerprint = None
        self.next_frame_idx = 0
        self.terminal = False
        self.tick_output_type = None
        self.text_kv_by_branch.clear()
        self.latents.clear()
        self.vae_decoder_feat_cache = None
        self.vae_decoder_initialized = False
        self.last_vae_decode_input_frames = 0
        self.max_vae_decode_input_frames = 0

    @property
    def accumulated_latents(self) -> torch.Tensor | None:
        if not self.latents:
            return None
        return torch.cat(self.latents, dim=2)
