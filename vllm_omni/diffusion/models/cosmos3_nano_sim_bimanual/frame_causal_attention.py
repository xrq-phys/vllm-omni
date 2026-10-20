# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Frame-causal attention contexts using upstream AR-Diffusion allocation and lifetime.

Only current frames are staged. Once every layer finishes a clean forward,
publish those pages through the upstream allocator, which reclaims whole old
frames and compacts metadata. History is never gathered or copied for attention.
"""

from dataclasses import dataclass

import torch

from vllm_omni.experimental.ar_diffusion.kv_cache.paged_attention import (
    ARDiffusionPagedLayerInputs,
    _layer_idx_tensor,
)


def _device(value, device, dtype=torch.int32):
    tensor = torch.tensor(value, dtype=dtype)
    if torch.device(device).type == "cuda":
        return tensor.pin_memory().to(device, non_blocking=True)
    return tensor.to(device)


class FrameCausalAttentionForwardContext:
    def __init__(self, state, branch, seq_len, commit_current, frame_causal):
        self.state = state
        self.kv_cache = state.kv_cache
        self.kv_branch = branch
        self.adapter = state.adapter(branch)
        self.seq_len = seq_len
        self.commit_current = commit_current
        self.frame_causal = frame_causal
        self.block_size = self.kv_cache.block_size
        self.chunk_size = self.kv_cache.spec.chunk_size
        if self.block_size != self.chunk_size or seq_len <= 0 or seq_len % self.block_size:
            raise ValueError("Cosmos3 Sim requires exactly one complete latent frame per page")
        self.num_frames = seq_len // self.block_size
        if self.num_frames > self.kv_cache.frames_per_block:
            raise ValueError("Current frame span exceeds the declared scratch frame capacity")
        self.current_video_block_ids = self.kv_cache.scratch_block_ids(branch, 0, self.num_frames)
        self._history_tokens = self.adapter.num_computed_tokens
        self._prepared = False

    def video_block_table(self, device=None, seq_len=None):
        """Return physical pages in sink/recent/current order, entirely on host."""
        history_frames = self._history_tokens // self.block_size
        current_frames = self.num_frames if seq_len is None else seq_len // self.block_size
        end = history_frames + current_frames
        sink = min(self.kv_cache.spec.sink_chunks, end)
        recent = self.kv_cache.spec.window_chunks
        start = max(sink, end - recent - (1 if self.frame_causal else current_frames))
        indices = [*range(sink), *range(start, end)]
        old = [index for index in indices if index < history_frames]
        pages = dict(zip(old, self.kv_cache.block_ids_at(self.adapter, old)))
        blocks = [
            pages[index] if index < history_frames else self.current_video_block_ids[index - history_frames]
            for index in indices
        ]
        if self.kv_cache.null_block_id in blocks:
            raise RuntimeError("An evicted frame was selected for Cosmos3 Sim attention")
        return blocks, len(blocks) * self.block_size

    def prepare(self, device, action_len, query_len):
        if self._prepared:
            return
        if query_len != self.seq_len or action_len <= 0:
            raise ValueError("Invalid frame-page query or text length")
        size = self.block_size
        self.action_scratch_block_ids = self.kv_cache.scratch_block_ids(
            self.kv_branch,
            self.kv_cache.frames_per_block,
            (action_len + size - 1) // size,
        )
        self.video_slots = _device(
            [page * size + i for page in self.current_video_block_ids for i in range(size)], device, torch.int64
        )
        self.action_slots = _device(
            [self.action_scratch_block_ids[i // size] * size + i % size for i in range(action_len)], device, torch.int64
        )
        views = (
            [self.video_block_table(device, end) for end in range(size, self.seq_len + 1, size)]
            if self.frame_causal
            else [self.video_block_table(device)]
        )
        capacity = (
            self.kv_cache.spec.sink_chunks
            + self.kv_cache.spec.window_chunks
            + self.kv_cache.frames_per_block
            + (action_len + size - 1) // size
        )
        rows = [pages + self.action_scratch_block_ids for pages, _ in views]
        self.block_table = _device(
            [row + [self.kv_cache.null_block_id] * (capacity - len(row)) for row in rows], device
        )
        self.seq_lens = _device([length + action_len for _, length in views], device)
        self.max_query_len = size if self.frame_causal else self.seq_len
        self.query_start_loc = _device(list(range(0, self.seq_len + 1, self.max_query_len)), device)
        self.max_seq_len = capacity * size
        self._prepared = True

    def layer_inputs(self, layer_idx):
        if not self._prepared:
            raise RuntimeError("Frame-page metadata must be prepared before compiled blocks execute")
        return ARDiffusionPagedLayerInputs(
            layer_idx=_layer_idx_tensor(layer_idx),
            key_pool=self.kv_cache.key_cache(layer_idx).flatten(0, 1),
            value_pool=self.kv_cache.value_cache(layer_idx).flatten(0, 1),
            block_size=self.block_size,
            seq_len=self.seq_len,
            video_slots=self.video_slots,
            action_slots=self.action_slots,
            block_table=self.block_table,
            query_start_loc=self.query_start_loc,
            seq_lens=self.seq_lens,
            max_query_len=self.max_query_len,
            max_seq_len=self.max_seq_len,
            require_paged=True,
        )


@dataclass
class FrameCausalAttentionLayerContext:
    layer_idx: int
    forward_ctx: FrameCausalAttentionForwardContext

    def to_layer_inputs(self):
        return self.forward_ctx.layer_inputs(self.layer_idx)


class FrameCausalAttentionState:
    """Model-local attention adapter; upstream retains ownership of all pages."""

    def __init__(self, state):
        self.state = state
        self.kv_cache = state.kv_cache
        self.pending = {}
        if self.kv_cache.block_size != self.kv_cache.spec.chunk_size:
            raise ValueError("Cosmos3 Sim only supports one page per latent frame")
        if self.kv_cache.history_staging:
            raise ValueError("Frame-page attention does not support gathered history staging")

    def __getattr__(self, name):
        return getattr(self.state, name)

    def get_kv_caches(self, branch, seq_len, commit_current=False, *, extra_visible_tokens=None, frame_causal=False):
        context = FrameCausalAttentionForwardContext(self, branch, seq_len, commit_current, frame_causal)
        self.pending[branch] = context
        return [FrameCausalAttentionLayerContext(index, context) for index in range(self.num_layers)]

    def commit_paged_context(self, branch):
        context = self.pending.pop(branch)
        if not context.commit_current:
            return
        # Every layer has finished reading old history before upstream may
        # evict it. Copy only newly computed frames; never compact/gather K/V.
        for source in context.current_video_block_ids:
            table = self.kv_cache.allocate_chunk(context.adapter)
            destination = table[context.adapter.num_computed_tokens // context.block_size]
            for layer in range(self.num_layers):
                self.kv_cache.key_cache(layer)[destination].copy_(self.kv_cache.key_cache(layer)[source])
                self.kv_cache.value_cache(layer)[destination].copy_(self.kv_cache.value_cache(layer)[source])
            self.kv_cache.commit_chunk(context.adapter)
            self.state._committed[branch] += context.block_size
