# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from dataclasses import dataclass

import torch
from vllm.logger import init_logger

from vllm_omni.diffusion.attention.backends.abstract import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
)

logger = init_logger(__name__)

try:
    from flashinfer.prefill import BatchPrefillWithRaggedKVCacheWrapper

    HAS_FLASHINFER = True
except Exception as e:
    HAS_FLASHINFER = False
    logger.warning(
        "FlashInfer is unavailable; FLASHINFER_ATTN backend will not work. Reason: %s",
        e,
    )


class FlashInferAttentionBackend(AttentionBackend):
    accept_output_buffer: bool = True

    @classmethod
    def supports_attention_mask(cls) -> bool:
        # FlashInfer's ragged prefill wrapper accepts a flattened boolean
        # ``custom_mask`` (``True`` = keep) for non-causal attention.
        return True

    @staticmethod
    def get_supported_head_sizes() -> list[int]:
        # FlashInfer dense prefill is well-tested for these head_dims on
        # Ampere/Hopper/Blackwell. Covers the dominant diffusion DiT shapes
        # (SD3 = 64, Flux/HV/Wan = 128, joint-attn = 256).
        return [64, 128, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASHINFER_ATTN"

    @staticmethod
    def get_impl_cls() -> type["FlashInferAttentionImpl"]:
        return FlashInferAttentionImpl


class FlashInferAttentionImpl(AttentionImpl):
    _QK_DTYPES = {torch.float16, torch.bfloat16}
    _VO_DTYPES = {torch.float16, torch.bfloat16, torch.float8_e4m3fn}

    @dataclass(frozen=True)
    class _WrapperPlanKey:
        batch_size: int
        qo_len: int
        kv_len: int
        num_q_heads: int
        num_kv_heads: int
        head_dim_qk: int
        head_dim_k: int
        head_dim_vo: int
        q_dtype: torch.dtype
        k_dtype: torch.dtype
        v_dtype: torch.dtype
        causal: bool
        softmax_scale: float
        has_custom_mask: bool

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        softmax_scale: float,
        causal: bool = False,
        num_kv_heads: int | None = None,
        prefix: str = "",
        backend_kwargs: dict | None = None,
        **extra_impl_args,
    ) -> None:
        self.causal = causal
        self.softmax_scale = softmax_scale
        backend_kwargs = backend_kwargs or {}
        self.dtype_qk = self._check_dtype(backend_kwargs.get("dtype_qk"), "dtype_qk", self._QK_DTYPES)
        self.dtype_vo = self._check_dtype(backend_kwargs.get("dtype_vo"), "dtype_vo", self._VO_DTYPES)
        requested_backend = backend_kwargs.get("flashinfer_backend", "auto")
        if not HAS_FLASHINFER:
            raise ImportError("FLASHINFER_ATTN backend requires flashinfer")

        self.device = torch.device("cuda", torch.accelerator.current_device_index())
        self.flashinfer_backend = self._select_backend(requested_backend, self.device)
        workspace_size = 0 if self.flashinfer_backend == "cute-dsl" else 128 * 1024 * 1024
        self._workspace = torch.empty(workspace_size, device=self.device, dtype=torch.uint8)
        self._wrapper = BatchPrefillWithRaggedKVCacheWrapper(
            self._workspace,
            kv_layout="NHD",
            backend=self.flashinfer_backend,
        )
        self._qo_indptr: torch.Tensor | None = None
        self._kv_indptr: torch.Tensor | None = None
        self._plan_key: FlashInferAttentionImpl._WrapperPlanKey | None = None

        logger.info_once(
            "FLASHINFER_ATTN initialized backend=%s on %s.",
            self.flashinfer_backend,
            self.device,
        )
        if self.dtype_qk is not None or self.dtype_vo is not None:
            logger.info_once(
                "FLASHINFER_ATTN dtype override: Q/K=%s, V=%s.",
                self.dtype_qk,
                self.dtype_vo,
            )

    @classmethod
    def _check_dtype(
        cls,
        dtype: torch.dtype | None,
        option_name: str,
        allowed: set[torch.dtype],
    ) -> torch.dtype | None:
        if dtype is None:
            return None
        if dtype not in allowed:
            choices = ", ".join(sorted(str(item) for item in allowed))
            raise ValueError(f"Unsupported {option_name}={dtype}; expected one of: {choices}")
        return dtype

    @staticmethod
    def _pack_mask_for_flashinfer(
        attn_mask: torch.Tensor, batch_size: int, qo_len: int, kv_len: int
    ) -> torch.Tensor | None:
        """Convert a diffusion-style attn_mask into the boolean form
        FlashInfer's ``custom_mask`` expects (``True`` = keep).

        Returns either ``(qo_len, kv_len)`` (shared across the batch) or
        ``(batch_size, qo_len, kv_len)`` (per-sample), or ``None`` when the
        mask is all-keep (elide). Only boolean masks are handled here;
        additive/float masks raise ``ValueError`` so the caller falls back to
        SDPA, which applies them with the correct softmax semantics. Shape
        mismatches also raise ``ValueError``.
        """
        mask = attn_mask
        if mask.dtype != torch.bool:
            # Additive masks (0 / -inf / -1e4 / finfo.min) cannot be faithfully
            # reduced to a boolean keep-mask here; SDPA handles them correctly.
            raise ValueError(
                f"non-boolean attn_mask (dtype={mask.dtype}); FlashInfer custom_mask "
                "is boolean-only — deferring to SDPA"
            )
        # Diffusion masks arrive as (qo,kv), (1,1,kv), (B,1,1,kv), (B,1,qo,kv)
        # or (B,H,qo,kv). The mask is identical across heads, so collapse the
        # head dim, but keep a real batch dim — indexing mask[0] would reuse
        # sample 0's padding for every sample (wrong under CFG / mixed lengths).
        if mask.dim() == 4:
            mask = mask[:, 0]  # (B, qo|1, kv)
        if mask.dim() == 3 and mask.shape[0] == 1:
            mask = mask[0]  # (qo|1, kv) — shared across the batch
        try:
            if mask.dim() >= 3:
                mask = mask.broadcast_to((batch_size, qo_len, kv_len))
            else:
                mask = mask.broadcast_to((qo_len, kv_len))
        except RuntimeError as e:
            raise ValueError(
                f"attn_mask shape {tuple(attn_mask.shape)} cannot broadcast to "
                f"(batch={batch_size}, qo_len={qo_len}, kv_len={kv_len})"
            ) from e
        if mask.all():
            return None
        # ``broadcast_to`` returns a non-contiguous view; materialize for the
        # kernel, which reads from GPU memory directly.
        return mask.contiguous()

    def _sdpa_fallback(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None,
    ) -> torch.Tensor:
        from vllm_omni.diffusion.attention.backends.sdpa import SDPAImpl

        impl = SDPAImpl(
            num_heads=query.shape[2],
            head_size=query.shape[3],
            softmax_scale=self.softmax_scale,
            causal=self.causal,
        )
        return impl.forward_cuda(query, key, value, attn_metadata)

    @staticmethod
    def _select_backend(requested_backend: str, device: torch.device) -> str:
        if requested_backend != "auto":
            return requested_backend
        major, _minor = torch.cuda.get_device_capability(device)
        if major >= 10:
            return "cute-dsl"
        if major >= 9:
            return "fa3"
        return "fa2"

    @torch.compiler.disable
    def _plan_wrapper(
        self,
        key: _WrapperPlanKey,
        flat_mask: torch.Tensor | None,
    ) -> None:
        self._wrapper.plan(
            self._qo_indptr,
            self._kv_indptr,
            key.num_q_heads,
            key.num_kv_heads,
            key.head_dim_qk,
            head_dim_vo=key.head_dim_vo,
            custom_mask=flat_mask,
            causal=key.causal,
            sm_scale=key.softmax_scale,
            q_data_type=key.q_dtype,
            # CuTe FMHA accepts K in dtype_qk and V in dtype_vo independently.
            kv_data_type=key.k_dtype,
            o_data_type=key.q_dtype,
        )

    def _ensure_plan(
        self,
        key: _WrapperPlanKey,
        flat_mask: torch.Tensor | None,
    ) -> None:
        key_changed = key != self._plan_key
        if not key_changed and flat_mask is None:
            return

        if key_changed:
            self._qo_indptr = torch.arange(
                0,
                (key.batch_size + 1) * key.qo_len,
                key.qo_len,
                device=self.device,
                dtype=torch.int32,
            )
            self._kv_indptr = torch.arange(
                0,
                (key.batch_size + 1) * key.kv_len,
                key.kv_len,
                device=self.device,
                dtype=torch.int32,
            )

        # A custom mask's values may change without its shape changing, so it
        # must be copied into the wrapper on every masked invocation.
        self._plan_wrapper(key, flat_mask)
        self._plan_key = key

    def _run_batch_prefill(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        custom_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        """Flatten a dense batch and invoke FlashInfer's ragged wrapper."""
        if query.device != self.device or key.device != self.device or value.device != self.device:
            raise ValueError(
                "FLASHINFER_ATTN inputs must remain on the layer initialization "
                f"device {self.device}; got Q={query.device}, K={key.device}, V={value.device}"
            )

        batch_size, qo_len, num_q_heads, head_dim_qk = query.shape
        kv_len = key.shape[1]
        num_kv_heads = key.shape[2]
        head_dim_k = key.shape[3]
        head_dim_vo = value.shape[3]

        q = query.reshape(batch_size * qo_len, num_q_heads, head_dim_qk)
        k = key.reshape(batch_size * kv_len, num_kv_heads, head_dim_k)
        v = value.reshape(batch_size * kv_len, num_kv_heads, head_dim_vo)
        if self.dtype_qk is not None:
            q = q.to(self.dtype_qk)
            k = k.to(self.dtype_qk)
        if self.dtype_vo is not None:
            v = v.to(self.dtype_vo)

        flat_mask = None
        if custom_mask is not None:
            if custom_mask.dim() == 2:
                custom_mask = custom_mask.unsqueeze(0).expand(batch_size, -1, -1)
            flat_mask = custom_mask.contiguous().view(-1)

        self._ensure_plan(
            FlashInferAttentionImpl._WrapperPlanKey(
                batch_size=batch_size,
                qo_len=qo_len,
                kv_len=kv_len,
                num_q_heads=num_q_heads,
                num_kv_heads=num_kv_heads,
                head_dim_qk=head_dim_qk,
                head_dim_k=head_dim_k,
                head_dim_vo=head_dim_vo,
                q_dtype=q.dtype,
                k_dtype=k.dtype,
                v_dtype=v.dtype,
                causal=self.causal,
                softmax_scale=self.softmax_scale,
                has_custom_mask=flat_mask is not None,
            ),
            flat_mask,
        )
        out = self._wrapper.run(q, k, v)
        out = out.reshape(batch_size, qo_len, num_q_heads, head_dim_vo)
        return out.to(query.dtype) if out.dtype != query.dtype else out

    def forward_cuda(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_metadata: AttentionMetadata | None = None,
    ) -> torch.Tensor:
        if not HAS_FLASHINFER:
            raise ImportError(
                "FLASHINFER_ATTN backend requires flashinfer. "
                "Install it or set DIFFUSION_ATTENTION_BACKEND to another backend."
            )

        # Try the custom-mask path; if it cannot be represented by FlashInfer's
        # ragged wrapper, fall back to SDPA rather than changing semantics.
        batch_size = query.shape[0]

        custom_mask = None
        if attn_metadata is not None and attn_metadata.attn_mask is not None:
            try:
                custom_mask = self._pack_mask_for_flashinfer(
                    attn_metadata.attn_mask,
                    batch_size=batch_size,
                    qo_len=query.shape[1],
                    kv_len=key.shape[1],
                )
            except ValueError as e:
                logger.debug("Falling back to SDPA for mask path: %s", e)
                return self._sdpa_fallback(query, key, value, attn_metadata)
            # FlashInfer cannot combine causal masking with a custom_mask; rather
            # than silently dropping the explicit mask (diverging from SDPA), let
            # SDPA handle the causal+mask case correctly.
            if custom_mask is not None and self.causal:
                logger.debug("causal=True with explicit attn_mask; deferring to SDPA")
                return self._sdpa_fallback(query, key, value, attn_metadata)

        if custom_mask is not None and self.flashinfer_backend == "cute-dsl":
            logger.debug("CuTe DSL does not support custom masks; deferring to SDPA")
            return self._sdpa_fallback(query, key, value, attn_metadata)

        return self._run_batch_prefill(query, key, value, custom_mask)
