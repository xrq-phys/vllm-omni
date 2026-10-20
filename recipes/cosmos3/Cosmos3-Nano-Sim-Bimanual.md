# Cosmos3-Nano-Sim-Bimanual

The distilled action-conditioned world model uses the checkpoint's sampling
schedule and normalized unified action schema. Install the CUDA dependencies
required by this branch:

```bash
pip install -e .
```

Frame-page attention requires a backend that supports the selected page size,
physical page tables, and explicit sequence lengths. The cache layout and
attention visibility contract are independent of the backend implementation.

The model uses the shared AR-Diffusion attention router in
`experimental/ar_diffusion/kv_cache/paged_attention.py`. On CUDA, the router
selects a FlashAttention version through vLLM and calls
`vllm.vllm_flash_attn.flash_attn_varlen_func`. A separate `flash-attn-4`
installation is not required by the model. The model also reuses the shared
`paged_write_attn` custom op. Its local `frame_causal_attention.py` supplies
frame visibility metadata and delayed clean-frame publication through the
upstream cache manager.

The configured autoregressive attention backend must support the complete frame
page. By default, we select FlashAttention 4 to support arbitrary page sizes.
Unsupported page sizes raise an error. The model features multi-frame causal
structure, so routes that gather K/V or use reference attention are rejected.

## Deployment

Start from [the deployment configuration](../../vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml)
or [the bounded-window configuration](../../vllm_omni/deploy/cosmos3_nano_sim_bimanual_windowed.yaml).
Keep `engine_backend: vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine`
and `max_num_seqs: 1`. For regional compilation, set these stage options:

```yaml
enforce_eager: false
diffusion_compile_granularity: regional
diffusion_compile_dynamic: true
```

`model_config.use_cuda_graphs: true` selects Inductor CUDA graphs and requires
regional compilation. The paired single-GPU profiles use compilation with
CUDA graphs disabled. Tensor parallelism and HSDP code paths are available;
this validation covers one GPU. HSDP requires CUDA graphs disabled.

## Actions and geometry

Provide `N-1` action rows for `N` output video frames, where `(N-1) % 4 == 0`.
Set `action_space="normalized_unified_v1"`; each row contains 59 normalized
values. The runtime validates domain IDs and pads rows to the model's 64D
width. Use `domain_name` or `domain_id` from the checkpoint action schema.
Full rollouts normally use `reset=true` and `close_session=true`.

Pass the original reference image. With explicit `height` and `width`, the
runtime validates the requested patch grid and `max_pixels`. Without them,
preprocessing selects an aspect-matched canvas, downsizes if needed, pads for
VAE encoding, and crops the latent to aligned content dimensions. Resolution
changes within a session require a reset and rebuild the pool after admission.

## KV pages and attention visibility

At 720×1280, the latent patch grid contains **920 video tokens**. Four action
tokens make a frame page **924 tokens**, with no divisibility-by-16 requirement.
Every attention row is logically:

```text
[protected sink frame pages | recent frame pages | current frame pages | text pages]
```

Each frame starts at offset zero. Text occupies separate pages of the same
size; sequence lengths exclude unused text-page capacity. The attention backend
reads the block table directly. Valid frames and text have no intervening padding,
and history is never gathered into contiguous K/V.

The upstream allocator owns pages and session lifetime. After all layers
finish a clean forward, newly computed frame pages are published from bounded
scratch storage. Upstream whole-frame eviction reclaims old pages and compacts
host metadata. Only new frames are copied during publication. There is no
16-token paging, private tail packing, page rotation, or model-specific history
eviction implementation.

`kv_cache_inference_size` includes the sinks and current frame;
`attention_sink_size` sets protected initial frames. For example, a capacity
of 4 with 1 sink retains 2 recent past frames. Both settings use latent-frame
units. Changing visibility can change generated output. Without overrides,
the checkpoint history policy applies. A full-history checkpoint sizes the
cache for the admitted rollout and necessarily retains all its frames.

Typed tick requests retain model and KV state across calls, enforce session
fingerprints and chunk order, and return transport-safe correlation metadata.
Invalid admission preserves existing history; execution failure releases it.
Fresh full-video requests decode chunks to CPU to bound VAE GPU history.
`clean_commit_mode: batched` refreshes clean frames together, and
`overlap_vae_decode: true` overlaps supported full-rollout decoding and refresh.

## Validation

`tests/diffusion/models/cosmos3_nano_sim_bimanual/test_frame_causal_attention.py` covers
257-frame bounded reclamation, exact window/text metadata, real upstream cache
binding, and a shared-router GPU comparison against a dense numerical oracle.
