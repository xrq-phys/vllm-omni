# Cosmos3-Nano-Sim-Bimanual

Cosmos3-Nano-Sim-Bimanual is the causal, action-conditioned Cosmos3 world
model. Use an exported artifact whose transformer configuration contains a
`cosmos3_nano_sim_bimanual` manifest. The runtime reads the checkpoint
identity, sampling schedule, history policy, and action contract from that
manifest.

## Actions

Provide 1 normalized unified action row per video-frame transition:

- `action_space="normalized_unified_v1"`
- 59 values per row
- 1 `domain_id`, or 1 domain ID per action row for mixed-domain sequences

The runtime validates the values and domain IDs, pads each action row with 5
zeros to the model's 64D width, and interleaves 4 action tokens with every
latent video frame. Each domain populates the fields it supports in the shared
59D layout and sets unsupported fields to 0. For example, camera-domain rows
use the first 9 values. Input normalization happens before the request reaches
vLLM-Omni.

Full rollouts provide all action rows and normally use `reset=true` and
`close_session=true`. A request with `num_frames=N` requires `N-1` action rows
and `(N-1) % 4 == 0`. Fresh full-video requests decode causal chunks into CPU
memory to bound GPU VAE history; host video memory grows with rollout length.
Typed AgiBot tick requests use `robot_action.v2` and carry 8 normalized unified
rows for each 2-latent-frame chunk.

## Reference images

Pass the original reference image. When height and width are omitted, the
preprocessor selects an aspect-matched canvas from `resolution` (480 by
default), downsizes only when necessary, pads the bottom and right for VAE
encoding, and crops the encoded latent to the aligned content dimensions. For
example, a 640×360 image produces 640×352 video. Explicit height and width use
the requested output geometry.

## Deployment

Start from
[`cosmos3_nano_sim_bimanual.yaml`](../../vllm_omni/deploy/cosmos3_nano_sim_bimanual.yaml).
The distilled checkpoint artifact contains its denoising schedule. Omit
`num_inference_steps` to use the checkpoint-defined schedule.

For dense attention with inference optimizations, merge these settings into
diffusion stage 0:

```yaml
engine_backend: default
enforce_eager: false
diffusion_compile_granularity: regional
diffusion_compile_dynamic: true
vae_fast_path: channels_last
model_config:
  use_cuda_graphs: true
  clean_commit_mode: batched
  overlap_vae_decode: true
  kv_cache_inference_size: 64
  attention_sink_size: 8
```

- `engine_backend: default` selects dense attention. It used less peak memory
  with comparable latency in the Bimanual benchmark. The supplied deployment
  instead selects `vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine`
  for paged KV storage.
- Regional `torch.compile` optimizes transformer blocks; `use_cuda_graphs: true`
  enables Inductor CUDA graph replay to reduce launch overhead and requires
  `enforce_eager: false`.
- `clean_commit_mode: batched` refreshes clean-frame KV entries together instead
  of 1 frame at a time.
- `overlap_vae_decode: true` overlaps causal VAE decoding with clean-frame refresh
  for fresh full-video requests.
- `vae_fast_path: channels_last` optimizes VAE decoding with channels-last memory
  layout to reduce decode time.

`kv_cache_inference_size` and `attention_sink_size` configure sliding history for
both attention backends without changing the checkpoint manifest. Units are
latent frames: 64 with 8 sinks retains 8 initial frames and up to 55 recent past
frames, reserving 1 position for the current frame. Other valid cache/sink sizes
are supported. Changing the history window can change generated outputs.

With neither override, the checkpoint history policy applies. A checkpoint with
`window_frames: null` and `sink_frames: 0` uses full history and sizes the cache
for the requested rollout.

For typed tick video, the diffusion-stage option
`video_output_transport: {enable_device_postprocess: true}` converts
`output_type="np"` video to uint8 before GPU-to-host transfer. Full-video
rollouts already copy decoded chunks to CPU and do not gain this transfer
optimization. Guardrails, interpolation, and other output formats can retain
the floating-point path; see the
[transport guide](../../docs/user_guide/diffusion/device_side_video_postprocess.md).

Compatible serialized ModelOpt FP8/NVFP4 checkpoints also support
`cosmos3_mixed_precision`. The precision schedule restarts for every denoising
chunk; clean-frame KV refresh uses the final step's precision. For a 4-step
checkpoint, `first_steps=1` and `last_steps=1` select A16/native/native/A16.
This does not enable quantization for BF16 checkpoints. See the
[ModelOpt configuration](../../docs/user_guide/quantization/modelopt.md#cosmos3-mixed-precision-schedule)
for artifact requirements and how to enable the policy.

## Experimental FA4 frame pages

Set `VLLM_OMNI_COSMOS_SIM_FA4_FRAME_PAGES=1` with the AR-Diffusion deployment
to use one latent frame per page. At 720×1280, the patch grid is 23×40:
920 video tokens plus 4 interleaved action tokens gives **924 tokens/page**.
This path calls the installed `flash_attn.cute.flash_attn_varlen_func` with
`page_table` and `seqused_k` directly. Unsupported kernels fail; attention does
not fall back to another backend or gather history into contiguous K/V.

Each row lists protected sink pages, recent frame pages, current frame pages,
then separate text pages of the same size. Every frame and the first text
token start at page offset zero. Sequence lengths exclude the unused suffix
of the final text page. Batched clean refresh uses one row per frame so later
frames stay invisible to earlier queries.

The experiment uses vLLM's ordinary full-attention allocator. Sink/window
visibility is enforced by the attention block table; generated pages remain
allocated until the session ends. Memory grows with history, and a rollout
that exhausts the pool fails. Use short rollouts; bounded-memory reclamation
is not implemented for this path. The contiguous-history staging option is
ignored while the experiment is enabled.

After the checkpoint is cached, run the same 720p AgiBot example with both paths:

```bash
export HF_HOME=/home/scratch.ruqingx_research/cosmos-checkpoints
PYTHON=/home/scratch.ruqingx_research/vllm-omni/.venv/bin/python

VLLM_OMNI_COSMOS_SIM_FA4_FRAME_PAGES=0 "$PYTHON" \
  benchmarks/diffusion/bench_cosmos_sim_frame_pages.py --output-dir "$PWD/var/cosmos-sim-fa4-frame-pages/results"
VLLM_OMNI_COSMOS_SIM_FA4_FRAME_PAGES=1 "$PYTHON" \
  benchmarks/diffusion/bench_cosmos_sim_frame_pages.py --output-dir "$PWD/var/cosmos-sim-fa4-frame-pages/results"
```

The harness saves raw frames, an MP4, deployment settings, and a run report for
each path. It uses 25 pixel frames, seed 2026, the checkpoint's four-step
schedule, one sink frame, and two recent past frames to exercise sliding
visibility. The run report records the installed FA4 version. The GPU op test
also prints CUDA kernel names:

```bash
"$PYTHON" -m pytest -s \
  tests/diffusion/models/cosmos3_nano_sim_bimanual/test_fa4_frame_pages.py
```

### SM103 validation

Validated on SM103 with PyTorch `2.13.0+cu132` and installed `flash-attn-4`
`4.0.0b33`. CUDA profiling records the installed package's
`FlashAttentionForwardSm100` kernel with 924-token pages, 32 query heads,
8 KV heads, and head dimension 128. No FA4 source patch was needed.

For the harness's 25-frame 720p rollout, the generated frames (excluding the
initial condition) have **49.34 dB raw-float RGB PSNR** and **0.998801 RGB8
SSIM** against the existing paged implementation. The inspected frames show
equivalent visual quality in this short sample; the outputs are not
bit-identical. The harness uses the reference prompt's first three sentences
to stay within the checkpoint's 512-token text cache, with guidance 1.0.

This development node also had an unrelated TorchAudio CUDA-version mismatch.
Validation used the temporary process-local helper
`$PWD/var/cosmos-sim-fa4-frame-pages/source/compat/sitecustomize.py` through
`PYTHONPATH=$PWD/var/cosmos-sim-fa4-frame-pages/source/compat:$PWD`;
installed packages were not changed.
Videos, raw frames, the side-by-side comparison, and the detailed report are
under `$PWD/var/cosmos-sim-fa4-frame-pages/results/`. Restored source, environment
settings, and the complete rerun command are under the sibling `source/`
directory; `bash var/cosmos-sim-fa4-frame-pages/source/run.sh` repeats both
rollouts and the comparison. The environment file routes temporary files and
compiler caches into the experiment directory as well.

### Longer rollouts and Nsight Systems profiles

The 25-frame rollout already advances the attention window: temporal compression
is 4, so it contains 7 latent frames. With one sink and two recent past frames,
the clean attention row first excludes latent frame 1 when processing frame 4
(zero-based). The FA4 experiment retains physical pages while advancing visibility.

The local experiment's `source/profile_long.sh` profiles paired 241-frame
rollouts (61 latent frames, 8.03 seconds at 30 fps), each after a 25-frame warmup:

```bash
bash var/cosmos-sim-fa4-frame-pages/source/profile_long.sh
```

Outputs are under `$PWD/var/cosmos-sim-fa4-frame-pages/profile_241/`, including
`baseline.nsys-rep`, `fa4_frame_pages.nsys-rep`, videos, window metadata audits,
and exported statistics. The script uses
`/home/scratch.ruqingx_gpu_1/nsys-2026.3.1.157-x64/bin/nsys` with CUDA, NVTX, and
OS runtime tracing. CPU sampling, context switches, CPU symbol resolution, and
remote debuginfod downloads are disabled. The parent-process `cosmos.generate`
NVTX range brackets generation across the worker processes; `cosmos.warmup`
labels warmup. Model startup and MP4 export are outside the measured range.
`profile_summary.json` uses that timestamp interval to count worker CUDA kernels.

The environment uses a short workspace symlink, `var/c3fa4-tmp`, pointing into
the experiment's `tmp/` directory to keep UNIX socket paths under 107 bytes.

For the profiled 241-frame pair, generation after warmup took 27.51 seconds
(reference) and 27.45 seconds (FA4), with no material difference in these single
observations. The FA4 generation interval contains 6,480 installed FA4 SM100
kernel launches on SM103. Generated-frame agreement decreases over the longer
rollout: 29.58 dB PSNR and 0.963029 RGB8 SSIM. The initial frames match exactly,
then the outputs diverge. The audited reference includes 0, 4, 8, 12, or 16 extra
visible tokens at its page boundaries; FA4 uses exact frame boundaries. Thus
these paths do not always attend to identical token sets. The longer result
does not establish numerical equivalence from the short sample.

To profile regional `torch.compile` execution separately from the eager pair:

```bash
bash var/cosmos-sim-fa4-frame-pages/source/profile_compiled.sh
```

This writes to `profile_241_compiled/`, sets `enforce_eager=False`, enables
dynamic regional Inductor compilation, and keeps CUDA graphs disabled. Two
25-frame warmups precede each measured 241-frame rollout. Worker-side
`*.compile.jsonl` audits require all transformer blocks to have compiled
forwards; the summary rejects measurements with new Dynamo/Inductor compilation
and requires fused Inductor kernels in the trace.

For this compiled pair, measured generation took 25.33 seconds (reference) and
25.15 seconds (FA4). Both use 36 compiled transformer blocks and four graph
variants, with no graph breaks or new compilation during measurement. These
are single profiled observations with CUDA graphs disabled.
