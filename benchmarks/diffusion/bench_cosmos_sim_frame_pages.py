"""Paired Cosmos Sim rollouts; select frame pages with VLLM_OMNI_COSMOS_SIM_FA4_FRAME_PAGES=1."""

import argparse
import json
import os
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch
import yaml
from diffusers.utils import export_to_video
from huggingface_hub import snapshot_download
from PIL import Image

from vllm_omni.entrypoints.omni import Omni
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="nvidia/Cosmos3-Nano-Sim-Bimanual")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-frames", type=int, default=25)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--warmup-frames", type=int, default=0)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--compile", action="store_true", help="Enable regional torch.compile (Inductor)")
    args = parser.parse_args()
    checkpoint = Path(args.model)
    if not checkpoint.is_dir():
        checkpoint = Path(snapshot_download(args.model, local_files_only=True))
    manifest = checkpoint / "examples/inference/sim_bimanual_unifiedv1_hf.jsonl"
    example = json.loads(manifest.read_text().splitlines()[0])
    prompt = ". ".join(example["prompt"].split(". ")[:3]) + "."
    sidecar = json.loads(
        (checkpoint / "assets/sim_bimanual/agibot/actions/agibot_001_normalized_unifiedv1.json").read_text()
    )
    for count in (args.num_frames, args.warmup_frames):
        if count and (count < 1 or (count - 1) % 4 or count - 1 > len(sidecar["action"])):
            raise ValueError("Frame counts must be 1 + 4*N and fit the available action rows")
    mode = "fa4_frame_pages" if os.environ.get("VLLM_OMNI_COSMOS_SIM_FA4_FRAME_PAGES") == "1" else "baseline"
    output_dir = args.output_dir / mode
    output_dir.mkdir(parents=True, exist_ok=True)
    deploy = {
        "pipeline": "cosmos3_nano_sim_bimanual",
        "async_chunk": False,
        "distributed_executor_backend": "mp",
        "dtype": "bfloat16",
        "default_resolution": [720, 1280],
        "max_pixels": 921600,
        "stages": [
            {
                "stage_id": 0,
                "devices": "0",
                "max_num_seqs": 1,
                "enforce_eager": not args.compile,
                "diffusion_compile_granularity": "regional",
                "diffusion_compile_dynamic": True,
                "model_class_name": "Cosmos3NanoSimBimanualPipeline",
                "engine_backend": "vllm_omni.experimental.ar_diffusion.engine.ARDiffusionEngine",
                "model_config": {
                    "guardrails": False,
                    "clean_commit_mode": "batched",
                    "overlap_vae_decode": False,
                    "use_cuda_graphs": False,
                    "kv_cache_inference_size": 4,
                    "attention_sink_size": 1,
                    "ar_diffusion_kv_config": {"gpu_memory_fraction": 0.1, "warmup_cudagraph": False},
                },
            }
        ],
    }
    deploy_path = output_dir / "deploy.yaml"
    deploy_path.write_text(yaml.safe_dump(deploy))
    image = Image.open(checkpoint / "assets/sim_bimanual/agibot/images/agibot_001_initial.png").convert("RGB")
    sampling = OmniDiffusionSamplingParams(
        height=720,
        width=1280,
        num_frames=args.num_frames,
        num_inference_steps=4,
        guidance_scale=1.0,
        fps=30,
        seed=args.seed,
        output_type="np",
        extra_args={
            "action": sidecar["action"][: args.num_frames - 1],
            "action_space": sidecar["action_space"],
            "domain_name": example["domain_name"],
            "reset": True,
            "close_session": True,
            "guardrails": False,
        },
    )
    omni = Omni(model=str(checkpoint), deploy_config=str(deploy_path))
    try:
        request = {"prompt": prompt, "multi_modal_data": {"image": image}}
        if args.warmup_frames:
            sampling.num_frames = args.warmup_frames
            for index in range(args.warmup_runs):
                with torch.cuda.nvtx.range(f"cosmos.warmup.{index}"):
                    omni.generate(request, sampling)
            sampling.num_frames = args.num_frames
        started = time.perf_counter()
        with torch.cuda.nvtx.range("cosmos.generate"):
            result = omni.generate(request, sampling)[0]
        elapsed = time.perf_counter() - started
        frames = np.asarray(result.images)
        while frames.ndim > 4 and frames.shape[0] == 1:
            frames = frames[0]
        if frames.shape != (args.num_frames, 720, 1280, 3):
            raise RuntimeError(f"Unexpected video output: {frames.shape}")
        if not np.isfinite(frames).all():
            raise RuntimeError("Non-finite video output")
        np.save(output_dir / "frames.npy", frames)
        export_to_video(list(frames), str(output_dir / "rollout.mp4"), fps=30, macro_block_size=1)
        report = {
            "mode": mode,
            "checkpoint": str(checkpoint),
            "seed": args.seed,
            "prompt": prompt,
            "num_inference_steps": 4,
            "guidance_scale": 1.0,
            "fps": 30,
            "shape": list(frames.shape),
            "elapsed_seconds": elapsed,
            "warmup_frames": args.warmup_frames,
            "warmup_runs": args.warmup_runs if args.warmup_frames else 0,
            "torch_compile": args.compile,
            "compile_granularity": "regional" if args.compile else None,
            "compile_dynamic": True if args.compile else None,
            "cuda_graphs": False,
            "latent_frames": (args.num_frames - 1) // 4 + 1,
            "video_tokens_per_frame": 920,
            "action_tokens_per_frame": 4,
            "sink_frames": 1,
            "recent_past_frames": 2,
            "torch": torch.__version__,
            "flash_attn_4": version("flash-attn-4"),
            "gpu": torch.cuda.get_device_name(0),
            "compute_capability": torch.cuda.get_device_capability(0),
        }
        (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2), flush=True)
    finally:
        omni.close()


if __name__ == "__main__":
    main()
