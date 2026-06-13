"""
FFHQ256 LDM sampling with optional DeepCache acceleration and FID evaluation.

Flow: sample -> save gen PNGs -> export real PNGs -> compute FID -> save summary -> cleanup PNGs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import logging
import os
import shutil
import sys
import time
from typing import Any, Dict, Optional

import numpy as np
import torch
from omegaconf import OmegaConf
from PIL import Image

torch.set_grad_enabled(False)

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

TAMING_ROOT = os.path.join(REPO_ROOT, "src", "taming-transformers")
if os.path.isdir(TAMING_ROOT) and TAMING_ROOT not in sys.path:
    sys.path.insert(0, TAMING_ROOT)

from ldm.models.diffusion.ddim import DDIMSampler
from ldm.util import instantiate_from_config
from scripts.sample_diffusion import compute_fid, export_real_from_image_dir

FID_DIMS = 2048
IMG_SIZE = 256


def load_model_from_config(config_path: str, ckpt_path: str, device: torch.device) -> torch.nn.Module:
    print(f"Loading model from {ckpt_path}")
    pl_sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = pl_sd["state_dict"]
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model)
    model.load_state_dict(sd, strict=False)
    model.to(device)
    model.eval()
    return model


def resolve_repo_path(path_str: str) -> str:
    if os.path.isabs(path_str):
        return os.path.abspath(path_str)
    return os.path.join(REPO_ROOT, path_str)


def build_run_label(
    replicate_interval: Optional[int],
    nonuniform: bool,
    pow: float,
    num_samples: int,
    ddim_steps: int,
    eta: float,
) -> str:
    if nonuniform:
        return (
            f"interval{replicate_interval}_nonuniform_pow{pow}_"
            f"samples{num_samples}_steps{ddim_steps}_eta{eta}"
        )
    return f"interval{replicate_interval}_samples{num_samples}_steps{ddim_steps}_eta{eta}"


def format_fid_at(num_samples: int) -> str:
    if num_samples % 1000 == 0:
        return f"{num_samples // 1000}k"
    return str(num_samples)


def count_pngs(dir_path: str) -> int:
    if not os.path.isdir(dir_path):
        return 0
    return len(glob.glob(os.path.join(dir_path, "*.png")))


def save_batch_pngs(samples_np: np.ndarray, gen_dir: str, start_index: int) -> None:
    for i, img in enumerate(samples_np):
        Image.fromarray(img).save(os.path.join(gen_dir, f"{start_index + i:06d}.png"))


def append_json_list(path: str, record: Dict[str, Any]) -> None:
    history = []
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if isinstance(existing, list):
                history = existing
            elif isinstance(existing, dict):
                history = [existing]
        except Exception:
            history = []
    history.append(record)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)


def prepare_real_images(real_image_dir: str, eval_dir: str, num_images: int) -> None:
    export_real_from_image_dir(
        real_image_dir=real_image_dir,
        eval_dir=eval_dir,
        num_images=num_images,
        img_size=IMG_SIZE,
        real_image_list=None,
    )
    real_count = count_pngs(eval_dir)
    if real_count < num_images:
        raise RuntimeError(
            f"real image count {real_count} < num_images {num_images} in {eval_dir}"
        )


def cleanup_image_dirs(gen_dir: str, eval_dir: str) -> None:
    for path in (gen_dir, eval_dir):
        if os.path.isdir(path):
            shutil.rmtree(path)
            logging.info("removed %s", path)


def main() -> None:
    parser = argparse.ArgumentParser(description="FFHQ256 LDM DeepCache sampling with FID")
    parser.add_argument("--config", type=str, default="models/ldm/ffhq256/config.yaml")
    parser.add_argument("--ckpt", type=str, default="models/ldm/ffhq256/model.ckpt")
    parser.add_argument("--num_samples", type=int, default=10000)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--ddim_steps", type=int, default=200)
    parser.add_argument("--eta", type=float, default=0.0)
    parser.add_argument("--out_dir", type=str, default="./deepcache_results")
    parser.add_argument(
        "--real_image_dir",
        type=str,
        required=True,
        help="FFHQ real image directory (e.g. ffhq-dataset/images1024x1024)",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--replicate_interval", type=int, default=None)
    parser.add_argument("--nonuniform", action="store_true")
    parser.add_argument("--pow", type=float, default=1.5)
    parser.add_argument(
        "--results_json",
        type=str,
        default="results/fid_results_deepcache.json",
        help="Append FID summary records to this JSON list",
    )
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.info("args: %s", args)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this script.")

    device = torch.device("cuda")
    torch.cuda.set_device(0)

    if args.resume:
        torch.manual_seed(int(time.time()))
    else:
        torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    config_path = resolve_repo_path(args.config)
    ckpt_path = resolve_repo_path(args.ckpt)
    real_image_dir = resolve_repo_path(args.real_image_dir)
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"config not found: {config_path}")
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"ckpt not found: {ckpt_path}")
    if not os.path.isdir(real_image_dir):
        raise FileNotFoundError(f"real_image_dir not found: {real_image_dir}")

    run_label = build_run_label(
        args.replicate_interval,
        args.nonuniform,
        args.pow,
        args.num_samples,
        args.ddim_steps,
        args.eta,
    )
    run_dir = os.path.join(resolve_repo_path(args.out_dir), run_label)
    gen_dir = os.path.join(run_dir, "gen_images")
    eval_dir = os.path.join(run_dir, "real_eval")
    summary_path = os.path.join(run_dir, "summary.json")

    os.makedirs(run_dir, exist_ok=True)
    if args.resume:
        os.makedirs(gen_dir, exist_ok=True)
    else:
        if os.path.isdir(gen_dir):
            shutil.rmtree(gen_dir)
        os.makedirs(gen_dir, exist_ok=True)

    model = load_model_from_config(config_path, ckpt_path, device)
    logging.info("conditioning_key=%s", model.model.conditioning_key)

    sampler = DDIMSampler(model)
    shape = [3, 64, 64]

    n_saved = count_pngs(gen_dir)
    if args.resume:
        logging.info("resume: found %d existing PNGs in %s", n_saved, gen_dir)
    elif n_saved > 0:
        logging.warning("found %d PNGs in gen_dir without --resume; continuing from existing count", n_saved)

    start_dt = dt.datetime.now()
    sample_time_min = 0.0
    fid_time_min = 0.0
    score: Optional[float] = None

    try:
        logging.info("sampling to %s", gen_dir)
        t0 = time.time()
        while n_saved < args.num_samples:
            current_batch_size = min(args.batch_size, args.num_samples - n_saved)
            batch_t0 = time.time()

            with model.ema_scope("Sampling"):
                samples_ddim, _ = sampler.sample(
                    S=args.ddim_steps,
                    batch_size=current_batch_size,
                    shape=shape,
                    conditioning=None,
                    verbose=False,
                    eta=args.eta,
                    replicate_interval=args.replicate_interval,
                    nonuniform=args.nonuniform,
                    pow=args.pow,
                )
                x_samples = model.decode_first_stage(samples_ddim)

            x_samples = torch.clamp((x_samples + 1.0) / 2.0, min=0.0, max=1.0)
            samples = x_samples.mul(255).add_(0.5).clamp_(0, 255).to(torch.uint8)
            samples = samples.permute(0, 2, 3, 1).contiguous().cpu().numpy()

            save_batch_pngs(samples, gen_dir, n_saved)
            n_saved += current_batch_size

            batch_t1 = time.time()
            logging.info(
                "created %d / %d samples (batch throughput: %.2f img/s)",
                n_saved,
                args.num_samples,
                current_batch_size / max(batch_t1 - batch_t0, 1e-6),
            )

        sample_time_min = (time.time() - t0) / 60.0

        gen_count = count_pngs(gen_dir)
        if gen_count != args.num_samples:
            raise RuntimeError(
                f"generated image count mismatch: expected {args.num_samples}, got {gen_count}"
            )

        logging.info("preparing real images from %s", real_image_dir)
        t1 = time.time()
        if os.path.isdir(eval_dir):
            shutil.rmtree(eval_dir)
        prepare_real_images(real_image_dir, eval_dir, args.num_samples)

        logging.info("computing FID (dims=%d, batch_size=%d)", FID_DIMS, args.batch_size)
        score = float(compute_fid(eval_dir, gen_dir, args.batch_size, device, FID_DIMS))
        fid_time_min = (time.time() - t1) / 60.0
        logging.info("FID=%.6f", score)

    except Exception:
        end_dt = dt.datetime.now()
        summary_obj = {
            "status": "failed",
            "run_label": run_label,
            "start_time": start_dt.isoformat(timespec="seconds"),
            "end_time": end_dt.isoformat(timespec="seconds"),
            "duration_sec": round((end_dt - start_dt).total_seconds(), 2),
            "num_samples": args.num_samples,
            "seed": args.seed,
            "replicate_interval": args.replicate_interval,
            "nonuniform": args.nonuniform,
            "pow": args.pow,
            "ddim_steps": args.ddim_steps,
            "eta": args.eta,
            "gen_dir": gen_dir,
            "real_image_dir": real_image_dir,
        }
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary_obj, f, indent=2)
        raise

    end_dt = dt.datetime.now()
    fid_at = format_fid_at(args.num_samples)
    fid_metric_key = f"fid_{fid_at}"

    summary_obj = {
        "status": "success",
        "run_label": run_label,
        "start_time": start_dt.isoformat(timespec="seconds"),
        "end_time": end_dt.isoformat(timespec="seconds"),
        "duration_sec": round((end_dt - start_dt).total_seconds(), 2),
        "num_samples": args.num_samples,
        "seed": args.seed,
        "replicate_interval": args.replicate_interval,
        "nonuniform": args.nonuniform,
        "pow": args.pow,
        "ddim_steps": args.ddim_steps,
        "eta": args.eta,
        "fid_dims": FID_DIMS,
        "fid_at": fid_at,
        fid_metric_key: score,
        "sample_time_min": float(sample_time_min),
        "fid_time_min": float(fid_time_min),
        "real_image_dir": real_image_dir,
        "config": config_path,
        "ckpt": ckpt_path,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary_obj, f, indent=2)
    logging.info("summary saved to %s", summary_path)

    results_json_path = resolve_repo_path(args.results_json)
    append_json_list(
        results_json_path,
        {
            "timestamp": end_dt.isoformat(timespec="seconds"),
            "run_label": run_label,
            "fid": score,
            "fid_at": fid_at,
            "num_samples": args.num_samples,
            "replicate_interval": args.replicate_interval,
            "nonuniform": args.nonuniform,
            "pow": args.pow,
            "summary_path": summary_path,
        },
    )
    logging.info("appended result to %s", results_json_path)

    cleanup_image_dirs(gen_dir, eval_dir)
    logging.info("sampling and FID complete")


if __name__ == "__main__":
    main()
