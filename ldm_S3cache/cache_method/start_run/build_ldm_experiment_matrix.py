#!/usr/bin/env python3
"""Build ldm_experiment_matrix.json and ldm_experiment_matrix_summary.md."""

import json
import re
from collections import defaultdict
from pathlib import Path

LDM_ROOT = Path(__file__).resolve().parent / "results"
RESULTS_ROOT = Path("/home/jimmy/latent-diffusion/results")
OUT_JSON = LDM_ROOT / "ldm_experiment_matrix.json"
OUT_MD = LDM_ROOT / "ldm_experiment_matrix_summary.md"
TODAY = "2026-06-27"

QUANTIZATION_METHODS = {
    "Q-LDM": {
        "method": "TFMQ-DM",
        "paper_title": "TFMQ-DM: Temporal Feature Maintenance Quantization for Diffusion Models",
        "authors": "Huang et al.",
        "venue": "CVPR 2024",
        "venue_note": "Highlight Poster (top 2.8%)",
        "bit_width": "W8A8",
        "checkpoint": "/home/jimmy/TFMQ-DM/cali_ckpt/ffhq256_w8a8.pth",
        "arxiv": "https://arxiv.org/abs/2311.16503",
        "open_access": "https://openaccess.thecvf.com/content/CVPR2024/html/Huang_TFMQ-DM_Temporal_Feature_Maintenance_Quantization_for_Diffusion_Models_CVPR_2024_paper.html",
        "github": "https://github.com/ModelTC/TFMQ-DM",
        "journal_extension": {
            "title": "Temporal Feature Matters for Diffusion Model Quantization",
            "venue": "TPAMI 2025",
            "arxiv": "https://arxiv.org/abs/2407.19547",
        },
    },
}


def parse_stage1(name: str):
    if not name:
        return None
    clean = name.replace("_50k", "")
    m = re.match(r"K(\d+)_sw(\d+)_lam([\d.]+)", clean)
    if m:
        lam = float(m.group(3))
        return {"K": int(m.group(1)), "sw": int(m.group(2)), "lambda": lam}
    return None


def sch_key(sch: str) -> str:
    if "baseline_no_npz_qldm" in sch or sch == "baseline_no_npz_qldm":
        return "baseline_qldm"
    if "baseline" in sch:
        return "baseline"
    return sch.replace("_50k", "")


def fid_from_summary(s: dict, num_images: int):
    if num_images >= 50000:
        return s.get("fid_50k") or s.get("fid_5k")
    return s.get("fid_5k") or s.get("fid_50k")


def rel_path(p: str) -> str:
    home = "/home/jimmy/latent-diffusion/"
    return p.replace(home, "") if p.startswith(home) else p


def build_experiment(
    model: str,
    is_baseline: bool,
    sch_norm: str,
    stage1,
    canonical: dict,
    primary: bool = False,
    role: str = "",
):
    prefix = "QL" if model == "Q-LDM" else "FP"
    baseline_id = f"{prefix}-B-DDIM200"
    if is_baseline:
        exp_id = baseline_id
    elif stage1:
        lam = stage1["lambda"]
        lam_s = str(int(lam)) if lam == int(lam) else str(lam)
        exp_id = f"{prefix}-S3-K{stage1['K']}sw{stage1['sw']}lam{lam_s}"
    else:
        exp_id = f"{prefix}-S3-{sch_norm}"

    r5 = canonical.get((model, sch_norm, "5k"))
    r50 = canonical.get((model, sch_norm, "50k"))
    fid_5k = r5["fid"] if r5 else None
    fid_50k = r50["fid"] if r50 else None

    if is_baseline:
        rho = 1.0
    else:
        src = r5 or r50
        rho = src["full_compute_ratio"] if src else None

    exp = {
        "id": exp_id,
        "model": model,
        "cache_method": None if is_baseline else "S3-Cache",
        "sampler": "DDIM",
        "steps": 200,
        "stage1_config": stage1,
        "fid_5k": fid_5k,
        "fid_50k": fid_50k,
        "full_compute_ratio": rho,
        "delta_fid_5k_abs": None,
        "delta_fid_5k_pct": None,
        "delta_fid_50k_abs": None,
        "delta_fid_50k_pct": None,
        "baseline_ref": None if is_baseline else baseline_id,
        "is_primary": primary,
        "role": role,
        "additional_metrics": None,
    }

    if r5:
        exp["source_file"] = r5["path"]
        if Path(r5["detail_stats"]).exists():
            exp["source_file_detail_stats"] = r5["detail_stats"]
    if r50:
        exp["source_file_fid_50k"] = r50["path"]

    if not is_baseline:
        b5 = canonical.get((model, "baseline_qldm" if model == "Q-LDM" else "baseline", "5k"))
        b50 = canonical.get((model, "baseline_qldm" if model == "Q-LDM" else "baseline", "50k"))
        if fid_5k and b5 and b5["fid"]:
            d = fid_5k - b5["fid"]
            exp["delta_fid_5k_abs"] = round(d, 6)
            exp["delta_fid_5k_pct"] = round(d / b5["fid"] * 100, 2)
        if fid_50k and b50 and b50["fid"]:
            d = fid_50k - b50["fid"]
            exp["delta_fid_50k_abs"] = round(d, 6)
            exp["delta_fid_50k_pct"] = round(d / b50["fid"] * 100, 2)

    return exp


def quantization_md_lines() -> list[str]:
    q = QUANTIZATION_METHODS["Q-LDM"]
    j = q["journal_extension"]
    return [
        "---",
        "",
        "## 量化方法",
        "",
        "本矩陣中 **Q-LDM** 實驗採用下列論文的 post-training quantization（PTQ）方法；**FP LDM** 為全精度 baseline，未做量化。",
        "",
        "| 模型標籤 | 量化方法 | 論文 | 頂會 | 本實驗設定 |",
        "|----------|----------|------|------|------------|",
        (
            f"| Q-LDM | **{q['method']}** | {q['authors']}, *{q['paper_title']}* "
            f"| **{q['venue']}**（{q['venue_note']}） "
            f"| {q['bit_width']}；cali ckpt：`{q['checkpoint']}`；base ckpt：FFHQ256 LDM |"
        ),
        "",
        "**參考連結**",
        "",
        (
            f"- TFMQ-DM 論文：[arXiv:2311.16503]({q['arxiv']}) · "
            f"[CVPR 2024 Open Access]({q['open_access']}) · [GitHub]({q['github']})"
        ),
        (
            f"- 期刊延伸版：*{j['title']}* — **{j['venue']}**（"
            f"[arXiv:2407.19547]({j['arxiv']})）"
        ),
        "",
        "---",
        "",
    ]


def write_md(matrix: dict) -> None:
    meta = matrix["metadata"]
    exps = matrix["experiments"]
    primary = [e for e in exps if e.get("is_primary")]

    lines = [
        "# LDM / Q-LDM S3-Cache 實驗矩陣摘要",
        "",
        f"**建立日期：** {meta['created']}  ",
        f"**最後更新：** {meta['last_updated']}  ",
        "**資料來源：** `start_run/results/**/summary.json` 逐檔驗證 + `results/fid_results_*.json`  ",
        f"**機器設定：** LDM FFHQ256, DDIM steps=200, eta=0, seed=0  ",
        f"**Checkpoint：** `{meta['checkpoint']}`",
        "",
        *quantization_md_lines(),
        "## 主結果（`is_primary: true`）",
        "",
        "| ID | Model | Cache | Steps | FID@5K | FID@50K | ρ(M) | ΔFID@5K | ΔFID@50K |",
        "|----|-------|-------|-------|--------|---------|------|---------|----------|",
    ]

    for e in primary:
        cache = e.get("cache_method") or "—"
        f5 = f"{e['fid_5k']:.3f}" if e.get("fid_5k") else "—"
        f50 = f"{e['fid_50k']:.3f}" if e.get("fid_50k") else "—"
        rho = f"{e['full_compute_ratio']:.3f}" if e.get("full_compute_ratio") is not None else "—"
        d5 = f"{e['delta_fid_5k_pct']:+.1f}%" if e.get("delta_fid_5k_pct") is not None else "—"
        d50 = f"{e['delta_fid_50k_pct']:+.1f}%" if e.get("delta_fid_50k_pct") is not None else "—"
        lines.append(
            f"| {e['id']} | {e['model']} | {cache} | {e['steps']} | {f5} | {f50} | {rho} | {d5} | {d50} |"
        )

    lines += ["", "---", "", "## 全部結果", ""]

    sections = [
        ("A. FP LDM Baseline", lambda e: e["model"] == "FP LDM" and not e.get("cache_method")),
        ("B. FP LDM + S3-Cache", lambda e: e["model"] == "FP LDM" and e.get("cache_method") == "S3-Cache"),
        ("C. Q-LDM Baseline", lambda e: e["model"] == "Q-LDM" and not e.get("cache_method")),
        ("D. Q-LDM + S3-Cache", lambda e: e["model"] == "Q-LDM" and e.get("cache_method") == "S3-Cache"),
        ("E. DeepCache（對照）", lambda e: e.get("cache_method") == "DeepCache"),
    ]

    for title, pred in sections:
        subset = [e for e in exps if pred(e)]
        if not subset:
            continue
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| ID | FID@5K | FID@50K | ρ | ΔFID@5K | Source |")
        lines.append("|----|--------|---------|---|---------|--------|")
        for e in subset:
            f5 = f"{e['fid_5k']:.3f}" if e.get("fid_5k") else "—"
            f50 = f"{e['fid_50k']:.3f}" if e.get("fid_50k") else "—"
            rho = f"{e['full_compute_ratio']:.3f}" if e.get("full_compute_ratio") is not None else "—"
            if e.get("delta_fid_5k_pct") is not None:
                d5 = f"{e['delta_fid_5k_pct']:+.1f}%"
            elif e.get("delta_fid_50k_pct") is not None:
                d5 = f"{e['delta_fid_50k_pct']:+.1f}% (@50K)"
            else:
                d5 = "—"
            src = rel_path(e.get("source_file") or e.get("source_file_fid_50k") or "")
            if e.get("source_file_fid_50k") and e.get("source_file"):
                src50 = rel_path(e["source_file_fid_50k"])
                src = f"`{rel_path(e['source_file'])}` + `{src50}`"
            else:
                src = f"`{src}`" if src else "—"
            star = " ★" if e.get("is_primary") else ""
            lines.append(f"| {e['id']}{star} | {f5} | {f50} | {rho} | {d5} | {src} |")
        lines.append("")

    lines += [
        "---",
        "",
        "## 驗證過程中發現的不一致",
        "",
        "### 1. Q-LDM baseline 5K 有多筆 run（已選 canonical）",
        "",
        "- **0614_12**（batch_size=32）：FID@5K ≈ 13.88 — 排除",
        "- **0614_17**（batch_size=16）：FID@5K = 11.777 — **canonical**",
        "",
        "### 2. 早期 Q-LDM S3 sweep（0613，batch32）",
        "",
        "`fid_results_qldm.json` 中 0613 的 K15/K20 sweep（FID@5K ≈ 15.9）與 0614 sweep 設定不同，",
        "本矩陣以 `start_run/results/fid_5k_qldm/20260614/` 的 0614 sweep 為準。",
        "",
        "### 3. FP LDM K15 5K 重複 run",
        "",
        "0420_09 / 0420_10 / 0613_20 結果相同（FID=11.710），canonical 取最新 **0613_20**。",
        "",
        "### 4. Q-LDM S3 50K FID 略優於 baseline",
        "",
        "QL-S3-K8sw5λ0.5 FID@50K=5.788 vs QL-B FID@50K=5.968（ΔFID=-3.0%）。",
        "建議論文引用時註明此為單次 run 結果。",
        "",
        "---",
        "",
        "## 待完成實驗",
        "",
    ]
    for p in meta.get("pending_experiments", []):
        lines.append(f"- {p}")
    lines += [
        "",
        "---",
        "",
        "## 檔案位置",
        "",
        f"- **結構化 JSON：** `{OUT_JSON}`",
        f"- **本摘要：** `{OUT_MD}`",
        f"- **重建腳本：** `ldm_S3cache/cache_method/start_run/build_ldm_experiment_matrix.py`",
        "",
    ]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")


def main():
    raw_runs = []
    for p in sorted(LDM_ROOT.rglob("summary.json")):
        s = json.loads(p.read_text())
        ni = s.get("num_images", 5000)
        rel_campaign = p.relative_to(LDM_ROOT).parts[0]
        is_qldm = "qldm" in rel_campaign
        mode = s.get("mode", "cache")
        sch = s.get("scheduler_name", "")
        is_baseline = mode == "baseline" or sch.startswith("baseline")
        raw_runs.append(
            {
                "path": str(p.resolve()),
                "detail_stats": str((p.parent / "detail_stats.json").resolve()),
                "model": "Q-LDM" if is_qldm else "FP LDM",
                "is_baseline": is_baseline,
                "scheduler_name": sch,
                "sch_norm": sch_key(sch),
                "stage1_config": parse_stage1(sch),
                "num_images": ni,
                "fid_at": "50k" if ni >= 50000 else "5k",
                "fid": fid_from_summary(s, ni),
                "full_compute_ratio": s.get("full_compute_ratio"),
                "start_time": s.get("start_time", ""),
            }
        )

    groups = defaultdict(list)
    for r in raw_runs:
        groups[(r["model"], r["sch_norm"], r["fid_at"])].append(r)
    canonical = {}
    for key, items in groups.items():
        items.sort(key=lambda x: x["start_time"])
        canonical[key] = items[-1]

    experiments = []

    experiments.append(
        build_experiment(
            "FP LDM",
            True,
            "baseline",
            None,
            canonical,
            primary=True,
            role="FP LDM baseline (DDIM200, no_npz)",
        )
    )
    for sch_norm, stage1, primary, role in [
        ("K15_sw3_lam1.0", {"K": 15, "sw": 3, "lambda": 1.0}, True, "FP LDM + S3-Cache selected (K15/sw3/λ1.0)"),
        ("K15_sw3_lam0.5", {"K": 15, "sw": 3, "lambda": 0.5}, False, "FP LDM + S3-Cache ablation"),
        ("K20_sw3_lam1.0", {"K": 20, "sw": 3, "lambda": 1.0}, False, "FP LDM + S3-Cache ablation"),
    ]:
        experiments.append(
            build_experiment("FP LDM", False, sch_norm, stage1, canonical, primary=primary, role=role)
        )

    experiments.append(
        build_experiment(
            "Q-LDM",
            True,
            "baseline_qldm",
            None,
            canonical,
            primary=True,
            role="Q-LDM baseline (DDIM200, no_npz); canonical 5K=0614_17 batch16",
        )
    )
    for sch_norm, stage1, primary, role in [
        ("K8_sw5_lam0.5", {"K": 8, "sw": 5, "lambda": 0.5}, True, "Q-LDM + S3-Cache selected (K8/sw5/λ0.5)"),
        ("K12_sw5_lam0.5", {"K": 12, "sw": 5, "lambda": 0.5}, False, "Q-LDM + S3-Cache ablation"),
        ("K15_sw3_lam0.5", {"K": 15, "sw": 3, "lambda": 0.5}, False, "Q-LDM + S3-Cache ablation"),
    ]:
        experiments.append(
            build_experiment("Q-LDM", False, sch_norm, stage1, canonical, primary=primary, role=role)
        )

    for jf, model in [
        ("fid_results_deepcache.json", "FP LDM"),
        ("fid_results_deepcache_quant.json", "Q-LDM"),
    ]:
        fp = RESULTS_ROOT / jf
        if not fp.exists():
            continue
        for e in json.loads(fp.read_text()):
            prefix = "QL" if model == "Q-LDM" else "FP"
            fid_at = e.get("fid_at", "5k")
            exp = {
                "id": f"{prefix}-DC-int{e['replicate_interval']}",
                "model": model,
                "cache_method": "DeepCache",
                "sampler": "DDIM",
                "steps": 200,
                "stage1_config": {"replicate_interval": e["replicate_interval"], "pow": e.get("pow", 1.5)},
                "fid_5k": e["fid"] if fid_at == "5k" else None,
                "fid_50k": e["fid"] if fid_at == "50k" else None,
                "full_compute_ratio": None,
                "delta_fid_5k_abs": None,
                "delta_fid_5k_pct": None,
                "delta_fid_50k_abs": None,
                "delta_fid_50k_pct": None,
                "baseline_ref": f"{prefix}-B-DDIM200",
                "is_primary": False,
                "role": f"DeepCache comparison (interval={e['replicate_interval']})",
                "source_file": e.get("summary_path"),
                "additional_metrics": None,
            }
            b = next((x for x in experiments if x["id"] == f"{prefix}-B-DDIM200"), None)
            if b:
                if fid_at == "5k" and b.get("fid_5k") and e.get("fid"):
                    d = e["fid"] - b["fid_5k"]
                    exp["delta_fid_5k_abs"] = round(d, 6)
                    exp["delta_fid_5k_pct"] = round(d / b["fid_5k"] * 100, 2)
                if fid_at == "50k" and b.get("fid_50k") and e.get("fid"):
                    d = e["fid"] - b["fid_50k"]
                    exp["delta_fid_50k_abs"] = round(d, 6)
                    exp["delta_fid_50k_pct"] = round(d / b["fid_50k"] * 100, 2)
            experiments.append(exp)

    metadata = {
        "created": TODAY,
        "last_updated": TODAY,
        "description": "LDM / Q-LDM S3-Cache experiment matrix - verified results",
        "model": "LDM FFHQ256",
        "dataset": "ffhq256",
        "ddim_steps": 200,
        "eta": 0.0,
        "seed": 0,
        "batch_size_canonical": {"FP LDM": 32, "Q-LDM": 16},
        "fid_dims": 2048,
        "fid_evaluator": "clean-fid (PyTorch Inception)",
        "checkpoint": "/home/jimmy/latent-diffusion/models/ldm/ffhq256/model.ckpt",
        "quantization_methods": QUANTIZATION_METHODS,
        "notes": [
            "FID values taken from summary.json (fid_5k / fid_50k per num_images).",
            "Canonical Q-LDM baseline 5K uses 0614_17 (batch_size=16); earlier 0614_12 (batch32, FID≈13.88) excluded.",
            "Early Q-LDM S3 runs in fid_results_qldm.json (0613, batch32) excluded; canonical from 0614 sweep.",
            "FP LDM duplicate K15 5K runs (0420_09/0420_10/0613_20) — canonical is latest 0613_20.",
            "DeepCache results from latent-diffusion/results/fid_results_deepcache*.json (external paths).",
            "ρ = full_compute_ratio from summary.json.",
        ],
        "pending_experiments": [
            "FP LDM S3 50K for K20/K15_lam0.5 configs",
            "Q-LDM S3 50K for K12/K15 configs",
            "Q-LDM DeepCache 50K",
        ],
    }

    matrix = {"metadata": metadata, "experiments": experiments}
    OUT_JSON.write_text(json.dumps(matrix, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_md(matrix)
    print(f"Wrote {OUT_JSON} ({len(experiments)} experiments)")
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
