#!/usr/bin/env python3
"""Aggregate all JIGSAWS LOUO result JSONs into one master table.

Usage:
    python -m dino_foresight.aggregate_results \
        --root outputs/jigsaws_masterplan \
        --output outputs/jigsaws_masterplan/master_results.md
"""

import argparse
import json
from pathlib import Path
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=str, default="outputs/jigsaws_masterplan")
    p.add_argument("--output", type=str, default="outputs/jigsaws_masterplan/master_results.md")
    return p.parse_args()


def load_json(path):
    with open(path) as f:
        return json.load(f)


def find_result_files(root):
    """Find result JSONs in known subdirectories."""
    root = Path(root)
    files = []
    for subdir in [
        "eval", "eval_pilot", "eval_decoder", "eval_singlelayer",
        "copy_last_baseline", "vptr_baseline",
        "eval_tracka2_rgb", "eval_tracka2_p3_rgb",
        "eval_feature_copy_last_tracka2", "eval_feature_copy_last_tracka2_p3",
    ]:
        d = root / subdir
        if d.exists():
            files.extend(d.glob("*.json"))
    return files


def format_num(v, decimals=4):
    if v is None:
        return "-"
    return f"{v:.{decimals}f}"


def main():
    args = parse_args()
    root = Path(args.root)
    root.mkdir(parents=True, exist_ok=True)

    files = find_result_files(root)
    # Feature-space JSONs (non stride*_t* keys)
    feat_dirs = [
        "eval_feature_copy_last",
        "eval_feature_copy_last_tracka2",
        "eval_feature_copy_last_tracka2_p3",
    ]
    feat_rows = []
    for sub in feat_dirs:
        d = root / sub
        jf = d / "feature_copy_last_vs_predictor.json"
        if jf.exists():
            data = load_json(jf)
            for split, by_h in data.items():
                for hkey, block in by_h.items():
                    t = int(hkey.replace("t", ""))
                    for method in ("copy_last", "predictor"):
                        m = block[method]
                        feat_rows.append({
                            "method": sub,
                            "split": split,
                            "t": t,
                            "variant": method,
                            "smooth_l1": m.get("smooth_l1"),
                            "cosine": m.get("cosine"),
                        })

    if not files and not feat_rows:
        print(f"No result JSONs found under {args.root}")
        return

    rows = []
    if files:
        for file in sorted(files):
            data = load_json(file)
            method = file.parent.name
            if "_stride" in file.stem:
                stride_tag = file.stem.split("_")[-1]
            else:
                stride_tag = file.stem
            for key, res in sorted(data.items()):
                if not (isinstance(key, str) and "_t" in key and key.startswith("stride")):
                    continue
                parts = key.split("_t")
                if len(parts) != 2:
                    continue
                try:
                    stride = int(parts[0].replace("stride", ""))
                    t = int(parts[1])
                except ValueError:
                    continue
                rows.append({
                    "method": method,
                    "run": stride_tag,
                    "stride": stride,
                    "t": t,
                    "PSNR": res.get("psnr_avg"),
                    "SSIM": res.get("ssim_avg"),
                    "LPIPS": res.get("lpips_avg"),
                    "n_clips": res.get("n_clips"),
                })

    if not rows and not feat_rows:
        print("No valid result entries found.")
        return

    lines = []
    if rows:
        lines = [
            "# JIGSAWS LOUO Master Results",
            "",
            "| Method | Run | Stride | t | PSNR | SSIM | LPIPS | n_clips |",
            "|--------|-----|--------|---|------|------|-------|---------|",
        ]
        for row in rows:
            psnr_v = format_num(row["PSNR"], 2)
            ssim_v = format_num(row["SSIM"], 4)
            lpips_v = format_num(row["LPIPS"], 4)
            lines.append(
                f"| {row['method']} | {row['run']} | {row['stride']} | {row['t']} | "
                f"{psnr_v} | {ssim_v} | {lpips_v} | {row['n_clips']} |"
            )

        csv_lines = ["method,run,stride,t,psnr,ssim,lpips,n_clips"]
        for row in rows:
            csv_lines.append(
                f"{row['method']},{row['run']},{row['stride']},{row['t']},"
                f"{row['PSNR'] or ''},{row['SSIM'] or ''},{row['LPIPS'] or ''},{row['n_clips']}"
            )
        out_md = Path(args.output)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        with open(out_md, "w") as f:
            f.write("\n".join(lines) + "\n")
        out_csv = out_md.with_suffix(".csv")
        with open(out_csv, "w") as f:
            f.write("\n".join(csv_lines) + "\n")
        print("\n".join(lines))
        print(f"\nSaved {out_md} and {out_csv}")
    else:
        out_md = Path(args.output)

    if feat_rows:
        feat_md = out_md.parent / "master_feature_results.md"
        flines = [
            "# JIGSAWS LOUO Feature-Space Results",
            "",
            "| Method | Split | t | Variant | SmoothL1 | Cosine |",
            "|--------|-------|---|---------|----------|--------|",
        ]
        for r in feat_rows:
            flines.append(
                f"| {r['method']} | {r['split']} | {r['t']} | {r['variant']} | "
                f"{format_num(r['smooth_l1'], 4)} | {format_num(r['cosine'], 4)} |"
            )
        with open(feat_md, "w") as f:
            f.write("\n".join(flines) + "\n")
        print(f"Saved {feat_md}")


if __name__ == "__main__":
    main()
