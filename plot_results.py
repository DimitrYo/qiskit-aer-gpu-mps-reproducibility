#!/usr/bin/env python3
"""Rebuild the current article's two figures and supporting CSVs from raw records.

Run: python plot_results.py --output-dir output/figures
This redraws the same data and diagram with Matplotlib. Fonts and PDF bytes can
differ from the published assets. No new simulations or timings are performed.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

from verify_results import geometric, sha256, verify


LABELS = {"brickwork": "BW", "qft_prepared": "QFT", "qaoa_ring": "QAOA",
          "variational_chain": "Вар. ланцюг", "ghz": "GHZ"}
FAMILIES = ("brickwork", "qft_prepared", "qaoa_ring", "variational_chain", "ghz")
WORKLOADS = ("BW-14", "BW-15", "BW-16", "pQFT-16", "BW-18")
CPU, GPU = "#245b89", "#b65020"


def collect(result):
    independent = []
    for workload in WORKLOADS:
        row = next(r for r in result["independent"]["analyses"] if r["workload"] == workload)
        points = [{"input_id": p["input_id"], "cpu_over_gpu": p["geometric_mean_speedup"],
                   "cpu_seconds": geometric([b["cpu_median_seconds"] for b in p["blocks"]]),
                   "gpu_seconds": geometric([b["gpu_median_seconds"] for b in p["blocks"]])}
                  for p in row["input_summaries"]]
        independent.append({"workload": workload.replace("pQFT", "QFT"), "role": row["role"],
                            "inference": row["inference"], "input_count": len(points),
                            "cpu_seconds": geometric([p["cpu_seconds"] for p in points]),
                            "gpu_seconds": geometric([p["gpu_seconds"] for p in points]),
                            "cpu_over_gpu": row["geometric_mean_speedup"],
                            "confidence_interval": row["confidence_interval"], "inputs": points})
    old = result["historical"]
    medians = {(r["circuit_id"], r["backend"]): r["median_seconds"] for r in old["cells"]}
    observed = {r["circuit_id"] for r in result["historical_offload"]["cells"] if r["cutensor_csvd_wrapper_calls"] > 0}
    specs = {r["id"]: r for r in result["historical_manifest"]["circuits"]}
    historical = []
    for row in old["end_to_end_comparisons"]:
        spec = specs[row["circuit_id"]]
        lo, hi = row["bootstrap"]["interval"]
        historical.append({"circuit_id": row["circuit_id"], "kind": spec["kind"], "qubits": spec["qubits"],
                           "cpu_seconds": medians[(row["circuit_id"], "custom_cpu")],
                           "gpu_seconds": medians[(row["circuit_id"], "custom_gpu")],
                           "cpu_over_gpu": 1 / row["point_estimate"],
                           "confidence_interval": [1 / hi, 1 / lo],
                           "svd_observed": row["circuit_id"] in observed})
    return {"verification": result["verification"], "archive_sha256": result["archive_sha256"],
            "ratio_orientation": "CPU/GPU; above 1 means shorter GPU runtime",
            "independent_interval": "95% whole-input bootstrap; paired CPU/GPU and both blocks retained; secondary intervals descriptive pointwise",
            "independent_runtime": "geometric mean of both block medians within each input, then across inputs; no runtime confidence intervals",
            "fixed_interval": "minimum and maximum of two fixed-input block ratios; not a confidence interval",
            "historical_interval": "95% independent bootstrap of CPU/GPU timing repetitions of the same fixed circuit; not between-input inference",
            "offload_filter_applied": False, "cpu_selection": result["cpu_selection"],
            "independent": independent, "fixed": result["fixed"]["rows"], "historical": historical}


def write_tables(data, output, edition="current"):
    if edition == "current":
        tables = {
            "table01_methods.csv": (["parameter", "independent_inputs", "fixed_inputs"], [
                ["circuits", "36: BW14/15/16/18 and QFT16", "26: BW1-22 and four additional 20-qubit families"],
                ["bond_dimension_limit", 1024, 2048], ["timing_blocks", 2, 2],
                ["warmups_per_cell", 5, 5], ["retained_calls_per_cell", 20, 10],
                ["shots_per_call", 100, 100],
                ["interval", "95% whole-input bootstrap", "min-max of two fixed-input block ratios"]]),
            "results.csv": (["sample", "workload", "independent_inputs", "cpu_seconds", "gpu_seconds", "cpu_over_gpu", "interval_type", "lower", "upper"],
                [["independent", r["workload"], r["input_count"], r["cpu_seconds"], r["gpu_seconds"], r["cpu_over_gpu"], "95% whole-input CI", *r["confidence_interval"]] for r in data["independent"]] +
                [["fixed", f"{LABELS[r['kind']]}-{r['qubits']}", 1, r["cpu_seconds"], r["gpu_seconds"], r["cpu_gpu_ratio"], "two-block min-max", r["block_ratio_min"], r["block_ratio_max"]] for r in data["fixed"]])}
        for name, (header, rows) in tables.items():
            with (output / name).open("w", encoding="utf-8", newline="") as stream:
                writer = csv.writer(stream); writer.writerow(header); writer.writerows(rows)
        return list(tables)
    candidates = ("custom_cpu", "custom_cpu_lapack", "upstream_cpu_lapack")
    cpu_rows = []
    for label, circuit in (("BW-14", "bw14_d24_s20260724_p1"), ("BW-15", "bw15_d24_s20260724_p1"),
                           ("BW-16", "bw16_d24_s20260724_p1"), ("BW-18", "bw18_d24_s20260724_p1"), ("QFT-16", "pqft16_p1")):
        cpu_rows.append([label] + [next(r["geometric_mean_seconds"] for r in data["cpu_selection"]["rows"]
                                     if r["candidate"] == c and r["circuit_id"] == circuit) for c in candidates])
    cpu_rows.append(["Equal input weights"] + [data["cpu_selection"]["geometric_mean_seconds"][c] for c in candidates])
    tables = {"table03_cpu_selection.csv": (["workload", *candidates], cpu_rows),
              "table04_independent_inputs.csv": (["workload", "independent_inputs", "cpu_seconds", "gpu_seconds", "cpu_over_gpu"],
                  [[r[k] for k in ("workload", "input_count", "cpu_seconds", "gpu_seconds", "cpu_over_gpu")] for r in data["independent"]]),
              "table05_fixed_families.csv": (["workload", "cpu_seconds", "gpu_seconds", "cpu_over_gpu"],
                  [[LABELS[kind], r["cpu_seconds"], r["gpu_seconds"], r["cpu_gpu_ratio"]]
                   for kind in FAMILIES for r in data["fixed"] if r["kind"] == kind and r["qubits"] == 20])}
    for name, (header, rows) in tables.items():
        with (output / name).open("w", encoding="utf-8", newline="") as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(rows)
    return list(tables)


def render(data, output, edition="current"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Polygon
    from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 8.5, "axes.labelsize": 8.5,
                         "xtick.labelsize": 8, "ytick.labelsize": 8.5, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "svg.hashsalt": "gpu-mps-article-reproduction",
                         "axes.spines.top": False, "axes.spines.right": False})
    files = {}

    def save(fig, stem):
        if edition == "current":
            current_names = {"fig04_fixed_brickwork": "fig02_fixed_brickwork", "fig03_independent_ratio": "fig01_independent_ratio"}
            if stem not in current_names:
                plt.close(fig)
                return
            stem = current_names[stem]
        for extension in ("pdf", "svg", "png"):
            path = output / f"{stem}.{extension}"
            metadata = {"Creator": "GPU MPS article reproduction", "CreationDate": None, "ModDate": None} if extension == "pdf" else None
            fig.savefig(path, dpi=200, metadata=metadata)
            files[path.name] = sha256(path.read_bytes())
        plt.close(fig)

    def ratio_axis(ax, lows, highs, horizontal=True, high_floor=2):
        low, high = min(0.5, min(lows)), max(high_floor, max(highs))
        margin = max(math.log(high / low) * .07, math.log(1.08))
        low, high = low / math.exp(margin), high * math.exp(margin)
        if horizontal:
            ax.set_xscale("log", base=2)
            ax.set_xlim(low, high)
            ax.axvline(1, linestyle="--", color="#50585f", lw=.9)
            ax.set_xlabel("Відношення часу CPU/GPU")
            ax.grid(axis="x", color="#d7dfe5", linewidth=.55)
            axis = ax.xaxis
        else:
            ax.set_yscale("log", base=2)
            ax.set_ylim(low, high)
            ax.axhline(1, linestyle="--", color="#50585f", lw=.9)
            ax.set_ylabel("Відношення часу CPU/GPU")
            ax.grid(axis="y", color="#d7dfe5", linewidth=.55)
            axis = ax.yaxis
        ticks = [v for v in LogLocator(base=2, numticks=7).tick_values(low, high) if low <= v <= high]
        axis.set_major_locator(FixedLocator(sorted(set(ticks + [1]))))
        axis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}×"))
        axis.set_minor_formatter(NullFormatter())

    # Source-derived diagram: it describes dispatch, never measured phase times.
    fig, ax = plt.subplots(figsize=(157 / 25.4, 55 / 25.4))
    fig.subplots_adjust(left=.015, right=.985, bottom=.04, top=.96)
    ax.set(xlim=(0, 10), ylim=(0, 4))
    ax.axis("off")
    def box(x, y, w, h, text, face="#edf3f8"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=.05,rounding_size=.08", facecolor=face, edgecolor="#536c80", lw=.8))
        ax.text(x+w/2, y+h/2, text, ha="center", va="center", fontsize=8)
    box(.05, 1.5, 2.0, 1, "Згортання або SVD\nДані на CPU")
    ax.add_patch(Polygon([(2.5, 2), (3.7, 2.9), (4.9, 2), (3.7, 1.1)], facecolor="#fff5df", edgecolor="#98762d", lw=.8))
    ax.text(3.7, 2, "Умови GPU\nвиконано?", ha="center", va="center", fontsize=8)
    box(5.45, 2.55, 2.15, 1, "Обчислення\nна CPU")
    box(5.45, .3, 2.15, 1.35, "CPU → GPU\ncuBLAS / cuTensorNet\nGPU → CPU", "#e8f3ee")
    box(8.25, 1.5, 1.65, 1, "Результат\nна CPU")
    def arrow(a, b):
        ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, lw=.8, color="#536c80"))
    arrow((2.1, 2), (2.47, 2))
    arrow((4.65, 2.2), (5.39, 3.05)); ax.text(5.03, 2.82, "Ні", ha="center", fontsize=8)
    arrow((4.65, 1.8), (5.39, .97)); ax.text(5.01, 1.08, "Так", ha="center", fontsize=8)
    arrow((7.66, 3.05), (8.21, 2.3)); arrow((7.66, .97), (8.21, 1.7))
    save(fig, "fig01_pipeline")

    rows = data["independent"]
    ys = list(reversed(range(len(rows))))
    fig, ax = plt.subplots(figsize=(157 / 25.4, 67 / 25.4))
    fig.subplots_adjust(left=.21, right=.97, bottom=.21, top=.86)
    for y, row in zip(ys, rows):
        ax.plot([row["cpu_seconds"], row["gpu_seconds"]], [y, y], color="#a5abb0", lw=1)
        ax.plot(row["cpu_seconds"], y, "o", color=CPU, ms=5)
        ax.plot(row["gpu_seconds"], y, "s", color=GPU, ms=5)
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(FixedLocator([.2, .5, 1, 2, 5]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_yticks(ys, [r["workload"] for r in rows])
    ax.set_xlabel("Геометричне середнє часу, с")
    ax.grid(axis="x", color="#d7dfe5", linewidth=.55)
    fig.legend(handles=[Line2D([], [], marker="o", ls="", color=CPU, label="CPU"), Line2D([], [], marker="s", ls="", color=GPU, label="GPU")], loc="upper center", frameon=False, ncol=2)
    save(fig, "fig02_independent_runtime")

    fig, ax = plt.subplots(figsize=(157 / 25.4, 67 / 25.4))
    fig.subplots_adjust(left=.24, right=.98, bottom=.2, top=.94)
    ratio_axis(ax, [min(r["confidence_interval"][0], *[p["cpu_over_gpu"] for p in r["inputs"]]) for r in rows],
               [max(r["confidence_interval"][1], *[p["cpu_over_gpu"] for p in r["inputs"]]) for r in rows], high_floor=1)
    for y, row in zip(ys, rows):
        offsets = [-.19 + .38*i/max(1, len(row["inputs"])-1) for i in range(len(row["inputs"]))]
        ax.scatter([p["cpu_over_gpu"] for p in row["inputs"]], [y+o for o in offsets], s=13, color="#a5abb0", edgecolor="white", linewidth=.3, zorder=3)
        low, high = row["confidence_interval"]
        ax.hlines(y, low, high, color=CPU, lw=1.7, zorder=4)
        ax.vlines([low, high], y-.075, y+.075, color=CPU, lw=1.2)
        ax.plot(row["cpu_over_gpu"], y, "D", color=CPU, ms=4.5, zorder=5)
    ax.set_yticks(ys, [f"{r['workload']} (n={r['input_count']})" for r in rows])
    save(fig, "fig03_independent_ratio")

    markers = {(False, False): ("o", "white", "#63717c"), (True, False): ("o", CPU, CPU),
               (False, True): ("s", "#e8c391", "#86591b"), (True, True): ("D", "#28765d", "#17553f")}
    def marker(ax, x, y, row):
        svd, blas = row["svd_submitted_calls"], row["cublas_submitted_calls"]
        shape, face, edge = ("x", "none", "#888888") if svd is None or blas is None else markers[(svd > 0, blas > 0)]
        ax.plot(x, y, marker=shape, ls="", ms=4.2, markerfacecolor=face, markeredgecolor=edge, markeredgewidth=.85, zorder=4)

    fixed = data["fixed"]
    bw = sorted((r for r in fixed if r["kind"] == "brickwork"), key=lambda r: r["qubits"])
    families = [next(r for r in fixed if r["kind"] == kind and r["qubits"] == 20) for kind in FAMILIES]
    for stem, selected, horizontal in (("fig04_fixed_brickwork", bw, False), ("fig05_fixed_families", families, True)):
        fig, ax = plt.subplots(figsize=(157 / 25.4, 60 / 25.4))
        fig.subplots_adjust(left=.22 if horizontal else .16, right=.975, bottom=.22, top=.95)
        ratio_axis(ax, [r["block_ratio_min"] for r in selected], [r["block_ratio_max"] for r in selected], horizontal)
        for i, row in enumerate(selected):
            low, high, value = row["block_ratio_min"], row["block_ratio_max"], row["cpu_gpu_ratio"]
            if horizontal:
                y = len(selected)-i-1
                ax.hlines(y, low, high, color="#42596c", lw=1.05)
                ax.vlines([low, high], y-.08, y+.08, color="#42596c", lw=.85)
                marker(ax, value, y, row)
            else:
                x = row["qubits"]
                ax.vlines(x, low, high, color="#42596c", lw=1.05)
                ax.hlines([low, high], x-.12, x+.12, color="#42596c", lw=.85)
                marker(ax, x, value, row)
        if horizontal:
            ax.set_yticks(list(reversed(range(len(selected)))), [LABELS[r["kind"]] for r in selected])
            ax.set_ylim(-.45, len(selected)-.55)
        else:
            ax.set_xlim(.5, 22.5)
            ax.set_xticks(range(1, 23))
            ax.set_xlabel("Кількість кубітів")
        save(fig, stem)

    matched = sorted((r for r in data["historical"] if r["qubits"] == 16), key=lambda r: r["cpu_over_gpu"], reverse=True)
    fig, ax = plt.subplots(figsize=(157 / 25.4, 50 / 25.4))
    fig.subplots_adjust(left=.22, right=.975, bottom=.27, top=.95)
    ratio_axis(ax, [r["confidence_interval"][0] for r in matched], [r["confidence_interval"][1] for r in matched])
    for y, row in enumerate(matched):
        value = row["cpu_over_gpu"]
        low, high = row["confidence_interval"]
        ax.errorbar(value, y, xerr=[[value-low], [high-value]], fmt="o", ms=5, capsize=3, color=CPU, mfc=CPU if row["svd_observed"] else "white", lw=1.1)
    ax.set_yticks(range(len(matched)), [LABELS[r["kind"]] for r in matched])
    ax.set_ylim(len(matched)-.6, -.4)
    ax.set_xlabel("Відношення медіан часу CPU/GPU")
    save(fig, "fig06_historical_families")
    return files


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent / "output/figures")
    parser.add_argument("--edition", choices=("current", "020"), default="current",
                        help="Current two-figure article (default), or all six figures of archived revision 020")
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("Choose a new output directory; existing figures are preserved.")
    data = collect(verify(args.data_dir))
    args.output_dir.mkdir(parents=True)
    table_files = write_tables(data, args.output_dir, args.edition)
    data["figure_files_sha256"] = render(data, args.output_dir, args.edition)
    data["article_edition"] = args.edition
    data["fixed_marker_definition"] = "open circle: neither call observed; filled circle: SVD only; square: cuBLAS only; diamond: both; x: unrecorded. Separate diagnostic call; no speedup attribution."
    data["historical_marker_definition"] = "filled circle: GPU SVD trace observed; open circle: not observed"
    (args.output_dir / "plotted_data.json").write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "figure_count": 2 if args.edition == "current" else 6, "tables": table_files, "output_directory": str(args.output_dir.resolve()), "new_experiments": False}))


if __name__ == "__main__":
    main()
