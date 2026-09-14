#!/usr/bin/env python3
"""Verify archived file hashes and recompute the article's timing statistics.

Standard-library Python; no simulation, GPU, Qiskit or archive extraction.
Saved correctness metrics, article settings and the cited diagnostic counts are
checked as well. Statevector computations and source regression tests are not rerun.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import statistics
import zipfile


EXPECTED_STUDIES = {
    "implementation-tests", "cpu-selection", "independent-pilot", "independent-confirmation",
    "mechanism-diagnostics", "fixed-input", "historical-scaling-16", "historical-scaling-18",
    "historical-scaling-20", "historical-workloads", "execution-order", "svd-variants",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(value):
    return hashlib.sha256(value).hexdigest()


def agree(actual, expected, label):
    require(math.isfinite(actual) and math.isclose(actual, expected, rel_tol=1e-12, abs_tol=1e-15),
            f"{label}: recorded {actual!r}, recomputed {expected!r}")


def geometric(values):
    return math.exp(math.fsum(math.log(x) for x in values) / len(values))


def quantile(ordered, p):
    position = (len(ordered) - 1) * p
    lo, hi = math.floor(position), math.ceil(position)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (position - lo)


class Study:
    def __init__(self, root, slug, record):
        self.slug = slug
        archive = root / record["archive"]
        require(archive.resolve().parent == root.resolve(), "Archive must be directly in data directory")
        require(sha256(archive.read_bytes()) == record["sha256"], f"Archive hash mismatch: {slug}")
        self.archive = zipfile.ZipFile(archive)
        names = self.archive.namelist()
        require(len(names) == len(set(names)) and set(names) == set(record["files"]), f"Archive inventory mismatch: {slug}")
        for name, digest in record["files"].items():
            require(not name.startswith(("/", "\\")) and ".." not in Path(name).parts, "Unsafe archive member")
            require(sha256(self.archive.read(name)) == digest, f"Raw file hash mismatch: {slug}/{name}")
        self.file_count = len(names)
        self.timing_cells = 0
        self.retained_calls = 0

    def read(self, name):
        return json.loads(self.archive.read(name))

    def jsonl(self, name):
        return [json.loads(line) for line in self.archive.read(name).decode("utf-8").splitlines() if line.strip()]

    def bound_summary(self, name):
        result = self.read(name)
        require(result["status"] == "PASS", f"Non-PASS summary: {self.slug}")
        require(result["contract_sha256"] == sha256(self.archive.read("contract.json")), "Contract hash mismatch")
        return result

    def median(self, directory, count=20):
        rows = self.jsonl(directory + "/timings.jsonl")
        warmups = self.jsonl(directory + "/warmups.jsonl")
        require(len(rows) == count and len(warmups) == 5, f"Wrong repetition count: {self.slug}/{directory}")
        require(all(r["is_warmup"] is False for r in rows), "Warmup included in timing")
        require(all(r["is_warmup"] is True for r in warmups), "Invalid warmup marker")
        require([r["sample_index"] for r in rows] == list(range(count)), "Timing repetition order differs")
        values = [r["elapsed_seconds"] for r in rows]
        require(all(math.isfinite(x) and x > 0 for x in values), "Nonpositive or nonfinite timing")
        self.timing_cells += 1
        self.retained_calls += len(rows)
        return statistics.median(values)


def cpu_selection(study):
    summary = study.bound_summary("cpu_selection.json")
    for row in summary["rows"]:
        medians = [study.median(f"timing/block_{b}/{row['circuit_id']}/{row['candidate']}") for b in (0, 1)]
        for actual, expected in zip(row["block_medians_seconds"], medians):
            agree(actual, expected, "CPU selection block median")
        agree(row["geometric_mean_seconds"], geometric(medians), "CPU selection input time")
    for candidate, value in summary["geometric_mean_seconds"].items():
        agree(value, geometric([r["geometric_mean_seconds"] for r in summary["rows"] if r["candidate"] == candidate]), "Equal input weight selection")
    require(summary["selected_cpu"] == min(summary["geometric_mean_seconds"], key=summary["geometric_mean_seconds"].get), "CPU selection differs")
    return summary


def paired_inputs(study, points, cpu, gpu):
    logs = []
    for point in points:
        block_logs = []
        for block in point["blocks"]:
            prefix = f"timing/session_{block['block_id'] - 1}/{point['input_id']}"
            c, g = study.median(prefix + "/" + cpu), study.median(prefix + "/" + gpu)
            agree(block["cpu_median_seconds"], c, "Paired CPU median")
            agree(block["gpu_median_seconds"], g, "Paired GPU median")
            agree(block["speedup_cpu_over_gpu"], c / g, "Paired block CPU/GPU")
            block_logs.append(math.log(c / g))
        log = math.fsum(block_logs) / len(block_logs)
        agree(point["mean_log_speedup"], log, "Mean input log ratio")
        agree(point["geometric_mean_speedup"], math.exp(log), "Input geometric ratio")
        logs.append(log)
    return logs


def pilot(study):
    summary = study.bound_summary("analysis.json")
    plan = summary["sample_size_plan"]
    logs = paired_inputs(study, plan["pilot_input_summaries"], plan["cpu_candidate"], plan["gpu_candidate"])
    require(len(logs) == 5, "Sample-size pilot requires five inputs")
    mean = math.fsum(logs) / len(logs)
    sd = math.sqrt(math.fsum((x - mean) ** 2 for x in logs) / (len(logs) - 1))
    required = max(2, math.ceil((plan["planning_critical_value"] * sd / math.log1p(0.15)) ** 2))
    agree(plan["pilot_sample_sd_log"], sd, "Pilot standard deviation")
    require(plan["plug_in_required_inputs"] == required and plan["selected_primary_inputs"] == min(20, max(12, required)), "Sample-size planning mismatch")
    return plan


def confirmation(study, plan):
    summary = study.bound_summary("analysis.json")
    require(len(summary["analyses"]) == 5, "Expected five independent workloads")
    for row in summary["analyses"]:
        ids = row["input_ids"]
        require(ids == [p["input_id"] for p in row["input_summaries"]] and len(ids) == len(set(ids)), "Missing or duplicate independent input")
        require(not set(ids).intersection(plan["pilot_input_ids"]), "Pilot input reused for confirmation")
        require(len(ids) == (plan["selected_primary_inputs"] if row["role"] == "primary" else 6), "Independent input count differs")
        require(row["block_ids"] == [1, 2] and all([b["block_id"] for b in p["blocks"]] == [1, 2] for p in row["input_summaries"]), "Whole inputs must retain both blocks")
        require(row["offload_summary"]["filter_applied"] is False, "Offload filtering changes the sample")
        logs = paired_inputs(study, row["input_summaries"], row["cpu_candidate"], row["gpu_candidate"])
        bootstrap = row["bootstrap"]
        require(bootstrap["unit"] == "whole_input" and bootstrap["within_input_resampling"] is False and bootstrap["cpu_gpu_pair_preserved"] and bootstrap["all_blocks_preserved"], "Bootstrap unit differs")
        require(bootstrap["repetitions"] == 10000 and row["confidence_level"] == 0.95, "Bootstrap protocol differs")
        rng = random.Random(bootstrap["seed"])
        sampled = sorted(math.fsum(logs[rng.randrange(len(logs))] for _ in logs) / len(logs) for _ in range(10000))
        tail = (1.0 - row["confidence_level"]) / 2
        ci = [math.exp(quantile(sampled, p)) for p in (tail, 1 - tail)]
        for actual, expected in zip(row["confidence_interval"], ci):
            agree(actual, expected, "Whole-input bootstrap confidence bound")
        agree(row["geometric_mean_speedup"], math.exp(math.fsum(logs) / len(logs)), "Aggregate paired ratio")
    return summary


def fixed_inputs(study):
    summary = study.bound_summary("analysis.json")
    require(summary["input_count"] == 26 and len(summary["rows"]) == 26, "Fixed input inventory differs")
    for row in summary["rows"]:
        require(row["independent_inputs"] == 1 and "confidence_interval" not in row, "Fixed input cannot imply population inference")
        values = []
        for candidate, key in (("custom_cpu_lapack", "cpu"), ("custom_gpu_lapack", "gpu")):
            medians = [study.median(f"cells/timing/block_{b}/{row['circuit_id']}/{candidate}", 10) for b in (0, 1)]
            for actual, expected in zip(row[f"block_{key}_medians"], medians):
                agree(actual, expected, "Fixed input block median")
            agree(row[f"{key}_seconds"], geometric(medians), "Fixed input geometric time")
            values.append(medians)
        ratios = [c / g for c, g in zip(*values)]
        for actual, expected in zip(row["block_cpu_gpu_ratios"], ratios):
            agree(actual, expected, "Fixed input block ratio")
        for field, expected in (("cpu_gpu_ratio", geometric(ratios)), ("block_ratio_min", min(ratios)), ("block_ratio_max", max(ratios))):
            agree(row[field], expected, field)
    return summary


def historical(study):
    aggregate = study.read("crossover_aggregate.json")
    values = defaultdict(list)
    for row in study.jsonl("timings_raw.jsonl"):
        require(row["is_warmup"] is False, "Historical warmup included in timing")
        values[(row["circuit_id"], row["backend"])].append(row["elapsed_seconds"])
    for row in aggregate["cells"]:
        sample = values[(row["circuit_id"], row["backend"])]
        require(len(sample) == row["timed_sample_count"] == 20, "Historical cell count differs")
        agree(row["median_seconds"], statistics.median(sample), "Historical raw median")
    for row in aggregate["end_to_end_comparisons"]:
        cpu, gpu = (values[(row["circuit_id"], backend)] for backend in ("custom_cpu", "custom_gpu"))
        bootstrap = row["bootstrap"]
        require(bootstrap["replications"] == 10000 and bootstrap["confidence_level"] == 0.95, "Historical bootstrap protocol differs")
        rng = random.Random(bootstrap["seed"])
        ratios = []
        for _ in range(10000):
            c = statistics.median(rng.choice(cpu) for _ in cpu)
            g = statistics.median(rng.choice(gpu) for _ in gpu)
            ratios.append(g / c)
        ratios.sort()
        agree(row["point_estimate"], statistics.median(gpu) / statistics.median(cpu), "Historical GPU/CPU ratio")
        for actual, expected in zip(bootstrap["interval"], [quantile(ratios, p) for p in (0.025, 0.975)]):
            agree(actual, expected, "Historical timing bootstrap confidence bound")
    return aggregate


def article_evidence(studies):
    """Check the saved evidence for the current manuscript; do not execute Aer."""
    checks = {}
    for slug, count, chi, repetitions, prefix, suffix, order_key in (
            ("independent-confirmation", 72, 1024, 20, "correctness/", "/correctness.json", "sessions"),
            ("fixed-input", 52, 2048, 10, "cells/correctness/", "/report.json", "blocks")):
        study = studies[slug]
        contract = study.read("contract.json")
        for key, expected in {"warmups": 5, "timed_repetitions": repetitions, "shots": 100,
                              "threads": 6, "max_bond_dimension": chi,
                              "truncation_threshold": 0, "precision": "double"}.items():
            require(contract["protocol"][key] == expected, f"Article protocol differs: {slug}/{key}")
        require(contract["cpu_affinity"] == [0, 2, 4, 6, 8, 10], "Article CPU affinity differs")
        candidates = contract["candidates"]
        require(len(candidates) == 2 and all(c["mps_lapack"] for c in candidates), "Article CPU-LAPACK pairing differs")
        require({c["source_revision"] for c in candidates} == {"a1242579272784330b217085fe11b7e1481922ad"}, "Article source revision differs")
        require(len({c["extension_sha256"] for c in candidates}) == 1, "CPU/GPU must use the same module")
        first, second = contract[order_key]
        require(first["input_order"] == list(reversed(second["input_order"])), "Input order not reversed")
        require(all(order == list(reversed(second["candidate_orders"][key]))
                    for key, order in first["candidate_orders"].items()), "CPU/GPU order not reversed")
        if slug == "independent-confirmation":
            require(contract["minimum_session_gap_seconds"] >= 1800, "Session gap differs")
        names = [n for n in study.archive.namelist() if n.startswith(prefix) and n.endswith(suffix)]
        require(len(names) == count, f"Correctness report count differs: {slug}")
        norms, fidelities, devices = [], [], defaultdict(int)
        for name in names:
            report = study.read(name)
            require(report["status"] == "PASS" and len(report["records"]) == 1, "Failed saved correctness report")
            config = report["simulation_config"]
            require(config["method"] == "matrix_product_state" and config["max_bond_dimension"] == chi
                    and config["precision"] == "double" and config["truncation_threshold"] == 0
                    and config["mps_lapack"] and config["shots"] == 1, "Correctness configuration differs")
            require(config["device"] == ("CPU" if "/custom_cpu_lapack/" in name else "GPU"), "Correctness device differs")
            devices[config["device"]] += 1
            record = report["records"][0]
            ref, mps = record["reference"], record["mps"]
            require(record["status"] == "PASS", "Failed saved state check")
            require(ref["fidelity_min"] == 1 - 1e-12 and math.isfinite(ref["fidelity"]) and ref["fidelity"] >= ref["fidelity_min"], "Statevector references disagree")
            require(mps["fidelity_min"] == 1 - 1e-10 and math.isfinite(mps["fidelity"]) and mps["fidelity"] >= mps["fidelity_min"], "Saved MPS fidelity failed")
            require(mps["norm_error_max"] == 1e-10 and 0 <= mps["norm_error"] <= mps["norm_error_max"], "Saved MPS norm failed")
            norms.append(mps["norm_error"]); fidelities.append(mps["fidelity"])
        require(dict(devices) == {"CPU": count // 2, "GPU": count // 2}, "Correctness device counts differ")
        checks[slug] = {"saved_state_checks": count, "maximum_norm_error": max(norms),
                        "minimum_fidelity": min(fidelities), "protocol": contract["protocol"]}
    study = studies["mechanism-diagnostics"]
    counts = {}
    for mode in ("gpu_evolution", "gpu_shots100"):
        prefix = f"cells/bw_n16_pilot_s102000/{mode}/"
        summary = study.read(prefix + "summary.json")
        cpu = sum(r["phase"] == "factorization" for r in study.jsonl(prefix + "cpu_lapack.jsonl"))
        svd = len(study.jsonl(prefix + "svd.events.jsonl"))
        require(cpu == summary["cpu_lapack"]["factorization_count"] and svd == summary["svd_submitted_calls"], "Diagnostic summary differs from raw events")
        counts[mode] = {"cpu_factorizations": cpu, "gpu_svd_calls": svd}
    require(counts["gpu_evolution"] == {"cpu_factorizations": 149, "gpu_svd_calls": 31}
            and counts["gpu_shots100"] == {"cpu_factorizations": 1649, "gpu_svd_calls": 31}, "Article BW16 diagnostic counts differ")
    checks["bw16_diagnostics"] = counts
    return checks


def verify(data_dir):
    root = data_dir.resolve()
    provenance_path = root.parent / "PROVENANCE.json"
    require(provenance_path.is_file(), "Required PROVENANCE.json is missing")
    provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    required_files = {"data/index.json", *[f"data/{slug}.zip" for slug in EXPECTED_STUDIES]}
    require(required_files <= set(provenance["files"]), "Provenance omits a required evidence file")
    for name, record in provenance["files"].items():
        path = root.parent / name
        require(path.resolve().is_relative_to(root.parent), "Provenance path outside package")
        require(sha256(path.read_bytes()) == record["sha256"], f"Package file hash mismatch: {name}")
    index = json.loads((root / "index.json").read_text(encoding="utf-8"))
    require(index["schema_version"] == 1, "Unknown data index version")
    require(set(index["studies"]) == EXPECTED_STUDIES, "Data index must contain exactly the 12 article studies")
    require(all(record["archive"] == f"{slug}.zip" for slug, record in index["studies"].items()), "Unexpected archive filename")
    studies = {slug: Study(root, slug, record) for slug, record in index["studies"].items()}
    try:
        cpu = cpu_selection(studies["cpu-selection"])
        plan = pilot(studies["independent-pilot"])
        independent = confirmation(studies["independent-confirmation"], plan)
        fixed = fixed_inputs(studies["fixed-input"])
        old = historical(studies["historical-workloads"])
        scaling = {str(n): historical(studies[f"historical-scaling-{n}"]) for n in (16, 18, 20)}
        manifest = studies["historical-workloads"].read("manifest.json")
        proof = studies["historical-workloads"].read("offload_proof.json")
        diagnostics = studies["mechanism-diagnostics"].read("validation.json")
        article_checks = article_evidence(studies)
        report = {"status": "PASS", "new_experiments": False,
                  "archives_checked": len(studies), "files_hash_checked": sum(s.file_count for s in studies.values()),
                  "recomputed": ["CPU selection", "five-input sample-size plan", "independent paired medians and whole-input bootstrap", "fixed-input medians and two-block ranges", "historical workload/scaling medians and independent timing bootstrap"],
                  "saved_evidence_checked": article_checks,
                  "integrity_only": ["source regression logs", "remaining mechanism diagnostic logs", "execution-order and SVD-variant studies"],
                  "floating_point_comparison": "relative tolerance 1e-12, absolute tolerance 1e-15; file hashes exact"}
        return {"verification": report, "cpu_selection": cpu, "pilot_plan": plan,
                "independent": independent, "fixed": fixed, "historical": old,
                "historical_manifest": manifest, "historical_offload": proof,
                "historical_scaling": scaling, "diagnostics": diagnostics,
                "archive_sha256": {slug: record["sha256"] for slug, record in index["studies"].items()}}
    finally:
        for study in studies.values():
            study.archive.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent / "data")
    args = parser.parse_args()
    print(json.dumps(verify(args.data_dir)["verification"], indent=2))


if __name__ == "__main__":
    main()
