#!/usr/bin/env python3
"""Run the article's CPU selection, independent comparison and fixed-input study.

The small interface unpacks the original workers from data/*.zip. Fresh runs
freeze those sources and actual local build identities. Failed runs are kept.
Preparation and --help never run simulations. Collection requires Linux/WSL.
"""
from __future__ import annotations

import argparse
import difflib
import hashlib
import importlib
import io
import json
import os
from pathlib import Path
import platform
import shlex
import shutil
import subprocess
import sys
import tokenize
import zipfile
from datetime import datetime, timezone

CUSTOM_REVISION = "a1242579272784330b217085fe11b7e1481922ad"
UPSTREAM_REVISION = "51c679814c3a292d0d7c59bb39976bd6ff91f60e"
HISTORICAL_REVISION = "75137ba7dee76ec864cb723a6c3aa9650fbf4f43"
CPU_ENTRY = "run_scientific_cpu_pilot.py"
PAIRED_ENTRY = "run_scientific_paired_campaign.py"
FIXED_ENTRY = "run_supplementary_fixed_input_map.py"
MASTER_SEED = 20260912
SEED_POLICY = "20260912-v1-pcg64-jumped-streams"


def seeded_source(source, independent=False):
    """Adapt fresh worker copies; archived evidence is never rewritten."""
    tokens = []
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.NUMBER and token.string in {
                "20260722", "20260723", "20260724", "20260907", "202609071"}:
            token = token._replace(string=str(MASTER_SEED))
        tokens.append(token)
    text = tokenize.untokenize(tokens)
    text = text.replace("_s20260724", f"_s{MASTER_SEED}")
    text = text.replace(f"seed={MASTER_SEED} + offset", f"seed={MASTER_SEED}")
    text = text.replace("simulator.run(compiled, shots=1).result()",
                        f"simulator.run(compiled, shots=1, seed_simulator={MASTER_SEED}).result()")
    if independent:
        old_return = "    return SEED_BASES[stage] + SCOPES.index(scope) * SEEDS_PER_SCOPE + index"
        new_return = (f"    return {MASTER_SEED}\n\n\n"
                      "def stream_for(stage, kind, qubits, index):\n"
                      "    seed_for(stage, kind, qubits, index)  # validate stream coordinates\n"
                      "    return SEED_BASES[stage] + SCOPES.index((kind, qubits)) * SEEDS_PER_SCOPE + index")
        text = replacement(text, old_return, new_return)
        text = replacement(text, "rng = np.random.Generator(np.random.PCG64(seed))",
                           "rng = np.random.Generator(np.random.PCG64(seed).jumped(stream_for(stage, kind, qubits, index)))")
        text = replacement(text, 'f"{label}_n{qubits}_{stage}_s{seed}"',
                           'f"{label}_n{qubits}_{stage}_s{seed}_i{index}"')
        text = replacement(text, '"numpy_version": np.__version__, "seed": seed,',
                           '"numpy_version": np.__version__, "seed": seed,\n'
                           '            "stream_id": stream_for(stage, kind, qubits, index), "stream_method": "jumped",')
        text = replacement(text, '"seed": expected_seed, "distribution": "uniform",',
                           '"seed": expected_seed, "distribution": "uniform",\n'
                           '                "stream_id": stream_for(spec["ensemble_stage"], kind, qubits, spec["ensemble_index"]),\n'
                           '                "stream_method": "jumped",')
        text = replacement(text,
                           'expected = {(kind, qubits, seed) for (kind, qubits), seeds in schedule.items() for seed in seeds}',
                           'expected = {(kind, qubits, index) for (kind, qubits), seeds in schedule.items() for index in range(len(seeds))}')
        text = replacement(text,
                           'observed = {(spec.get("kind"), spec.get("qubits"), spec.get("seed")) for spec in circuits}',
                           'observed = {(spec.get("kind"), spec.get("qubits"), spec.get("ensemble_index")) for spec in circuits}')
        # These constants identify independent streams, not RNG seeds.
        text = text.replace("SEED_BASES", "STREAM_BASES").replace("SEEDS_PER_SCOPE", "STREAMS_PER_SCOPE")
        text = text.replace('DESIGN_ID = "independent_uniform_angles_v1"',
                            'DESIGN_ID = "independent_uniform_angles_seed20260912_streams_v1"')
    return text


def seed_workers(directory):
    changes = {}
    for path in sorted(directory.glob("*.py")):
        original = path.read_text(encoding="utf-8")
        updated = seeded_source(original, independent=path.name == "scientific_circuit_inputs.py")
        if updated != original:
            old_hash = digest(path)
            path.write_text(updated, encoding="utf-8", newline="\n")
            changes[path.name] = {"original_sha256": old_hash, "adapted_sha256": digest(path)}
    write_new(directory / "seed-adaptation.json", {"master_seed": MASTER_SEED,
        "policy": SEED_POLICY, "sources": changes, "scope": "fresh run copies only"})


def reseed_manifest(manifest, engine_dir):
    manifest = json.loads(json.dumps(manifest))
    for spec in manifest["circuits"]:
        if "seed" in spec:
            old = spec["seed"]
            spec["seed"] = MASTER_SEED
            spec["id"] = spec["id"].replace(f"s{old}", f"s{MASTER_SEED}")
    manifest["seed_policy"] = SEED_POLICY
    module = import_engine(engine_dir, "run_e2_numerical_correctness_pilot.py")
    return module.rendered_manifest(manifest)


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def extract_inputs(data, study, destination):
    """Extract only frozen inputs/workers, checking the archived ZIP digest."""
    info = read(data / "index.json")["studies"][study]
    archive = data / info["archive"]
    if archive.parent.resolve() != data.resolve() or digest(archive) != info["sha256"]:
        raise ValueError("Evidence archive path or SHA-256 mismatch: " + study)
    destination.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(archive) as zipped:
        for name in zipped.namelist():
            if name not in {"contract.json", "manifest.json"} and not name.startswith("runner_snapshot/"):
                continue
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe archive member")
            if name.endswith("/"):
                continue
            payload = zipped.read(name)
            expected = info.get("files", {}).get(name)
            if expected and hashlib.sha256(payload).hexdigest() != expected:
                raise ValueError("Archived file digest mismatch: " + name)
            target = destination / path
            target.parent.mkdir(parents=True, exist_ok=True)
            with target.open("xb") as stream:
                stream.write(payload)
    return {"archive": info["archive"], "sha256": info["sha256"]}


def load_build(path, revision):
    build = dict(path) if isinstance(path, dict) else read(path)
    if build.get("status") != "PASS" or build.get("source_revision") != revision:
        raise ValueError("Build receipt must PASS at the required source revision")
    source, extension = Path(build["source_directory"]), Path(build["extension"])
    # Keep the venv interpreter symlink: resolve() would escape the environment.
    python = Path(os.path.abspath(build["python"]))
    if not source.is_absolute() or not extension.is_absolute() or not python.is_file():
        raise ValueError("Build receipt requires absolute source/extension paths and an existing interpreter")
    extension.resolve().relative_to(source.resolve())
    if digest(extension) != build["extension_sha256"]:
        raise ValueError("Extension changed after build receipt")
    actual = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if actual != revision or dirty:
        raise ValueError("Build source must be clean at its pinned revision")
    build["python"] = str(python)
    return build


def import_engine(directory, name):
    sys.path.insert(0, str(directory))
    return importlib.import_module(name.removesuffix(".py"))


def child_environment(candidate):
    environment = os.environ.copy()
    for key in ("LD_PRELOAD", "LD_AUDIT", "OMP_PLACES"):
        environment.pop(key, None)
    environment.update(PYTHONPATH=candidate["source_directory"], PYTHONNOUSERSITE="1",
                       PYTHONHASHSEED=str(MASTER_SEED),
                       LD_LIBRARY_PATH=":".join(candidate["library_search_paths"]))
    return environment


def launch(config, script, arguments):
    command = [config["python"], str(script), *map(str, arguments)]
    print("Launching", script.name, "— raw results remain in the new run directory.", flush=True)
    subprocess.run(command, env=child_environment(config), check=True)


def prepare(args):
    root = args.run_root.resolve()
    if root.is_relative_to(args.data.resolve()) or root.is_relative_to(Path(__file__).resolve().parent / "data"):
        raise ValueError("--run-root must be outside the immutable article data directory")
    if root.exists():
        raise ValueError("Choose a new --run-root; previous and failed runs are preserved")
    affinity = sorted(int(value) for value in args.affinity.split(","))
    if len(affinity) != 6 or len(set(affinity)) != 6 or not set(affinity) <= os.sched_getaffinity(0):
        raise ValueError("--affinity must select six distinct allowed CPU IDs (see lscpu -e)")
    custom = load_build(args.custom_build, CUSTOM_REVISION)
    upstream = load_build(args.upstream_build, UPSTREAM_REVISION)
    root.mkdir(parents=True)
    origins = {}
    for study in ("cpu-selection", "independent-pilot", "independent-confirmation", "fixed-input"):
        origins[study] = extract_inputs(args.data.resolve(), study, root / ".inputs" / study)
    engine = root / ".inputs/independent-confirmation/runner_snapshot"
    seed_workers(engine)
    template = root / ".inputs/cpu-selection"
    contract = read(template / "contract.json")
    contract["date"] = datetime.now(timezone.utc).date().isoformat()
    manifest = reseed_manifest(read(template / "manifest.json"), engine)
    write_new(root / ".inputs/cpu-inputs.json", manifest)
    contract["manifest"] = str(root / ".inputs/cpu-inputs.json")
    contract["manifest_sha256"] = manifest["manifest_sha256"]
    contract["seed_simulator"] = contract["seed_transpiler"] = MASTER_SEED
    contract["cpu_affinity"] = affinity
    contract["runner_sha256"] = {name: digest(engine / name) for name in contract["runner_sha256"]}
    for candidate in contract["candidates"]:
        build = upstream if candidate["backend"] == "upstream_cpu" else custom
        candidate.update(python=build["python"], pythonpath=build["source_directory"],
                         source_directory=build["source_directory"], source_revision=build["source_revision"],
                         custom_source_directory=custom["source_directory"], extension=build["extension"],
                         extension_sha256=build["extension_sha256"], library_search_paths=build["library_search_paths"])
    module = import_engine(engine, CPU_ENTRY)
    module.validate_contract(contract)
    for candidate in contract["candidates"]:
        module.verify_candidate(candidate)
    write_new(root / "cpu-contract.json", contract)
    write_new(root / "reproduction.json", {"schema_version": 1, "prepared_utc": datetime.now(timezone.utc).isoformat(),
        "master_seed": MASTER_SEED, "seed_policy": SEED_POLICY,
        "custom": custom, "upstream": upstream, "affinity": affinity, "archives": origins,
        "cpu_contract_sha256": digest(root / "cpu-contract.json"),
        "input_inventory": {str(p.relative_to(root)): digest(p) for p in (root / ".inputs").rglob("*")
                            if p.is_file() and "__pycache__" not in p.parts},
        "scope": "fresh measurements; original article observations remain unchanged"})
    print("PREPARED:", root, "(no simulations)")


def validate_inputs(root, config):
    if config.get("master_seed") != MASTER_SEED or config.get("seed_policy") != SEED_POLICY:
        raise ValueError("This run uses another seed policy; prepare a new --run-root for seed 20260912")
    if digest(root / "cpu-contract.json") != config["cpu_contract_sha256"]:
        raise ValueError("Prepared CPU selection contract changed")
    for name, expected in config["input_inventory"].items():
        if digest(root / name) != expected:
            raise ValueError("Frozen collection source/input changed: " + name)


def cpu_selection(root, config):
    engine = root / ".inputs/independent-confirmation/runner_snapshot"
    run = root / "cpu-selection"
    if run.exists():
        raise ValueError("CPU selection requires a new campaign; existing or failed records are preserved")
    try:
        launch(config["custom"], engine / CPU_ENTRY, ["--contract", root / "cpu-contract.json", "--run-dir", run])
    finally:
        # This original worker freezes its contract but not its source directory.
        # Preserve exactly the four sources whose hashes the contract binds.
        if run.exists() and not (run / "runner_snapshot").exists():
            snapshot = run / "runner_snapshot"
            snapshot.mkdir()
            for name in read(root / "cpu-contract.json")["runner_sha256"]:
                shutil.copyfile(engine / name, snapshot / name)


def smoke(root, config):
    """A two-qubit exactness/import check; no timing or offload claim."""
    out = root / "smoke"
    out.mkdir()
    engine = root / ".inputs/independent-confirmation/runner_snapshot"
    module = import_engine(engine, CPU_ENTRY)
    code = '''import json, os, sys
from pathlib import Path
task = json.loads(Path(sys.argv[1]).read_text())
sys.path.insert(0, task["engine"])
from run_e2_numerical_correctness_pilot import rendered_manifest, run_pilot
from run_scientific_cpu_pilot import verify_candidate, runtime_state
candidate = task["candidate"]
verify_candidate(candidate)
os.sched_setaffinity(0, task["affinity"])
manifest = rendered_manifest({"schema_version": 1, "purpose": "Two-qubit installation smoke check",
    "circuits": [{"id": "bell_smoke", "kind": "ghz", "qubits": 2,
                  "exact_reference": True, "offload_required": False}]})
output = Path(task["output"])
events = output.with_suffix(".events.jsonl") if candidate["backend"] == "custom_gpu" else None
report = run_pilot(manifest, candidate["backend"], 1024, 0.0, task["seed"], events,
                   offload_required=False, mps_lapack=True, dispatch_threshold=8401,
                   validate_svd_result=True)
report["runtime"] = runtime_state(candidate)
report["scope"] = "installation and two-qubit exactness only; no SVD offload or speedup proof"
with output.open("x") as stream:
    json.dump(report, stream, indent=2)
if report["status"] != "PASS":
    raise RuntimeError("two-qubit correctness check failed")
print(candidate["id"], "PASS")
'''
    worker = out / "worker.py"
    worker.write_text(code, encoding="utf-8", newline="\n")
    candidates = [c for c in read(root / "cpu-contract.json")["candidates"] if c["mps_lapack"]]
    candidates.append({**candidates[0], "id": "custom_gpu_lapack", "backend": "custom_gpu"})
    for candidate in candidates:
        task = out / (candidate["id"] + "-task.json")
        write_new(task, {"candidate": candidate, "engine": str(engine), "affinity": config["affinity"],
                         "seed": MASTER_SEED,
                         "output": str(out / (candidate["id"] + ".json"))})
        environment = module.controlled_environment(candidate)
        environment["PYTHONHASHSEED"] = str(MASTER_SEED)
        for key in ("LD_PRELOAD", "LD_AUDIT"):
            environment.pop(key, None)
        subprocess.run([candidate["python"], str(worker), str(task)], env=environment, check=True, timeout=120)
    print("SMOKE PASS: CPU/GPU imports and Bell-state correctness. This does not demonstrate SVD offload.")


def paired(root, config, stage):
    engine = root / ".inputs/independent-confirmation/runner_snapshot"
    module = import_engine(engine, PAIRED_ENTRY)
    inputs = import_engine(engine, "scientific_circuit_inputs.py")
    manifest = root / (stage + "-inputs.json")
    count = read(root / "pilot/sample_size_plan.json")["selected_primary_inputs"] if stage == "confirmation" else 16
    inputs.write_scientific_manifest(manifest, stage, primary_confirmation_count=count)
    contract = module.prepare_contract(root / "cpu-selection", manifest,
        order_seed=MASTER_SEED, bootstrap_seed=MASTER_SEED,
        pilot_run_dir=root / "pilot" if stage == "confirmation" else None)
    path = root / (stage + "-contract.json")
    write_new(path, contract)
    launch(config["custom"], engine / PAIRED_ENTRY, ["--contract", path, "--run-dir", root / stage])


def replacement(text, old, new, expected=1):
    if text.count(old) != expected:
        raise ValueError("Archived fixed-input worker changed; portable adaptation requires review: " + old[:70])
    return text.replace(old, new)


def prepare_fixed(root, config):
    """Adapt only original machine bindings; retain recipes/gates/timer/statistics."""
    original = root / ".inputs/fixed-input/runner_snapshot"
    adapted = root / ".fixed-engine"
    adapted.mkdir()
    for source in original.glob("*.py"):
        shutil.copyfile(source, adapted / source.name)
    seed_workers(adapted)
    python = config["custom"]["python"]
    version = subprocess.check_output([python, "-c", "import platform; print(platform.python_version())"], text=True).strip()
    if not version.startswith("3.11."):
        raise ValueError("Fixed-input reproduction requires Python 3.11")
    cpu = next(line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
               if line.startswith("model name"))
    gpu = subprocess.check_output(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                  text=True, timeout=15).strip().splitlines()
    if len(gpu) != 1:
        raise ValueError("Fixed-input protocol currently supports exactly one NVIDIA GPU")
    gpu_name, driver = [value.strip() for value in gpu[0].split(",")]
    source = (original / FIXED_ENTRY).read_text(encoding="utf-8")
    adapted_source = (adapted / FIXED_ENTRY).read_text(encoding="utf-8")
    for old, new, count in [
        ('EXTENSION_SHA = "0a65457e6b783fa4dd46e89e2b0ffc6f4f3899e0a1fdd81c48345f1ed04ae490"',
         'EXTENSION_SHA = ' + repr(config["custom"]["extension_sha256"]), 1),
        ('AFFINITY = [0, 2, 4, 6, 8, 10]', 'AFFINITY = ' + repr(config["affinity"]), 1),
        ('"3.11.15"', repr(version), 3),
        ('"i5-12400F"', repr(cpu), 2), ('"RTX 4060 Ti"', repr(gpu_name), 2),
        ('"591.86"', repr(driver), 2),
        ('baseline.get("stage") == "P2_CONFIRMATION"', 'baseline.get("stage") == "FIXED_INPUT_BUILD_BINDING"', 1),
    ]:
        adapted_source = replacement(adapted_source, old, new, count)
    # Both declared hardware strings and worker checks are frozen to this host.
    adapted_source = '# Portable replication: local bindings; see ../fixed-input-adaptation.json.\n' + adapted_source
    (adapted / FIXED_ENTRY).write_text(adapted_source, encoding="utf-8", newline="\n")
    (root / "fixed-input-adaptation.patch").write_text("".join(difflib.unified_diff(source.splitlines(True),
        adapted_source.splitlines(True), fromfile="archived_worker.py", tofile="local_worker.py")), encoding="utf-8")
    contract = read(root / "cpu-contract.json")
    cpu_candidate = next(c for c in contract["candidates"] if c["id"] == "custom_cpu_lapack")
    baseline = {"schema_version": 1, "stage": "FIXED_INPUT_BUILD_BINDING",
                "runtime_versions": {"qiskit": "2.5.0", "qiskit_aer": "0.17.2", "numpy": "2.4.6", "threadpoolctl": "3.6.0"},
                "candidates": [{**cpu_candidate, "role": "selected_cpu"},
                               {**cpu_candidate, "id": "custom_gpu_lapack", "backend": "custom_gpu", "role": "gpu"}]}
    write_new(root / "fixed-input-build.json", baseline)
    write_new(root / "fixed-input-adaptation.json", {"scope": "new replication of the fixed-input protocol",
        "source_sha256": digest(original / FIXED_ENTRY), "adapted_sha256": digest(adapted / FIXED_ENTRY),
        "master_seed": MASTER_SEED,
        "changes": ["all seeds set to 20260912; new circuit/QASM hashes", "actual local extension hash", "six local CPU IDs", "Python 3.11 patch version",
                    "actual CPU/GPU/driver", "standalone build binding rather than an original campaign identity"],
        "preserved": ["26 circuit topologies", "source revision", "package versions", "correctness gates",
                      "separate diagnostics", "two reversed blocks", "5 warmups / 10 calls / 100 shots",
                      "bond cap 2048, cutoff zero", "median-ratio statistics and descriptive ranges"]})
    launch(config["custom"], adapted / FIXED_ENTRY,
           ["--prepare", "--baseline-contract", root / "fixed-input-build.json", "--run-dir", root / "fixed-input"])


def interpreter_launcher(destination, build, custom_source):
    """Give the original environment-based executor an explicit import binding.

    This is a transparent launcher, not another installed virtual environment.
    CPU correctness must not inherit the custom PYTHONPATH from a GPU worker.
    """
    executable = destination / "bin/python"
    executable.parent.mkdir(parents=True)
    assignments = {"PYTHONPATH": build["source_directory"], "PYTHONNOUSERSITE": "1",
                   "PYTHONHASHSEED": str(MASTER_SEED),
                   "LD_LIBRARY_PATH": ":".join(build["library_search_paths"]),
                   "ARTICLE2_CUSTOM_SOURCE_DIR": custom_source}
    content = "#!/bin/sh\n" + "".join("export " + key + "=" + shlex.quote(value) + "\n"
                                      for key, value in assignments.items())
    content += "exec " + shlex.quote(build["python"]) + ' "$@"\n'
    with executable.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(content)
    executable.chmod(0o755)
    return destination


def historical(root, config, args):
    build = load_build(args.historical_build, HISTORICAL_REVISION)
    upstream = load_build(config["upstream"], UPSTREAM_REVISION)
    cache = args.historical_build.resolve().parent / "cmake/CMakeCache.txt"
    if digest(cache) != build["cmake_cache_sha256"]:
        raise ValueError("Historical CMake cache changed after build receipt")
    archive = args.historical_archive.resolve()
    expected = archive.with_suffix(".sha256").read_text(encoding="utf-8").split()[0]
    if digest(archive) != expected:
        raise ValueError("Historical worker archive digest mismatch")
    scope = args.historical_scope
    work = root / (".historical-" + scope)
    work.mkdir()
    with zipfile.ZipFile(archive) as zipped:
        for info in zipped.infolist():
            name = info.filename
            path = Path(name)
            if path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
                raise ValueError("Unsafe historical archive member")
            if info.is_dir():
                continue
            destination = work / name
            destination.parent.mkdir(parents=True, exist_ok=True)
            with destination.open("xb") as stream:
                stream.write(zipped.read(info))
    seed_workers(work / "experiment/tests/modular")
    custom_launcher = interpreter_launcher(work / "custom-launcher", build, build["source_directory"])
    upstream_launcher = interpreter_launcher(work / "upstream-launcher", upstream, build["source_directory"])
    # A preflight proves both launchers import their own compiled extension.
    check = "from qiskit_aer.backends import controller_wrappers as c; import hashlib,json; " \
            "print(json.dumps({'extension':c.__file__,'sha256':hashlib.sha256(open(c.__file__,'rb').read()).hexdigest()}))"
    imports = {}
    for label, launcher, receipt in (("custom", custom_launcher, build), ("upstream", upstream_launcher, upstream)):
        observed = json.loads(subprocess.check_output([str(launcher / "bin/python"), "-c", check], text=True))
        if Path(observed["extension"]).resolve() != Path(receipt["extension"]).resolve() or observed["sha256"] != receipt["extension_sha256"]:
            raise ValueError("Historical launcher imported an unexpected extension: " + label)
        imports[label] = observed
    write_new(work / "launch-provenance.json", {"archive_sha256": expected, "custom_build": build,
        "master_seed": MASTER_SEED, "seed_policy": SEED_POLICY,
        "upstream_build": upstream, "verified_imports": imports, "scope": scope,
        "thread_policy": "automatic, matching historical measurements; six-core policy belongs to newer studies"})
    environment = os.environ.copy()
    for key in ("LD_PRELOAD", "LD_AUDIT", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "BLIS_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "OMP_DYNAMIC", "OMP_PROC_BIND", "OMP_PLACES"):
        environment.pop(key, None)
    environment["ARTICLE2_CUSTOM_SOURCE_DIR"] = build["source_directory"]
    manifest = "t26_algorithm_crossover_map.json" if scope == "workloads" else "t17_exact_qubit_" + scope.removeprefix("scaling") + "_d24.json"
    experiments = work / "experiment"
    fresh_manifest = reseed_manifest(read(experiments / "manifests" / manifest), experiments / "tests/modular")
    write_new(work / "inputs.json", fresh_manifest)
    command = [str(custom_launcher / "bin/python"), str(experiments / "tests/modular/run_crossover_bundle.py"),
        "--run-id", "historical-" + scope, "--manifest", str(work / "inputs.json"),
        "--runs-root", str(root), "--custom-environment", str(custom_launcher),
        "--upstream-environment", str(upstream_launcher), "--source-dir", build["source_directory"],
        "--cmake-cache", str(cache), "--max-bond-dimension", "1024", "--dispatch-threshold", "8401",
        "--cell-wall-time-seconds", "3600" if scope == "workloads" else "5400",
        "--seed-simulator", str(MASTER_SEED), "--seed-transpiler", str(MASTER_SEED)]
    print("Historical", scope, "uses its original source and automatic thread policy.", flush=True)
    subprocess.run(command, env=environment, check=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "smoke", "cpu-selection", "pilot", "confirmation", "resume", "historical-workloads",
                                          "fixed-prepare", "fixed-probe", "fixed-input", "fixed-resume", "validate"])
    parser.add_argument("--run-root", type=Path, required=True, help="New measurement directory, never an article data directory")
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data")
    parser.add_argument("--custom-build", type=Path, default=Path(".build/custom/backend.json"))
    parser.add_argument("--upstream-build", type=Path, default=Path(".build/upstream/backend.json"))
    parser.add_argument("--affinity", help="For prepare: six distinct CPU IDs, comma separated")
    parser.add_argument("--historical-build", type=Path, default=Path(".build/historical/backend.json"))
    parser.add_argument("--historical-archive", type=Path, default=Path(__file__).resolve().parent / "backend/historical-runners.zip")
    parser.add_argument("--historical-scope", choices=["workloads", "scaling16", "scaling18", "scaling20"], default="workloads")
    parser.add_argument("--study", choices=["cpu-selection", "pilot", "confirmation", "fixed-input"],
                        default="confirmation", help="Study to validate")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Use Linux/WSL for fresh measurements and fresh-run validation")
    try:
        if args.action == "prepare":
            if not args.affinity:
                parser.error("prepare requires --affinity (see lscpu -e)")
            prepare(args)
            return 0
        root = args.run_root.resolve()
        config = read(root / "reproduction.json")
        validate_inputs(root, config)
        if args.action == "smoke":
            smoke(root, config)
        elif args.action == "cpu-selection":
            cpu_selection(root, config)
        elif args.action in {"pilot", "confirmation"}:
            paired(root, config, args.action)
        elif args.action == "resume":
            launch(config["custom"], root / "confirmation/runner_snapshot" / PAIRED_ENTRY,
                   ["--run-dir", root / "confirmation", "--resume"])
        elif args.action == "fixed-prepare":
            prepare_fixed(root, config)
        elif args.action == "historical-workloads":
            historical(root, config, args)
        elif args.action in {"fixed-probe", "fixed-input", "fixed-resume"}:
            action = {"fixed-probe": "--probe", "fixed-input": "--run", "fixed-resume": "--resume"}[args.action]
            launch(config["custom"], root / "fixed-input/runner_snapshot" / FIXED_ENTRY,
                   ["--run-dir", root / "fixed-input", action])
        else:
            run = root / args.study
            entry = CPU_ENTRY if args.study == "cpu-selection" else FIXED_ENTRY if args.study == "fixed-input" else PAIRED_ENTRY
            arguments = ["--run-dir", run, "--validate-only"]
            if args.study == "cpu-selection":
                arguments += ["--contract", run / "contract.json"]
            launch(config["custom"], run / "runner_snapshot" / entry, arguments)
        return 0
    except (OSError, ValueError, RuntimeError, KeyError, subprocess.SubprocessError) as error:
        print("FAILED:", error, file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
