#!/usr/bin/env python3
"""Build bundled current/historical GPU-MPS or upstream CPU Aer on Linux/WSL."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import sysconfig

REVISIONS = {
    "custom": "a1242579272784330b217085fe11b7e1481922ad",
    "upstream": "51c679814c3a292d0d7c59bb39976bd6ff91f60e",
    "historical": "75137ba7dee76ec864cb723a6c3aa9650fbf4f43",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend", choices=REVISIONS, default="custom")
    parser.add_argument("--source", type=Path, default=root / "backend/aer-history.bundle")
    parser.add_argument("--output", type=Path, help="New build directory; default .build/BACKEND")
    parser.add_argument("--cuda-root", type=Path, default=Path("/usr/local/cuda"))
    parser.add_argument("--cuda-arch", default="89", help="Recorded GPU: 89 (compute capability 8.9)")
    parser.add_argument("--jobs", type=int, default=2)
    parser.add_argument("--check-only", action="store_true", help="Check prerequisites without building")
    args = parser.parse_args()
    if sys.platform != "linux":
        parser.error("Build and new measurements require Linux or WSL2; see SETUP.md")
    if sys.version_info[:2] != (3, 11):
        parser.error("Use Python 3.11 (the recorded version is 3.11.15)")
    if args.jobs < 1 or not re.fullmatch(r"[1-9][0-9]+", args.cuda_arch):
        parser.error("--jobs must be positive; --cuda-arch must be numeric, e.g. 89")
    for executable in ("git", "cmake", "ninja", "ldd"):
        if not shutil.which(executable):
            parser.error(f"Missing {executable}; install native dependencies from SETUP.md")
    source = args.source.resolve()
    destination = (args.output or root / ".build" / args.backend).resolve()
    cuda = args.cuda_root.resolve()
    if not source.exists() or (destination.exists() and not args.check_only):
        parser.error("Source must exist and output must be new; use --output for another build")
    if args.backend != "upstream" and not (cuda / "bin/nvcc").is_file():
        parser.error("CUDA nvcc missing; install CUDA Toolkit 12.9 or set --cuda-root")
    try:
        import pybind11
    except ImportError:
        parser.error("Install requirements-gpu.txt in the active Python environment first")
    include = Path(sysconfig.get_path("include"))
    library = Path(sysconfig.get_config_var("LIBDIR")) / sysconfig.get_config_var("LDLIBRARY")
    if not (include / "Python.h").is_file() or not library.is_file():
        parser.error("Python development headers/shared library missing; see SETUP.md")
    site = Path(sysconfig.get_path("purelib"))
    libraries = [site / "cuquantum/lib", site / "cutensor/lib", cuda / "lib64"]
    if Path("/usr/lib/wsl/lib").is_dir():
        libraries.append(Path("/usr/lib/wsl/lib"))
    libraries = [str(path) for path in libraries if path.is_dir()]
    if args.backend != "upstream" and not (site / "cuquantum/include").is_dir():
        parser.error("cuQuantum headers missing; install requirements-gpu.txt")
    if args.check_only:
        print(json.dumps({"status": "PREREQUISITES_FOUND", "backend": args.backend,
                          "python": sys.version.split()[0], "source": str(source),
                          "cuda_root": str(cuda) if args.backend != "upstream" else None,
                          "library_search_paths": libraries,
                          "note": "No compile, GPU execution or numerical validation performed."}, indent=2))
        return 0
    destination.mkdir(parents=True)
    checkout, build = destination / "source", destination / "cmake"
    record = destination / "backend.json"
    # Preserve the venv executable symlink rather than resolving it out of the venv.
    python = os.path.abspath(sys.executable)
    state = {
        "schema_version": 1, "status": "BUILDING", "backend": args.backend,
        "python": python, "source_revision": REVISIONS[args.backend],
        "source_directory": str(checkout), "library_search_paths": libraries,
        "builder_sha256": sha256(__file__), "commands": [],
    }
    environment = os.environ.copy()
    environment["LD_LIBRARY_PATH"] = ":".join(libraries + [environment.get("LD_LIBRARY_PATH", "")])

    def save():
        record.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")

    def run(command, log_name, env=environment):
        command = [str(item) for item in command]
        state["commands"].append({"argv": command, "log": log_name})
        save()
        print(f"{args.backend}: {log_name}", flush=True)
        with (destination / log_name).open("w", encoding="utf-8") as stream:
            subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, env=env, check=True)

    try:
        run(["git", "clone", "--no-checkout", source, checkout], "clone.log")
        run(["git", "-C", checkout, "checkout", "--detach", REVISIONS[args.backend]], "checkout.log")
        rpath = ";".join(libraries)
        configure = [
            "cmake", "-S", checkout, "-B", build, "-G", "Ninja",
            "-DCMAKE_BUILD_TYPE=Release", "-DDISABLE_CONAN=ON",
            f"-DAER_THRUST_BACKEND={'CPU' if args.backend == 'upstream' else 'CUDA'}",
            f"-DPYTHON_EXECUTABLE={python}", f"-DPython_EXECUTABLE={python}",
            f"-DPYTHON_INCLUDE_DIR={include}", f"-DPYTHON_LIBRARY={library}",
            f"-Dpybind11_DIR={pybind11.get_cmake_dir()}",
            f"-DCMAKE_BUILD_RPATH={rpath}", f"-DCMAKE_INSTALL_RPATH={rpath}",
            f"-DCMAKE_INSTALL_PREFIX={destination / 'install'}",
        ]
        if args.backend != "upstream":
            configure.extend([
                f"-DCMAKE_CUDA_COMPILER={cuda / 'bin/nvcc'}",
                f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
                f"-DAER_CUDA_ARCH={args.cuda_arch[:-1]}.{args.cuda_arch[-1]}",
                f"-DAER_PYTHON_CUDA_ROOT={sys.prefix}",
            ])
        run(configure, "configure.log")
        run(["cmake", "--build", build, "-j", args.jobs], "compile.log")
        extensions = list((build / "qiskit_aer/backends/wrappers").glob("controller_wrappers*.so"))
        if len(extensions) != 1:
            raise RuntimeError(f"Expected one built Aer extension; found {len(extensions)}")
        extension = checkout / "qiskit_aer/backends" / extensions[0].name
        shutil.copy2(extensions[0], extension)
        run(["ldd", extension], "libraries.log")
        if "not found" in (destination / "libraries.log").read_text():
            raise RuntimeError("A shared library is missing; inspect libraries.log")
        import_environment = environment | {"PYTHONPATH": str(checkout)}
        run([python, "-c", "import qiskit_aer; print(qiskit_aer.__file__); "
             "from qiskit_aer import AerSimulator; print(AerSimulator().available_methods())"],
            "import.log", env=import_environment)
        state.update(status="PASS", extension=str(extension), extension_sha256=sha256(extension),
                     cmake_cache_sha256=sha256(build / "CMakeCache.txt"))
        save()
        print(f"Built: {record}")
        return 0
    except (OSError, RuntimeError, subprocess.CalledProcessError) as error:
        state.update(status="FAIL", error=str(error))
        save()
        print(f"Build failed: {error}\nLogs retained in {destination}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
