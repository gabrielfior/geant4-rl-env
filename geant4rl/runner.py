"""Run model-written scripts inside a locked-down Docker container.

The container sees only: the task workdir (rw) and the Geant4 datasets (ro). No network, read-only root fs, non-root user,
all capabilities dropped, memory/cpu/pids limits, none of the host's home directory, tokens, or this repo's answer key.
Geant4 abort() (fatal exceptions) is turned into a quiet exit with status 6."""
import os
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE = "geant4rl-sandbox"
DATA_HOST = ROOT / ".conda" / "share" / "Geant4" / "data"

_PRELUDE = (
    "import ctypes, runpy, sys\n"
    "_l = ctypes.CDLL(None)\n"
    "_l.signal.argtypes = [ctypes.c_int, ctypes.c_void_p]\n"
    "_l.signal(6, ctypes.cast(_l._exit, ctypes.c_void_p).value)   # SIGABRT -> _exit(6)\n"
    "sys.argv = [sys.argv[1]]\n"
    "runpy.run_path(sys.argv[0], run_name='__main__')\n"
)


def _host_env():
    """Host G4*DATA variables: from os.environ, else by sourcing scripts/env.sh (so callers need not remember to)."""
    env = {k: v for k, v in os.environ.items() if k.startswith("G4") and k.endswith("DATA")}
    if not env:
        out = subprocess.run(["bash", "-c", f'source "{ROOT}/scripts/env.sh" && env'], capture_output=True, text=True).stdout
        env = dict(l.split("=", 1) for l in out.splitlines() if l.startswith("G4") and "DATA=" in l.split("=", 1)[0] + "=")
    if not env:
        raise RuntimeError("Geant4 data variables not found; run scripts/env.sh setup")
    return env


def _data_env():
    """G4*DATA variables rewritten to the container mount point."""
    env = []
    for k, v in _host_env().items():
        if str(DATA_HOST) in v:
            env += ["-e", f"{k}={v.replace(str(DATA_HOST), '/g4data')}"]
    if not env:
        raise RuntimeError("no Geant4 dataset variables point into " + str(DATA_HOST))
    return env


def docker_cmd(workdir, script="solution.py", cpus=1, mem="4g"):
    return ["docker", "run", "--rm", "--platform", "linux/amd64", "--network", "none", "--read-only",
            "--tmpfs", "/tmp:rw,size=256m", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
            "--pids-limit", "256", "--memory", mem, "--cpus", str(cpus), "-u", "1000:1000",
            "-v", f"{Path(workdir).resolve()}:/work", "-v", f"{DATA_HOST}:/g4data:ro", "-w", "/work",
            *_data_env(), IMAGE, "python", "-I", "-u", "-c", _PRELUDE, script]


def run(workdir, script="solution.py", timeout=600):
    """-> (returncode, stdout, stderr); returncode -9 on timeout (container is killed)."""
    name = f"g4rl-{os.getpid()}-{abs(hash((str(workdir), script, os.times()[4]))) % 10**8}"
    cmd = docker_cmd(workdir, script)
    cmd[2:2] = ["--name", name]
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired as e:
        subprocess.run(["docker", "kill", name], capture_output=True)
        return -9, (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or ""), "timeout"
