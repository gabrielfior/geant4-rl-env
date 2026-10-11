"""CLI: materialize a task workdir for an agent, then grade the solution.py it wrote.

  python -m geant4rl.harness spec  --task slab_transmission --seed 3 --workdir work/t1
  python -m geant4rl.harness grade --task slab_transmission --seed 3 --workdir work/t1
"""
import argparse
import json
import sys
from pathlib import Path

from . import runner
from .tasks import TASKS, grade_answer, make_instance


def parse_result(stdout):
    """Last line containing a dict with an "answer" key. Lenient on purpose: JSON or Python-dict repr, optional prefix
    text (we grade the physics, not the formatting)."""
    import ast
    import re
    for line in reversed(stdout.splitlines()):
        m = re.search(r"\{.*[\"']answer[\"'].*\}", line)
        if not m:
            continue
        for loads in (json.loads, ast.literal_eval):
            try:
                d = loads(m.group(0))
                if isinstance(d, dict) and "answer" in d:
                    return d
            except Exception:  # noqa: BLE001
                continue
    return None


def materialize(task, seed, workdir):
    inst = make_instance(task, seed)
    wd = Path(workdir)
    wd.mkdir(parents=True, exist_ok=True)
    (wd / "prompt.md").write_text(inst.prompt + "\n")
    (wd / "spec.json").write_text(json.dumps(inst.public_spec(), indent=2))
    for name, content in (inst.files or {}).items():
        (wd / name).write_text(content)
    return inst


def grade(task, seed, workdir, timeout=600):
    inst = make_instance(task, seed)
    workdir = Path(workdir).resolve()
    sol = workdir / "solution.py"
    if not sol.exists():
        return False, "solution.py missing"
    rc, stdout, stderr = runner.run(workdir, "solution.py", timeout)
    if rc == -9:
        return False, "timeout"
    if rc != 0:
        return False, f"solution.py exited {rc}: {stderr[-300:] or stdout[-300:]}"
    result = parse_result(stdout)
    if result is None:
        return False, "no result object with an \"answer\" key found in stdout"
    return grade_answer(task, inst.params, result, stdout)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["spec", "grade"])
    ap.add_argument("--task", required=True, choices=list(TASKS))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--workdir", required=True)
    a = ap.parse_args()
    if a.cmd == "spec":
        print(materialize(a.task, a.seed, a.workdir).prompt)
    else:
        ok, why = grade(a.task, a.seed, a.workdir)
        print(json.dumps({"passed": ok, "reason": why}))
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
