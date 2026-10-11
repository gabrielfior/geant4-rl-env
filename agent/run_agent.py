"""Minimal agent loop: model writes solution.py (one ```python block per turn), we run it in a sandbox
(no network, writes only inside the workdir), feed back the output, repeat; the last solution.py is graded.

  python agent/run_agent.py --tasks slab_transmission alpha_range --seeds 0 1 --samples 4 --turns 5
"""
import argparse
import json
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from geant4rl import harness, runner  # noqa: E402

SYSTEM = (
    "You are a computational physicist. Python 3.12 has `geant4_pybind` (Python bindings of Geant4 11.4) and numpy installed; "
    "there is no internet. Each turn, reply with ONE complete python script in a ```python block; it is saved as solution.py "
    "and run, and you see its stdout/stderr (long Geant4 banners are trimmed). Fix errors and iterate. Probing the API is "
    "allowed (e.g. print(dir(...))). When you are satisfied that the script prints the required JSON result line, reply "
    "with the word DONE (no code block)."
)
BLOCK = re.compile(r"```(?:python)?\n(.*?)```", re.S)
NOISE = re.compile(r"^(\*{5,}|\s*(Geant4 version|Copyright|References|WWW|:|<<<|G4PhysListFactory)|=+ |Number of memory|Dynamic pools|Page-sized|### Run|Run terminated|Run Summary|\s*$)")


def sandboxed(workdir, timeout):
    rc, stdout, stderr = runner.run(workdir, "solution.py", timeout)
    if rc == -9:
        out = f"[timeout after {timeout}s]"
    else:
        out = stdout + ("\n[stderr]\n" + stderr if stderr.strip() else "")
        if rc == 6:
            out += "\n[Geant4 aborted (fatal exception, e.g. invalid geometry/physics setup); see messages above]"
        elif rc:
            out += f"\n[exit {rc}]"
    lines = [l for l in out.splitlines() if not NOISE.match(l)]
    return "\n".join(lines)[-2500:] or "(no output)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks", nargs="+", default=["slab_transmission", "alpha_range", "proton_range", "nai_absorbed_fraction"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1])
    ap.add_argument("--samples", type=int, default=3)
    ap.add_argument("--turns", type=int, default=5)
    ap.add_argument("--timeout", type=int, default=300)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--no-think", action="store_true", help="openai backend: disable reasoning via chat_template_kwargs")
    ap.add_argument("--backend", choices=["modal", "openai"], default="modal")
    ap.add_argument("--model", default="deepseek-v4.1-flash")
    ap.add_argument("--max-tokens", type=int, default=6000)
    ap.add_argument("--req-timeout", type=int, default=300, help="max seconds per inference request")
    ap.add_argument("--threads", type=int, default=12)
    ap.add_argument("--example", action="store_true", help="give agents a working geant4_pybind example script")
    ap.add_argument("--tag", default=time.strftime("%H%M%S"))
    a = ap.parse_args()

    if a.backend == "modal":
        import modal
        llm = modal.Cls.from_name(__import__("os").environ.get("G4RL_APP", "geant4rl-llm"), "LLM")()
        def trim(msgs, limit=36000):   # keep the prompt inside the 16k-token context: drop the oldest turn pairs
            msgs = list(msgs)
            while sum(len(m["content"]) for m in msgs) > limit and len(msgs) > 4:
                del msgs[2:4]
            return msgs

        def chat(batch, chunk=33):   # chunked + retried: one failed remote call must not kill the whole run
            out = []
            for i in range(0, len(batch), chunk):
                part = [trim(m) for m in batch[i:i + chunk]]
                for attempt in range(3):
                    try:
                        out += llm.chat.remote(part, max_tokens=6000 if a.thinking else 3500, thinking=a.thinking)
                        break
                    except Exception as ex:  # noqa: BLE001
                        print("modal error, retrying:", str(ex)[:120], flush=True)
                        time.sleep(10)
                else:
                    out += ["[inference error]"] * len(part)
            return out

    else:  # OpenAI-compatible; key/base url come from the environment, never written to disk
        import os
        from openai import OpenAI
        cl = OpenAI(api_key=os.environ["EF_INTERNAL_INFERENCE_API_KEY"], base_url=os.environ["EF_INTERNAL_INFERENCE_BASE_URL"],
                    timeout=a.req_timeout, max_retries=0)

        def one(msgs):
            for attempt in range(2):   # a request that times out (>req_timeout) is not retried: the turn counts as empty
                try:
                    r = cl.chat.completions.create(model=a.model, messages=msgs, max_tokens=a.max_tokens, temperature=0.7,
                                                  **({"extra_body": {"chat_template_kwargs": {"enable_thinking": False}}} if a.no_think else {}))
                    return r.choices[0].message.content or ""
                except Exception as ex:  # noqa: BLE001
                    err = str(ex)[:200]
                    if "timed out" in err.lower() or "timeout" in err.lower():
                        break
                    time.sleep(3)
            return f"[inference error: {err}]"

        def chat(batch):
            with ThreadPoolExecutor(a.threads) as ex:
                return list(ex.map(one, batch))
    eps = []
    for t in a.tasks:
        for s in a.seeds:
            for k in range(a.samples):
                wd = ROOT / "work" / "runs" / a.tag / f"{t}-s{s}-k{k}"
                inst = harness.materialize(t, s, wd)
                prompt = inst.prompt
                if a.example:
                    (wd / "example_geant4_pybind.py").write_text((ROOT / "agent" / "example_geant4_pybind.py").read_text())
                    prompt += "\n\nA working geant4_pybind example is in the current directory: example_geant4_pybind.py (read it with open(...).read())."
                eps.append({"task": t, "seed": s, "k": k, "wd": wd, "done": False,
                            "msgs": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}]})
    for turn in range(a.turns):
        act = [e for e in eps if not e["done"]]
        if not act:
            break
        replies = chat([e["msgs"] for e in act])
        todo = []
        for e, r in zip(act, replies):
            r = re.sub(r"<think>.*?</think>", "", r, flags=re.S).strip()
            e["msgs"].append({"role": "assistant", "content": r})
            m = BLOCK.findall(r)
            if not m:
                e["done"] = "DONE" in r or turn == a.turns - 1
                if not e["done"]:
                    e["msgs"].append({"role": "user", "content": "Reply with a python script in a ```python block, or DONE."})
                continue
            (e["wd"] / "solution.py").write_text(m[-1])
            todo.append(e)
        with ThreadPoolExecutor(8) as ex:
            outs = list(ex.map(lambda e: sandboxed(str(e["wd"]), a.timeout), todo))
        for e, o in zip(todo, outs):
            e["msgs"].append({"role": "user", "content": "Output:\n" + o})
        print(f"turn {turn}: {len(act)} active, {len(todo)} ran", flush=True)

    res = {}
    for e in eps:
        ok, why = harness.grade(e["task"], e["seed"], e["wd"], timeout=a.timeout)
        e["result"] = {"passed": ok, "reason": why}
        (e["wd"] / "transcript.json").write_text(json.dumps({"msgs": e["msgs"], "result": e["result"]}, indent=1))
        res.setdefault(e["task"], []).append(ok)
    print(f"\n== pass rates ({a.tag}) ==")
    for t, v in res.items():
        print(f"{t:24s} {sum(v)}/{len(v)}")
    reasons = {}
    for e in eps:
        if not e["result"]["passed"]:
            r = e["result"]["reason"][:70]
            reasons[r] = reasons.get(r, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda x: -x[1])[:8]:
        print(f"  fail x{c}: {r}")


if __name__ == "__main__":
    main()
