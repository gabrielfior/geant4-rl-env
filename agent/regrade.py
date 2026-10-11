"""Regrade finished runs from their transcripts: take the result printed by the LAST execution of solution.py in the session
(what the agent saw), parse leniently, grade. Usage: python agent/regrade.py work/runs/<tag> [...]"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from geant4rl.harness import parse_result  # noqa: E402
from geant4rl.tasks import grade_answer, make_instance  # noqa: E402


def regrade(tag_dir):
    res = defaultdict(list)
    for tf in sorted(Path(tag_dir).glob("*/transcript.json")):
        name = tf.parent.name                       # task-s0-k1
        task, rest = name.rsplit("-s", 1)
        seed = int(rest.split("-k")[0])
        d = json.loads(tf.read_text())
        outs = [m["content"] for m in d["msgs"] if m["role"] == "user" and m["content"].startswith("Output:")]
        r = parse_result(outs[-1]) if outs else None
        if r is None:
            ok, why = False, "no result printed by the last run"
        else:
            ok, why = grade_answer(task, make_instance(task, seed).params, r, "Geant4 version")
        res[task].append((ok, why))
    return res


if __name__ == "__main__":
    for td in sys.argv[1:]:
        res = regrade(td)
        print(f"== {td}")
        tot = 0
        for t, v in res.items():
            print(f"{t:26s} {sum(o for o, _ in v)}/{len(v)}")
            tot += sum(o for o, _ in v)
        print(f"{'TOTAL':26s} {tot}/{sum(len(v) for v in res.values())}")
        reasons = defaultdict(int)
        for v in res.values():
            for o, w in v:
                if not o:
                    reasons[w[:70]] += 1
        for w, c in sorted(reasons.items(), key=lambda x: -x[1]):
            print(f"  fail x{c}: {w}")
