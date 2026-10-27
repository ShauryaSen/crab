"""usage: python -I scripts/plan_from_prompt.py "make a heart, then spell YO" name [n]"""
import json, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from crabshow.prompt import to_plan

prompt, name = sys.argv[1], sys.argv[2]
n = int(sys.argv[3]) if len(sys.argv) > 3 else 72
plan = to_plan(prompt, name, n)
path = Path("plans") / f"{name}.json"
path.write_text(json.dumps(plan, indent=2))
for a in plan["acts"]:
    print(a)
print("wrote", path)
