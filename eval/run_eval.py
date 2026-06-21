"""
Evaluation runner: runs each eval case against the assistant and reports results.
Usage: python -m eval.run_eval
"""

import json
import os
import sys
import time

# Fix Windows console Unicode encoding
if sys.stdout.encoding != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from assistant.agent import DealerAgent

EVAL_PATH = os.path.join(os.path.dirname(__file__), "eval_cases.json")
RESULTS_PATH = os.path.join(os.path.dirname(__file__), "results.json")


def score_response(response: str, expected: dict) -> tuple[bool, list[str]]:
    failures = []
    response_lower = response.lower()

    for phrase in expected.get("must_contain", []):
        if phrase.lower() not in response_lower:
            failures.append(f"Missing: '{phrase}'")

    for phrase in expected.get("must_not_contain", []):
        if phrase.lower() in response_lower:
            failures.append(f"Unexpected: '{phrase}'")

    return len(failures) == 0, failures


def run():
    with open(EVAL_PATH) as f:
        cases = json.load(f)

    results = []
    counts = {"pass": 0, "fail": 0}
    by_type = {}

    print(f"\n{'='*60}")
    print(f"VIKMO Dealer Assistant — Evaluation ({len(cases)} cases)")
    print(f"{'='*60}\n")

    for case in cases:
        agent = DealerAgent()  # fresh agent per case (stateless eval)
        print(f"[{case['id']}] {case['description']}")
        print(f"  Input: {case['input']}")

        try:
            response = agent.chat(case["input"])
            passed, failures = score_response(response, case["expected"])
        except Exception as e:
            err = str(e)
            if "rate_limit_exceeded" in err or "429" in err:
                print(f"  Rate limit hit — waiting 60s before retrying...")
                time.sleep(60)
                try:
                    response = agent.chat(case["input"])
                    passed, failures = score_response(response, case["expected"])
                except Exception as e2:
                    response = f"ERROR: {e2}"
                    passed = False
                    failures = [str(e2)]
            else:
                response = f"ERROR: {e}"
                passed = False
                failures = [str(e)]

        status = "PASS" if passed else "FAIL"
        counts[status.lower()] += 1

        t = case["type"]
        by_type.setdefault(t, {"pass": 0, "fail": 0})
        by_type[t][status.lower()] += 1

        print(f"  Status: {status}")
        if failures:
            print(f"  Issues: {'; '.join(failures)}")
        print(f"  Response: {response[:200]}{'...' if len(response) > 200 else ''}\n")

        results.append({
            "id": case["id"],
            "type": case["type"],
            "description": case["description"],
            "input": case["input"],
            "response": response,
            "passed": passed,
            "failures": failures,
        })

        time.sleep(3)  # avoid rate limits

    # Summary
    total = len(cases)
    print(f"\n{'='*60}")
    print(f"RESULTS: {counts['pass']}/{total} passed ({counts['pass']/total*100:.0f}%)")
    print(f"\nBy type:")
    for t, c in by_type.items():
        t_total = c["pass"] + c["fail"]
        print(f"  {t:15s}: {c['pass']}/{t_total}")
    print(f"{'='*60}\n")

    with open(RESULTS_PATH, "w") as f:
        json.dump({"summary": {"total": total, **counts, "by_type": by_type}, "cases": results}, f, indent=2)

    print(f"Results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run()
