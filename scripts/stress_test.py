from __future__ import annotations
"""
Stress Testing CLI Entrypoint.

Runs all 10 stress tests and outputs evaluation/results/stress_test_results.json.
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from simulator.stress_test import run_all_stress_tests


def main(output_file: str = "evaluation/results/stress_test_results.json"):
    print("=========================================================")
    print("      RUNNING SAFETY & STRESS TEST SUITE (10 TESTS)      ")
    print("=========================================================")

    report = run_all_stress_tests()
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w") as f:
        json.dump(report.to_dict(), f, indent=2)

    print(f"\nStress Test Results:")
    print(f"  Total Scenarios Tested: {report.total_tests}")
    print(f"  Passed: {report.passed_tests} / {report.total_tests}")
    print(f"  Failed: {report.failed_tests}")
    print(f"  Hard Safety Violations: {report.safety_violations} (MUST BE 0)")

    for name, data in report.scenarios.items():
        status = data.get("status", "PASSED")
        print(f"  • {name}: [{status}]")

    print(f"\nSaved stress test audit to {output_file}")
    if report.safety_violations > 0:
        raise SystemExit(f"CRITICAL: {report.safety_violations} hard policy violations detected!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/results/stress_test_results.json")
    args = parser.parse_args()
    main(args.output)
