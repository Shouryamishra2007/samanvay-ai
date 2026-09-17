"""
Golden Benchmark Evaluation Suite for Samanvay-AI.
Evaluates the complete extraction and deterministic tolerance validation pipeline
against 150 curated cross-CPSE test cases (50 Tier-1, 50 Tier-2, 50 Tier-3 safety traps).
"""

import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend.app.ml.ner_tagger import extract_material_attributes
from backend.app.matching.tolerance import evaluate_compatibility
from backend.app.contracts.matching import EquivalenceTier
from data.evaluation.build_benchmark import generate_benchmark_suite


def run_benchmark():
    benchmark_file = Path("data/evaluation/benchmark_test_cases.json")
    if not benchmark_file.exists():
        print("Benchmark test cases not found. Generating now...")
        generate_benchmark_suite()

    with open(benchmark_file, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    total_cases = len(test_cases)
    print(f"\n=======================================================")
    print(f"   SAMANVAY-AI GOLDEN BENCHMARK EVALUATION SUITE")
    print(f"   Executing {total_cases} curated edge cases across IOCL, ONGC & BPCL")
    print(f"=======================================================\n")

    t1_total = 0
    t1_correct = 0
    t2_total = 0
    t2_correct = 0
    t3_total = 0
    t3_correct = 0

    # Safety-critical counters
    false_positives_in_tier3 = 0  # Incompatible items mistakenly approved as Tier 1 or 2
    safety_violations_caught = 0

    eval_results = []
    latencies = []

    start_all = time.perf_counter()

    for tc in test_cases:
        t_start = time.perf_counter()

        src_desc = tc["source_description"]
        cand_desc = tc["candidate_description"]
        expected_tier = tc["expected_tier"]

        # Step 1: Attribute Extraction via NER normalizer
        src_attrs = extract_material_attributes(src_desc)
        cand_attrs = extract_material_attributes(cand_desc)

        # Step 2: Deterministic ASME / ASTM Compatibility Evaluation
        res = evaluate_compatibility(src_attrs, cand_attrs)

        t_end = time.perf_counter()
        latencies.append((t_end - t_start) * 1000)

        actual_tier = res.tier.value if hasattr(res.tier, "value") else str(res.tier)
        is_tier_match = actual_tier == expected_tier

        if expected_tier == "TIER_1_IDENTICAL":
            t1_total += 1
            if is_tier_match:
                t1_correct += 1
        elif expected_tier == "TIER_2_SUBSTITUTE":
            t2_total += 1
            if is_tier_match:
                t2_correct += 1
        elif expected_tier == "TIER_3_INCOMPATIBLE":
            t3_total += 1
            if is_tier_match:
                t3_correct += 1
                safety_violations_caught += 1
            else:
                # If an incompatible part was classified as Tier 1 or 2, it's a catastrophic false positive
                if actual_tier in ("TIER_1_IDENTICAL", "TIER_2_SUBSTITUTE"):
                    false_positives_in_tier3 += 1

        eval_results.append({
            "test_id": tc["test_id"],
            "expected_tier": expected_tier,
            "actual_tier": actual_tier,
            "match": is_tier_match,
            "violations": [v.rule_name for v in res.violations],
            "rationale": res.rationale,
            "latency_ms": round((t_end - t_start) * 1000, 2),
        })

    total_time = time.perf_counter() - start_all
    avg_latency = sum(latencies) / len(latencies)

    # Metrics
    t1_acc = (t1_correct / t1_total) * 100 if t1_total else 0
    t2_acc = (t2_correct / t2_total) * 100 if t2_total else 0
    t3_acc = (t3_correct / t3_total) * 100 if t3_total else 0
    overall_acc = ((t1_correct + t2_correct + t3_correct) / total_cases) * 100

    # Safety Invariant Precision:
    # Any item classified as compatible MUST be truly compatible.
    # Tier 3 Safety Precision = 100% if false_positives_in_tier3 == 0.
    safety_invariant_precision = 100.0 if false_positives_in_tier3 == 0 else (1.0 - (false_positives_in_tier3 / t3_total)) * 100.0

    print("-------------------------------------------------------")
    print(f"  EVALUATION RESULTS SUMMARY")
    print("-------------------------------------------------------")
    print(f"  Total Test Cases Evaluated : {total_cases}")
    print(f"  Overall Classification Acc : {overall_acc:.2f}%")
    print(f"  Tier-1 Identical Recall    : {t1_acc:.2f}% ({t1_correct}/{t1_total})")
    print(f"  Tier-2 Substitute Recall   : {t2_acc:.2f}% ({t2_correct}/{t2_total})")
    print(f"  Tier-3 Trap Rejection Rate : {t3_acc:.2f}% ({t3_correct}/{t3_total})")
    print(f"  Tier-3 False Positives     : {false_positives_in_tier3} (MUST BE 0)")
    print(f"  ASME Safety Precision      : {safety_invariant_precision:.2f}%")
    print(f"  Average Latency per Item   : {avg_latency:.2f} ms")
    print(f"  Total Benchmark Time       : {total_time:.3f} s")
    print("-------------------------------------------------------")

    report = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_cases": total_cases,
        "overall_accuracy_pct": round(overall_acc, 2),
        "tier_1_identical": {
            "total": t1_total,
            "correct": t1_correct,
            "accuracy_pct": round(t1_acc, 2),
        },
        "tier_2_substitute": {
            "total": t2_total,
            "correct": t2_correct,
            "accuracy_pct": round(t2_acc, 2),
        },
        "tier_3_incompatible": {
            "total": t3_total,
            "rejected_safely": t3_correct,
            "false_positives": false_positives_in_tier3,
            "rejection_rate_pct": round(t3_acc, 2),
            "safety_precision_pct": round(safety_invariant_precision, 2),
        },
        "performance": {
            "avg_latency_ms": round(avg_latency, 2),
            "total_benchmark_time_s": round(total_time, 3),
        },
        "status": "PASSED" if false_positives_in_tier3 == 0 and overall_acc >= 90.0 else "FAILED",
        "detailed_results": eval_results,
    }

    report_file = Path("data/evaluation/benchmark_report.json")
    with open(report_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Saved full benchmark audit report to {report_file}")
    assert false_positives_in_tier3 == 0, f"SAFETY VIOLATION: {false_positives_in_tier3} incompatible items were wrongly approved!"
    print("ALL ASME & ASTM SAFETY INVARIANTS SATISFIED (100% PRECISION)")
    return report


if __name__ == "__main__":
    run_benchmark()
