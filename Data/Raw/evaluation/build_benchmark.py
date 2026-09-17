"""
Benchmark Test Case Generator for Samanvay-AI Golden Evaluation Suite.
Generates 150 rigorous test cases:
  - 50 Tier-1 Identical Parity pairs across IOCL, ONGC, and BPCL dialects
  - 50 Tier-2 Valid Upgrades (Pressure upgrade or Stainless/Low-temp metallurgical upgrade)
  - 50 Tier-3 Strict Incompatible Traps (Pressure down-rating, Size mismatch, Metallurgy corrosion downgrade)
"""

import json
from pathlib import Path


def generate_benchmark_suite():
    test_cases = []

    # ---------------------------------------------------------
    # TIER 1: 50 IDENTICAL PAIRS (Cross-CPSE Dialect Variations)
    # ---------------------------------------------------------
    t1_templates = [
        # (item_type, size, rating, alloy, facing, src_dialect, cand_standard)
        ("FLANGE_WELD_NECK", 50, 150, "A105", "RF", 'FLG WNRF 2" 150# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ("FLANGE_WELD_NECK", 100, 300, "A105", "RF", 'FLG WNRF 4IN 300# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ("FLANGE_WELD_NECK", 150, 600, "A105", "RF", 'FLG-WN-DN150-CL600-A105-RF', 'FLANGE, WELD NECK, 6" (DN150), CLASS 600, ASTM A105, RF, ASME B16.5'),
        ("FLANGE_WELD_NECK", 200, 150, "F316", "RF", 'FLANGE WN 8IN 150LBS SS316 RF', 'FLANGE, WELD NECK, 8" (DN200), CLASS 150, ASTM A182 F316, RF, ASME B16.5'),
        ("FLANGE_WELD_NECK", 250, 300, "F304", "RF", 'FLG WNRF 10" 300# SS304', 'FLANGE, WELD NECK, 10" (DN250), CLASS 300, ASTM A182 F304, RF, ASME B16.5'),
        ("FLANGE_BLIND", 50, 150, "A105", "RF", 'FLG BLRF 2" 150# A105', 'FLANGE, BLIND, 2" (DN50), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ("FLANGE_BLIND", 100, 300, "A105", "RF", 'FLANGE BLIND 4IN 300# SA105 RF', 'FLANGE, BLIND, 4" (DN100), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ("FLANGE_BLIND", 150, 600, "A105", "RF", 'FLG-BLD-DN150-CL600-A105-RF', 'FLANGE, BLIND, 6" (DN150), CLASS 600, ASTM A105, RF, ASME B16.5'),
        ("FLANGE_BLIND", 200, 300, "F316", "RF", 'BLIND FLANGE 8" 300# SS316 RF', 'FLANGE, BLIND, 8" (DN200), CLASS 300, ASTM A182 F316, RF, ASME B16.5'),
        ("FLANGE_SLIP_ON", 50, 150, "A105", "RF", 'FLG SORF 2IN 150LBS A105', 'FLANGE, SLIP ON, 2" (DN50), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ("GATE_VALVE", 50, 150, "WCB", "RF", 'VLV-GT-DN50-CL150-WCB-RF', 'VALVE, GATE, 2" (DN50), CLASS 150, ASTM A216 WCB, RF, ASME B16.34'),
        ("GATE_VALVE", 80, 300, "WCB", "RF", 'GATE VALVE 3" 300# WCB FLANGED RF', 'VALVE, GATE, 3" (DN80), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ("GATE_VALVE", 100, 300, "WCB", "RF", 'VALVE GATE 4IN CLASS 300 ASTM A216 WCB RF', 'VALVE, GATE, 4" (DN100), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ("GATE_VALVE", 150, 600, "WCB", "RF", 'VLV GT 6IN 600# WCB RF OS&Y', 'VALVE, GATE, 6" (DN150), CLASS 600, ASTM A216 WCB, RF, ASME B16.34'),
        ("BALL_VALVE", 50, 150, "CF8M", "RF", 'VLV-BL-DN50-CL150-CF8M-RF', 'VALVE, BALL, 2" (DN50), CLASS 150, ASTM A351 CF8M, RF, ASME B16.34'),
        ("BALL_VALVE", 100, 300, "CF8M", "RF", 'BALL VALVE 4" 300# SS316 RF', 'VALVE, BALL, 4" (DN100), CLASS 300, ASTM A351 CF8M, RF, ASME B16.34'),
        ("GLOBE_VALVE", 50, 300, "WCB", "RF", 'VLV GLB 2IN 300# WCB RF', 'VALVE, GLOBE, 2" (DN50), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ("CHECK_VALVE", 100, 300, "WCB", "RF", 'VLV CHK 4IN 300LBS WCB RF', 'VALVE, CHECK, 4" (DN100), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ("PIPE_SEAMLESS", 100, 300, "A106", "BE", 'PIPE SMLS 4IN SCH40 A106-B BE', 'PIPE, SEAMLESS, 4" (DN100), SCH 40, ASTM A106 GR.B, BE'),
        ("PIPE_SEAMLESS", 150, 300, "A106", "BE", 'PIPE SMLS 6" SCH80 ASTM A106 GR B', 'PIPE, SEAMLESS, 6" (DN150), SCH 80, ASTM A106 GR.B, BE'),
    ]

    # Expand to 50 Tier-1 cases with dialect and size permutations
    idx = 1
    for template in t1_templates:
        test_cases.append({
            "test_id": f"TEST-T1-{idx:03d}",
            "expected_tier": "TIER_1_IDENTICAL",
            "source_description": template[5],
            "candidate_description": template[6],
            "expected_compatibility": True,
            "category": "IDENTICAL_PARITY",
            "notes": f"Identical parity test for {template[0]} {template[1]}mm Class {template[2]}",
        })
        idx += 1

    # Permutations with different standard tags and sizes to reach 50
    more_t1 = [
        ('FLG WNRF 3IN 150# A105', 'FLANGE, WELD NECK, 3" (DN80), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ('FLG WNRF 3IN 300# A105', 'FLANGE, WELD NECK, 3" (DN80), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ('FLG WNRF 6IN 150# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ('FLG WNRF 6IN 300# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ('FLG WNRF 8IN 300# A105', 'FLANGE, WELD NECK, 8" (DN200), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ('FLG WNRF 12IN 150# A105', 'FLANGE, WELD NECK, 12" (DN300), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ('FLG WNRF 12IN 300# A105', 'FLANGE, WELD NECK, 12" (DN300), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ('FLG BLRF 3IN 150# A105', 'FLANGE, BLIND, 3" (DN80), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ('FLG BLRF 3IN 300# A105', 'FLANGE, BLIND, 3" (DN80), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ('FLG BLRF 4IN 150# A105', 'FLANGE, BLIND, 4" (DN100), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ('FLG BLRF 8IN 150# A105', 'FLANGE, BLIND, 8" (DN200), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ('FLG BLRF 10IN 150# A105', 'FLANGE, BLIND, 10" (DN250), CLASS 150, ASTM A105, RF, ASME B16.5'),
        ('FLG BLRF 12IN 300# A105', 'FLANGE, BLIND, 12" (DN300), CLASS 300, ASTM A105, RF, ASME B16.5'),
        ('VLV-GT-DN80-CL150-WCB-RF', 'VALVE, GATE, 3" (DN80), CLASS 150, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-GT-DN100-CL150-WCB-RF', 'VALVE, GATE, 4" (DN100), CLASS 150, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-GT-DN150-CL300-WCB-RF', 'VALVE, GATE, 6" (DN150), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-GT-DN200-CL150-WCB-RF', 'VALVE, GATE, 8" (DN200), CLASS 150, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-GT-DN200-CL300-WCB-RF', 'VALVE, GATE, 8" (DN200), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-BL-DN80-CL150-CF8M-RF', 'VALVE, BALL, 3" (DN80), CLASS 150, ASTM A351 CF8M, RF, ASME B16.34'),
        ('VLV-BL-DN150-CL150-CF8M-RF', 'VALVE, BALL, 6" (DN150), CLASS 150, ASTM A351 CF8M, RF, ASME B16.34'),
        ('VLV-BL-DN50-CL300-CF8M-RF', 'VALVE, BALL, 2" (DN50), CLASS 300, ASTM A351 CF8M, RF, ASME B16.34'),
        ('VLV-GLB-DN80-CL300-WCB-RF', 'VALVE, GLOBE, 3" (DN80), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-GLB-DN100-CL150-WCB-RF', 'VALVE, GLOBE, 4" (DN100), CLASS 150, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-GLB-DN150-CL300-WCB-RF', 'VALVE, GLOBE, 6" (DN150), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-CHK-DN50-CL150-WCB-RF', 'VALVE, CHECK, 2" (DN50), CLASS 150, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-CHK-DN80-CL300-WCB-RF', 'VALVE, CHECK, 3" (DN80), CLASS 300, ASTM A216 WCB, RF, ASME B16.34'),
        ('VLV-CHK-DN150-CL150-WCB-RF', 'VALVE, CHECK, 6" (DN150), CLASS 150, ASTM A216 WCB, RF, ASME B16.34'),
        ('PIPE SMLS 2IN SCH40 A106-B', 'PIPE, SEAMLESS, 2" (DN50), SCH 40, ASTM A106 GR.B, BE'),
        ('PIPE SMLS 3IN SCH40 A106-B', 'PIPE, SEAMLESS, 3" (DN80), SCH 40, ASTM A106 GR.B, BE'),
        ('PIPE SMLS 8IN SCH40 A106-B', 'PIPE, SEAMLESS, 8" (DN200), SCH 40, ASTM A106 GR.B, BE'),
    ]
    for src, cand in more_t1[:30]:
        test_cases.append({
            "test_id": f"TEST-T1-{idx:03d}",
            "expected_tier": "TIER_1_IDENTICAL",
            "source_description": src,
            "candidate_description": cand,
            "expected_compatibility": True,
            "category": "IDENTICAL_PARITY",
            "notes": "Exact match across dialect abbreviations",
        })
        idx += 1

    # ---------------------------------------------------------
    # TIER 2: 50 VALID UPGRADES (Pressure / Metallurgy Upgrades)
    # ---------------------------------------------------------
    t2_cases = [
        # Pressure upgrade (150 -> 300, 300 -> 600, 600 -> 900, 900 -> 1500)
        ('FLG WNRF 2IN 150# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 300, ASTM A105, RF', 'Pressure upgrade 150 -> 300'),
        ('FLG WNRF 3IN 150# A105', 'FLANGE, WELD NECK, 3" (DN80), CLASS 300, ASTM A105, RF', 'Pressure upgrade 150 -> 300'),
        ('FLG WNRF 4IN 150# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A105, RF', 'Pressure upgrade 150 -> 300'),
        ('FLG WNRF 6IN 150# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 300, ASTM A105, RF', 'Pressure upgrade 150 -> 300'),
        ('FLG WNRF 8IN 150# A105', 'FLANGE, WELD NECK, 8" (DN200), CLASS 300, ASTM A105, RF', 'Pressure upgrade 150 -> 300'),
        ('FLG WNRF 2IN 300# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 600, ASTM A105, RF', 'Pressure upgrade 300 -> 600'),
        ('FLG WNRF 3IN 300# A105', 'FLANGE, WELD NECK, 3" (DN80), CLASS 600, ASTM A105, RF', 'Pressure upgrade 300 -> 600'),
        ('FLG WNRF 4IN 300# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 600, ASTM A105, RF', 'Pressure upgrade 300 -> 600'),
        ('FLG WNRF 6IN 300# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 600, ASTM A105, RF', 'Pressure upgrade 300 -> 600'),
        ('FLG WNRF 2IN 600# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 900, ASTM A105, RF', 'Pressure upgrade 600 -> 900'),
        ('FLG BLRF 2IN 150# A105', 'FLANGE, BLIND, 2" (DN50), CLASS 300, ASTM A105, RF', 'Blind pressure upgrade 150 -> 300'),
        ('FLG BLRF 4IN 150# A105', 'FLANGE, BLIND, 4" (DN100), CLASS 300, ASTM A105, RF', 'Blind pressure upgrade 150 -> 300'),
        ('FLG BLRF 6IN 150# A105', 'FLANGE, BLIND, 6" (DN150), CLASS 300, ASTM A105, RF', 'Blind pressure upgrade 150 -> 300'),
        ('FLG BLRF 4IN 300# A105', 'FLANGE, BLIND, 4" (DN100), CLASS 600, ASTM A105, RF', 'Blind pressure upgrade 300 -> 600'),
        ('VLV-GT-DN50-CL150-WCB-RF', 'VALVE, GATE, 2" (DN50), CLASS 300, ASTM A216 WCB, RF', 'Valve pressure upgrade 150 -> 300'),
        ('VLV-GT-DN80-CL150-WCB-RF', 'VALVE, GATE, 3" (DN80), CLASS 300, ASTM A216 WCB, RF', 'Valve pressure upgrade 150 -> 300'),
        ('VLV-GT-DN100-CL150-WCB-RF', 'VALVE, GATE, 4" (DN100), CLASS 300, ASTM A216 WCB, RF', 'Valve pressure upgrade 150 -> 300'),
        ('VLV-GT-DN50-CL300-WCB-RF', 'VALVE, GATE, 2" (DN50), CLASS 600, ASTM A216 WCB, RF', 'Valve pressure upgrade 300 -> 600'),
        ('VLV-GT-DN100-CL300-WCB-RF', 'VALVE, GATE, 4" (DN100), CLASS 600, ASTM A216 WCB, RF', 'Valve pressure upgrade 300 -> 600'),
        ('VLV-BL-DN50-CL150-CF8M-RF', 'VALVE, BALL, 2" (DN50), CLASS 300, ASTM A351 CF8M, RF', 'Ball valve pressure upgrade 150 -> 300'),
        # Metallurgy upgrades (A105 -> F304, A105 -> F316, A105 -> LF2, WCB -> CF8M)
        ('FLG WNRF 2IN 150# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A182 F316, RF', 'Metallurgy upgrade CS -> SS316'),
        ('FLG WNRF 3IN 150# A105', 'FLANGE, WELD NECK, 3" (DN80), CLASS 150, ASTM A182 F316, RF', 'Metallurgy upgrade CS -> SS316'),
        ('FLG WNRF 4IN 150# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 150, ASTM A182 F316, RF', 'Metallurgy upgrade CS -> SS316'),
        ('FLG WNRF 6IN 300# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 300, ASTM A182 F316, RF', 'Metallurgy upgrade CS -> SS316'),
        ('FLG WNRF 2IN 150# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A182 F304, RF', 'Metallurgy upgrade CS -> SS304'),
        ('FLG WNRF 4IN 300# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A182 F304, RF', 'Metallurgy upgrade CS -> SS304'),
        ('FLG WNRF 2IN 150# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A350 LF2, RF', 'Metallurgy upgrade CS -> Low Temp LF2'),
        ('FLG WNRF 4IN 300# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A350 LF2, RF', 'Metallurgy upgrade CS -> Low Temp LF2'),
        ('FLG BLRF 2IN 150# A105', 'FLANGE, BLIND, 2" (DN50), CLASS 150, ASTM A182 F316, RF', 'Blind metallurgy upgrade CS -> SS316'),
        ('FLG BLRF 4IN 300# A105', 'FLANGE, BLIND, 4" (DN100), CLASS 300, ASTM A182 F316, RF', 'Blind metallurgy upgrade CS -> SS316'),
        ('FLG BLRF 6IN 150# A105', 'FLANGE, BLIND, 6" (DN150), CLASS 150, ASTM A182 F304, RF', 'Blind metallurgy upgrade CS -> SS304'),
        ('VLV-GT-DN50-CL150-WCB-RF', 'VALVE, GATE, 2" (DN50), CLASS 150, ASTM A351 CF8M, RF', 'Valve metallurgy upgrade WCB -> CF8M'),
        ('VLV-GT-DN80-CL150-WCB-RF', 'VALVE, GATE, 3" (DN80), CLASS 150, ASTM A351 CF8M, RF', 'Valve metallurgy upgrade WCB -> CF8M'),
        ('VLV-GT-DN100-CL300-WCB-RF', 'VALVE, GATE, 4" (DN100), CLASS 300, ASTM A351 CF8M, RF', 'Valve metallurgy upgrade WCB -> CF8M'),
        ('VLV-CHK-DN50-CL150-WCB-RF', 'VALVE, CHECK, 2" (DN50), CLASS 150, ASTM A351 CF8M, RF', 'Check valve metallurgy upgrade WCB -> CF8M'),
        # Dual upgrade: Pressure + Metallurgy upgrade
        ('FLG WNRF 2IN 150# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 300, ASTM A182 F316, RF', 'Dual upgrade: 150->300 & CS->SS316'),
        ('FLG WNRF 4IN 150# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A182 F316, RF', 'Dual upgrade: 150->300 & CS->SS316'),
        ('FLG WNRF 6IN 150# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 300, ASTM A182 F316, RF', 'Dual upgrade: 150->300 & CS->SS316'),
        ('VLV-GT-DN50-CL150-WCB-RF', 'VALVE, GATE, 2" (DN50), CLASS 300, ASTM A351 CF8M, RF', 'Dual upgrade: 150->300 & WCB->CF8M'),
        ('VLV-GT-DN100-CL150-WCB-RF', 'VALVE, GATE, 4" (DN100), CLASS 300, ASTM A351 CF8M, RF', 'Dual upgrade: 150->300 & WCB->CF8M'),
        # Additional safe substitutes across ratings
        ('FLG WNRF 8IN 150# A105', 'FLANGE, WELD NECK, 8" (DN200), CLASS 300, ASTM A182 F316, RF', 'Dual upgrade: 150->300 & CS->SS316'),
        ('FLG WNRF 10IN 150# A105', 'FLANGE, WELD NECK, 10" (DN250), CLASS 300, ASTM A105, RF', 'Pressure upgrade 150 -> 300'),
        ('FLG WNRF 12IN 150# A105', 'FLANGE, WELD NECK, 12" (DN300), CLASS 300, ASTM A105, RF', 'Pressure upgrade 150 -> 300'),
        ('FLG BLRF 8IN 150# A105', 'FLANGE, BLIND, 8" (DN200), CLASS 300, ASTM A105, RF', 'Blind pressure upgrade 150 -> 300'),
        ('FLG BLRF 10IN 150# A105', 'FLANGE, BLIND, 10" (DN250), CLASS 300, ASTM A105, RF', 'Blind pressure upgrade 150 -> 300'),
        ('VLV-GT-DN150-CL150-WCB-RF', 'VALVE, GATE, 6" (DN150), CLASS 300, ASTM A216 WCB, RF', 'Valve pressure upgrade 150 -> 300'),
        ('VLV-GT-DN200-CL150-WCB-RF', 'VALVE, GATE, 8" (DN200), CLASS 300, ASTM A216 WCB, RF', 'Valve pressure upgrade 150 -> 300'),
        ('VLV-BL-DN80-CL150-CF8M-RF', 'VALVE, BALL, 3" (DN80), CLASS 300, ASTM A351 CF8M, RF', 'Ball valve pressure upgrade 150 -> 300'),
        ('VLV-BL-DN100-CL150-CF8M-RF', 'VALVE, BALL, 4" (DN100), CLASS 300, ASTM A351 CF8M, RF', 'Ball valve pressure upgrade 150 -> 300'),
        ('VLV-GLB-DN50-CL150-WCB-RF', 'VALVE, GLOBE, 2" (DN50), CLASS 300, ASTM A216 WCB, RF', 'Globe valve pressure upgrade 150 -> 300'),
    ]

    for idx2, (src, cand, note) in enumerate(t2_cases, start=1):
        test_cases.append({
            "test_id": f"TEST-T2-{idx2:03d}",
            "expected_tier": "TIER_2_SUBSTITUTE",
            "source_description": src,
            "candidate_description": cand,
            "expected_compatibility": True,
            "category": "VALID_FUNCTIONAL_UPGRADE",
            "notes": note,
        })

    # ---------------------------------------------------------
    # TIER 3: 50 INCOMPATIBLE SAFETY TRAPS (Zero False Tolerance)
    # ---------------------------------------------------------
    t3_cases = [
        # PRESSURE DOWN-RATING TRAPS (Catastrophic explosion hazard)
        ('FLG WNRF 2IN 300# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A105, RF', 'Down-rating 300# to 150# (burst hazard)'),
        ('FLG WNRF 3IN 300# A105', 'FLANGE, WELD NECK, 3" (DN80), CLASS 150, ASTM A105, RF', 'Down-rating 300# to 150#'),
        ('FLG WNRF 4IN 300# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 150, ASTM A105, RF', 'Down-rating 300# to 150#'),
        ('FLG WNRF 6IN 300# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 150, ASTM A105, RF', 'Down-rating 300# to 150#'),
        ('FLG WNRF 8IN 300# A105', 'FLANGE, WELD NECK, 8" (DN200), CLASS 150, ASTM A105, RF', 'Down-rating 300# to 150#'),
        ('FLG WNRF 2IN 600# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 300, ASTM A105, RF', 'Down-rating 600# to 300#'),
        ('FLG WNRF 4IN 600# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A105, RF', 'Down-rating 600# to 300#'),
        ('FLG WNRF 6IN 600# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 150, ASTM A105, RF', 'Severe down-rating 600# to 150#'),
        ('FLG WNRF 2IN 900# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 600, ASTM A105, RF', 'Down-rating 900# to 600#'),
        ('FLG WNRF 4IN 1500# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 900, ASTM A105, RF', 'Down-rating 1500# to 900#'),
        ('VLV-GT-DN50-CL300-WCB-RF', 'VALVE, GATE, 2" (DN50), CLASS 150, ASTM A216 WCB, RF', 'Valve down-rating 300# to 150#'),
        ('VLV-GT-DN100-CL300-WCB-RF', 'VALVE, GATE, 4" (DN100), CLASS 150, ASTM A216 WCB, RF', 'Valve down-rating 300# to 150#'),
        ('VLV-GT-DN150-CL600-WCB-RF', 'VALVE, GATE, 6" (DN150), CLASS 300, ASTM A216 WCB, RF', 'Valve down-rating 600# to 300#'),
        ('FLG BLRF 4IN 300# A105', 'FLANGE, BLIND, 4" (DN100), CLASS 150, ASTM A105, RF', 'Blind down-rating 300# to 150#'),
        ('FLG BLRF 6IN 600# A105', 'FLANGE, BLIND, 6" (DN150), CLASS 300, ASTM A105, RF', 'Blind down-rating 600# to 300#'),

        # NOMINAL BORE / SIZE MISMATCHES (Geometric impossibility)
        ('FLG WNRF 2IN 150# A105', 'FLANGE, WELD NECK, 3" (DN80), CLASS 150, ASTM A105, RF', 'Size mismatch: Req 2" (50mm), offered 3" (80mm)'),
        ('FLG WNRF 2IN 150# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 150, ASTM A105, RF', 'Size mismatch: Req 2" (50mm), offered 4" (100mm)'),
        ('FLG WNRF 4IN 300# A105', 'FLANGE, WELD NECK, 6" (DN150), CLASS 300, ASTM A105, RF', 'Size mismatch: Req 4" (100mm), offered 6" (150mm)'),
        ('FLG WNRF 6IN 300# A105', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A105, RF', 'Size mismatch: Req 6" (150mm), offered 4" (100mm)'),
        ('FLG WNRF 8IN 150# A105', 'FLANGE, WELD NECK, 10" (DN250), CLASS 150, ASTM A105, RF', 'Size mismatch: Req 8" (200mm), offered 10" (250mm)'),
        ('FLG BLRF 2IN 150# A105', 'FLANGE, BLIND, 3" (DN80), CLASS 150, ASTM A105, RF', 'Blind size mismatch 2" vs 3"'),
        ('FLG BLRF 4IN 300# A105', 'FLANGE, BLIND, 6" (DN150), CLASS 300, ASTM A105, RF', 'Blind size mismatch 4" vs 6"'),
        ('VLV-GT-DN50-CL150-WCB-RF', 'VALVE, GATE, 3" (DN80), CLASS 150, ASTM A216 WCB, RF', 'Valve size mismatch DN50 vs DN80'),
        ('VLV-GT-DN100-CL300-WCB-RF', 'VALVE, GATE, 6" (DN150), CLASS 300, ASTM A216 WCB, RF', 'Valve size mismatch DN100 vs DN150'),
        ('VLV-BL-DN50-CL150-CF8M-RF', 'VALVE, BALL, 4" (DN100), CLASS 150, ASTM A351 CF8M, RF', 'Ball valve size mismatch 2" vs 4"'),
        ('PIPE SMLS 4IN SCH40 A106-B', 'PIPE, SEAMLESS, 6" (DN150), SCH 40, ASTM A106 GR.B, BE', 'Pipe size mismatch 4" vs 6"'),
        ('PIPE SMLS 2IN SCH40 A106-B', 'PIPE, SEAMLESS, 3" (DN80), SCH 40, ASTM A106 GR.B, BE', 'Pipe size mismatch 2" vs 3"'),
        ('FLG WNRF 10IN 150# A105', 'FLANGE, WELD NECK, 12" (DN300), CLASS 150, ASTM A105, RF', 'Size mismatch 10" vs 12"'),
        ('FLG WNRF 12IN 300# A105', 'FLANGE, WELD NECK, 8" (DN200), CLASS 300, ASTM A105, RF', 'Size mismatch 12" vs 8"'),
        ('VLV-GLB-DN50-CL300-WCB-RF', 'VALVE, GLOBE, 4" (DN100), CLASS 300, ASTM A216 WCB, RF', 'Globe valve size mismatch DN50 vs DN100'),

        # METALLURGICAL CORROSION DOWNGRADE TRAPS (Acidic / sour service catastrophe)
        ('FLG WNRF 2IN 150# F316', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A105, RF', 'Severe downgrade: Req SS316, offered Carbon Steel A105'),
        ('FLG WNRF 4IN 300# F316', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A105, RF', 'Severe downgrade: Req SS316, offered Carbon Steel A105'),
        ('FLG WNRF 6IN 150# F316', 'FLANGE, WELD NECK, 6" (DN150), CLASS 150, ASTM A105, RF', 'Severe downgrade: Req SS316, offered Carbon Steel A105'),
        ('FLG WNRF 2IN 150# F304', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A105, RF', 'Downgrade: Req SS304, offered Carbon Steel A105'),
        ('FLG BLRF 4IN 300# F316', 'FLANGE, BLIND, 4" (DN100), CLASS 300, ASTM A105, RF', 'Blind downgrade: Req SS316, offered Carbon Steel A105'),
        ('VLV-BL-DN50-CL150-CF8M-RF', 'VALVE, BALL, 2" (DN50), CLASS 150, ASTM A216 WCB, RF', 'Valve downgrade: Req Stainless CF8M, offered Carbon Steel WCB'),
        ('VLV-BL-DN100-CL300-CF8M-RF', 'VALVE, BALL, 4" (DN100), CLASS 300, ASTM A216 WCB, RF', 'Valve downgrade: Req Stainless CF8M, offered Carbon Steel WCB'),
        ('FLG WNRF 2IN 150# LF2', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A105, RF', 'Low temp downgrade: Req LF2, offered non-low-temp A105'),
        ('FLG WNRF 4IN 300# LF2', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A105, RF', 'Low temp downgrade: Req LF2, offered non-low-temp A105'),
        ('FLG WNRF 4IN 300# F316', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A182 F304, RF', 'Molybdenum corrosion downgrade: Req F316, offered F304'),

        # ITEM TYPE / FACING END INCOMPATIBILITY TRAPS
        ('FLG WNRF 4IN 300# A105', 'VALVE, GATE, 4" (DN100), CLASS 300, ASTM A216 WCB, RF', 'Component type mismatch: Flange vs Gate Valve'),
        ('GATE VALVE 2IN 150# WCB RF', 'VALVE, GLOBE, 2" (DN50), CLASS 150, ASTM A216 WCB, RF', 'Valve type mismatch: Gate Valve vs Globe Valve'),
        ('GATE VALVE 4IN 300# WCB RF', 'VALVE, CHECK, 4" (DN100), CLASS 300, ASTM A216 WCB, RF', 'Valve type mismatch: Gate Valve vs Check Valve'),
        ('FLANGE WN 4IN 300# A105 RTJ', 'FLANGE, WELD NECK, 4" (DN100), CLASS 300, ASTM A105, FF', 'Facing mismatch: Ring Type Joint (RTJ) vs Flat Face (FF)'),
        ('FLANGE WN 2IN 600# A105 RTJ', 'FLANGE, WELD NECK, 2" (DN50), CLASS 600, ASTM A105, RF', 'Facing mismatch: High-pressure RTJ vs RF'),
        ('PIPE SMLS 4IN SCH40 A106-B', 'FLANGE, WELD NECK, 4" (DN100), CLASS 150, ASTM A105, RF', 'Component type mismatch: Seamless Pipe vs Flange'),
        ('VLV-BL-DN50-CL150-CF8M-RF', 'VALVE, GATE, 2" (DN50), CLASS 150, ASTM A216 WCB, RF', 'Type & metallurgy mismatch: Ball Valve SS vs Gate Valve CS'),
        ('FLG WNRF 8IN 300# A105', 'FLANGE, BLIND, 8" (DN200), CLASS 300, ASTM A105, RF', 'Flange function mismatch: Weld Neck vs Blind Flange'),
        ('FLG BLRF 6IN 300# A105', 'FLANGE, SLIP ON, 6" (DN150), CLASS 300, ASTM A105, RF', 'Flange function mismatch: Blind vs Slip On'),
        ('FLG WNRF 4IN 300# A105', 'FLANGE, WELD NECK, 2" (DN50), CLASS 150, ASTM A105, RF', 'Compound violation: Size mismatch (4" vs 2") AND down-rating (300# to 150#)'),
    ]

    for idx3, (src, cand, note) in enumerate(t3_cases, start=1):
        test_cases.append({
            "test_id": f"TEST-T3-{idx3:03d}",
            "expected_tier": "TIER_3_INCOMPATIBLE",
            "source_description": src,
            "candidate_description": cand,
            "expected_compatibility": False,
            "category": "STRICT_SAFETY_VIOLATION",
            "notes": note,
        })

    out_dir = Path("data/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "benchmark_test_cases.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(test_cases, f, indent=2)

    print(f"Successfully generated {len(test_cases)} benchmark test cases into {out_file}")
    t1_count = sum(1 for c in test_cases if c["expected_tier"] == "TIER_1_IDENTICAL")
    t2_count = sum(1 for c in test_cases if c["expected_tier"] == "TIER_2_SUBSTITUTE")
    t3_count = sum(1 for c in test_cases if c["expected_tier"] == "TIER_3_INCOMPATIBLE")
    print(f"Distribution: {t1_count} Tier-1 Identical, {t2_count} Tier-2 Substitute, {t3_count} Tier-3 Incompatible")


if __name__ == "__main__":
    generate_benchmark_suite()
