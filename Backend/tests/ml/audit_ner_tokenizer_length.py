"""
Tokenizer length audit for distilbert-base-cased on NER training data.

This script measures the actual token-length distribution of the genuine
Hugging Face tokenizer on the ner_train.jsonl dataset to inform the choice
of max_length for NER model training.

CRITICAL MEASUREMENT RULES:
- Entity truncation detection uses TOKEN-LEVEL overlap with ordinal coordinate system
- Full sequence and truncated sequence tokens are mapped to common ordinal space
- Overlapping tokens identified in full sequence via offset_mapping
- Token survival determined by prefix preservation: ordinal < surviving_non_special_count
- No raw BatchEncoding index intersection
- Must use genuine distilbert-base-cased tokenizer with use_fast=True
- No stand-in tokenizers, no fabricated measurements
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Dict, Set
import sys


def load_ner_training_data(filepath: str) -> List[dict]:
    """Load NER training records from ner_train.jsonl."""
    records = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line_idx, line in enumerate(f):
            if line.strip():
                try:
                    record = json.loads(line)
                    # Validate no ground-truth leakage
                    if "canonical_id" in record:
                        raise ValueError(f"Record {line_idx} contains canonical_id (should be evaluation-only)")
                    if any(k.startswith("extracted_") for k in record.keys()):
                        raise ValueError(f"Record {line_idx} contains extracted_* field (should be evaluation-only)")
                    records.append(record)
                except json.JSONDecodeError as e:
                    print(f"ERROR: Failed to parse line {line_idx}: {e}")
                    raise
    return records


def get_non_special_tokens(encoding) -> List[int]:
    """
    Get indices of all non-special tokens in the encoding.
    
    Returns list of token indices (in order) that are non-special and not padding.
    
    Parameters
    ----------
    encoding : transformers.BatchEncoding
        Output from tokenizer with return_special_tokens_mask=True
    
    Returns
    -------
    List[int]
        List of token indices that are not special tokens or padding
    """
    special_tokens_mask = encoding.get("special_tokens_mask", [0] * len(encoding["input_ids"]))
    offset_mapping = encoding["offset_mapping"]
    
    non_special_indices = []
    for token_idx, is_special in enumerate(special_tokens_mask):
        if is_special == 1:
            continue
        # Also skip tokens with (0, 0) offset (padding)
        if offset_mapping[token_idx] == (0, 0):
            continue
        non_special_indices.append(token_idx)
    
    return non_special_indices


def get_entity_overlapping_token_ordinals(
    entity_start: int,
    entity_end: int,
    offset_mapping: List[Tuple[int, int]],
    index_to_ordinal: Dict[int, int]
) -> Set[int]:
    """
    Identify which non-special tokens (by ordinal) overlap with the entity span.
    
    A token overlaps with entity [entity_start, entity_end) if:
    token_start < entity_end AND token_end > entity_start
    
    Parameters
    ----------
    entity_start : int
        Character index of entity start (inclusive)
    entity_end : int
        Character index of entity end (exclusive)
    offset_mapping : List[Tuple[int, int]]
        Token character ranges from tokenizer
    index_to_ordinal : Dict[int, int]
        Mapping from token index to ordinal in non-special sequence
    
    Returns
    -------
    Set[int]
        Set of ordinals (in non-special coordinate system) for tokens overlapping entity
    """
    overlapping_ordinals = set()
    
    for token_idx, ordinal in index_to_ordinal.items():
        token_start, token_end = offset_mapping[token_idx]
        
        # Check for overlap: [token_start, token_end) ∩ [entity_start, entity_end)
        if token_start < entity_end and token_end > entity_start:
            overlapping_ordinals.add(ordinal)
    
    return overlapping_ordinals


def classify_entity_token_survival(
    overlapping_entity_ordinals: Set[int],
    surviving_non_special_count: int
) -> str:
    """
    Classify entity based on token survival in truncated sequence.
    
    Because truncation uses prefix preservation, a token with ordinal < surviving_non_special_count
    survives the truncation.
    
    Parameters
    ----------
    overlapping_entity_ordinals : Set[int]
        Ordinals of non-special tokens that overlap the entity in full sequence
    surviving_non_special_count : int
        Number of non-special tokens in truncated sequence
    
    Returns
    -------
    str
        One of: "fully_preserved", "partially_truncated", "completely_truncated"
    """
    if not overlapping_entity_ordinals:
        # No tokens overlap the entity (should not happen with valid entity)
        return "completely_truncated"
    
    # Find which overlapping tokens survive (ordinal < surviving_non_special_count)
    surviving_overlapping = {
        ordinal for ordinal in overlapping_entity_ordinals
        if ordinal < surviving_non_special_count
    }
    
    if len(surviving_overlapping) == 0:
        return "completely_truncated"
    elif len(surviving_overlapping) == len(overlapping_entity_ordinals):
        return "fully_preserved"
    else:
        return "partially_truncated"


def audit_tokenizer_lengths(
    model_name: str = "distilbert-base-cased",
    ner_data_path: str = "app/ner_train.jsonl",
    max_length_candidates: Tuple[int, ...] = (32, 64),
) -> Dict:
    """
    Audit token lengths and entity truncation using ordinal-coordinate token survival.
    
    Parameters
    ----------
    model_name : str
        HuggingFace model ID for tokenizer.
    ner_data_path : str
        Path to ner_train.jsonl.
    max_length_candidates : tuple
        Candidate max_length values to evaluate for entity truncation.
    
    Returns
    -------
    dict
        Audit results including statistics, truncation analysis, examples.
    """
    # Validate HuggingFace access
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("ERROR: transformers library not available. Cannot run audit.")
        print("Install with: pip install transformers")
        sys.exit(1)
    
    print(f"Loading NER training data from {ner_data_path}...")
    records = load_ner_training_data(ner_data_path)
    print(f"✓ Loaded {len(records)} records.")
    
    if len(records) != 5000:
        print(f"⚠️  WARNING: Expected 5,000 records but loaded {len(records)}")
    
    print(f"\nLoading tokenizer: {model_name}")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    except Exception as e:
        print(f"ERROR: Failed to load tokenizer from HuggingFace:")
        print(f"  {e}")
        print(f"\nThis environment does not have access to HuggingFace.")
        print(f"The tokenizer audit cannot be performed without genuine model access.")
        sys.exit(1)
    
    # Validate tokenizer properties
    if not tokenizer.is_fast:
        print(f"ERROR: Tokenizer is not fast (use_fast=True not working)")
        sys.exit(1)
    
    print(f"✓ Tokenizer loaded: {model_name}")
    print(f"  is_fast: {tokenizer.is_fast}")
    print(f"  vocab_size: {tokenizer.vocab_size}")
    
    # =========================================================================
    # MEASURE 1: Natural token lengths (no truncation)
    # =========================================================================
    print(f"\nMeasuring natural token lengths (no truncation)...")
    
    token_lengths = []
    
    for record in records:
        text = record["text"]
        encoding = tokenizer(
            text,
            return_offsets_mapping=True,
            return_special_tokens_mask=True,
        )
        natural_length = len(encoding["input_ids"])
        token_lengths.append(natural_length)
    
    token_lengths = np.array(token_lengths)
    stats = {
        "total_records": len(records),
        "min": int(token_lengths.min()),
        "max": int(token_lengths.max()),
        "mean": float(token_lengths.mean()),
        "median": float(np.median(token_lengths)),
        "std": float(token_lengths.std()),
        "p95": float(np.percentile(token_lengths, 95)),
        "p99": float(np.percentile(token_lengths, 99)),
    }
    
    print(f"✓ Natural token lengths measured:")
    print(f"    Min: {stats['min']}, Max: {stats['max']}, Mean: {stats['mean']:.2f}, Median: {stats['median']:.1f}")
    print(f"    P95: {stats['p95']:.1f}, P99: {stats['p99']:.1f}")
    
    # =========================================================================
    # MEASURE 2: Truncation analysis for each candidate max_length
    # =========================================================================
    print(f"\nAnalyzing truncation for max_length candidates: {max_length_candidates}...")
    
    truncation_analysis = {}
    
    for max_len in max_length_candidates:
        print(f"\n  Processing max_length={max_len}...")
        
        records_requiring_truncation = 0
        records_with_entity_truncation = 0
        entities_fully_preserved_count = 0
        entities_partially_truncated_count = 0
        entities_completely_truncated_count = 0
        truncation_examples = []
        
        # Track all entities for detailed analysis
        all_entity_statuses = {}  # (record_idx, entity_idx) -> status
        
        for record_idx, record in enumerate(records):
            text = record["text"]
            natural_length = token_lengths[record_idx]
            
            # Count records requiring truncation
            if natural_length > max_len:
                records_requiring_truncation += 1
            
            # =========================================================
            # Tokenize WITHOUT truncation (full sequence)
            # =========================================================
            encoding_full = tokenizer(
                text,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            offset_mapping_full = encoding_full["offset_mapping"]
            non_special_indices_full = get_non_special_tokens(encoding_full)
            
            # Create ordinal mapping for full sequence non-special tokens
            index_to_ordinal_full = {
                token_idx: ordinal
                for ordinal, token_idx in enumerate(non_special_indices_full)
            }
            
            # =========================================================
            # Tokenize WITH truncation at this max_length
            # =========================================================
            encoding_truncated = tokenizer(
                text,
                max_length=max_len,
                truncation=True,
                return_offsets_mapping=True,
                return_special_tokens_mask=True,
            )
            non_special_indices_truncated = get_non_special_tokens(encoding_truncated)
            
            # Count surviving non-special tokens
            surviving_non_special_count = len(non_special_indices_truncated)
            
            # =========================================================
            # ASSERTION: Validate coordinate system consistency
            # =========================================================
            assert surviving_non_special_count <= len(non_special_indices_full), \
                f"Record {record_idx}: Truncation increased non-special token count! " \
                f"Full: {len(non_special_indices_full)}, Truncated: {surviving_non_special_count}"
            
            # For records with natural_length <= max_len, no truncation should occur
            if natural_length <= max_len:
                assert surviving_non_special_count == len(non_special_indices_full), \
                    f"Record {record_idx}: natural_length {natural_length} <= max_length {max_len}, " \
                    f"but surviving_non_special_count ({surviving_non_special_count}) != " \
                    f"full_non_special_count ({len(non_special_indices_full)}). " \
                    f"Coordinate system or truncation logic error."
            
            # Check each gold entity for truncation
            labels = record.get("labels", [])
            record_has_entity_truncation = False
            
            for entity_idx, label_obj in enumerate(labels):
                entity_start = label_obj["start"]
                entity_end = label_obj["end"]
                entity_label = label_obj["label"]
                entity_text = text[entity_start:entity_end]
                
                # Identify overlapping tokens in FULL sequence using ordinal coordinate system
                overlapping_entity_ordinals = get_entity_overlapping_token_ordinals(
                    entity_start, entity_end, offset_mapping_full, index_to_ordinal_full
                )
                
                # Classify based on ordinal survival in truncated sequence
                status = classify_entity_token_survival(
                    overlapping_entity_ordinals, surviving_non_special_count
                )
                
                all_entity_statuses[(record_idx, entity_idx)] = status
                
                if status == "fully_preserved":
                    entities_fully_preserved_count += 1
                elif status == "partially_truncated":
                    entities_partially_truncated_count += 1
                    record_has_entity_truncation = True
                    
                    if len(truncation_examples) < 3:
                        truncation_examples.append({
                            "record_idx": record_idx,
                            "entity_idx": entity_idx,
                            "label": entity_label,
                            "span": entity_text,
                            "entity_interval": (entity_start, entity_end),
                            "natural_length": natural_length,
                            "text": text,
                            "overlapping_ordinals": sorted(list(overlapping_entity_ordinals)),
                            "surviving_non_special_count": surviving_non_special_count,
                            "total_non_special_in_full": len(non_special_indices_full),
                            "status": "partially_truncated",
                        })
                elif status == "completely_truncated":
                    entities_completely_truncated_count += 1
                    record_has_entity_truncation = True
                    
                    if len(truncation_examples) < 3:
                        truncation_examples.append({
                            "record_idx": record_idx,
                            "entity_idx": entity_idx,
                            "label": entity_label,
                            "span": entity_text,
                            "entity_interval": (entity_start, entity_end),
                            "natural_length": natural_length,
                            "text": text,
                            "overlapping_ordinals": sorted(list(overlapping_entity_ordinals)),
                            "surviving_non_special_count": surviving_non_special_count,
                            "total_non_special_in_full": len(non_special_indices_full),
                            "status": "completely_truncated",
                        })
            
            if record_has_entity_truncation:
                records_with_entity_truncation += 1
        
        total_entities_truncated = entities_partially_truncated_count + entities_completely_truncated_count
        
        truncation_analysis[max_len] = {
            "records_requiring_truncation": records_requiring_truncation,
            "pct_records_requiring_truncation": 100.0 * records_requiring_truncation / len(records),
            "records_with_entity_truncation": records_with_entity_truncation,
            "pct_records_with_entity_truncation": 100.0 * records_with_entity_truncation / len(records),
            "entities_fully_preserved": entities_fully_preserved_count,
            "entities_partially_truncated": entities_partially_truncated_count,
            "entities_completely_truncated": entities_completely_truncated_count,
            "entities_truncated": total_entities_truncated,
            "examples": truncation_examples,
            "all_entity_statuses": all_entity_statuses,
        }
        
        print(f"    ✓ Records requiring truncation: {records_requiring_truncation} ({truncation_analysis[max_len]['pct_records_requiring_truncation']:.2f}%)")
        print(f"    ✓ Records with entity truncation: {records_with_entity_truncation} ({truncation_analysis[max_len]['pct_records_with_entity_truncation']:.2f}%)")
        print(f"    ✓ Entities fully preserved: {entities_fully_preserved_count}")
        print(f"    ✓ Entities partially truncated: {entities_partially_truncated_count}")
        print(f"    ✓ Entities completely truncated: {entities_completely_truncated_count}")
        print(f"    ✓ Total entities truncated: {total_entities_truncated}")
    
    # =========================================================================
    # SANITY CHECKS: Full validation across ALL records
    # =========================================================================
    print(f"\n" + "="*100)
    print("SANITY CHECKS: Validating coordinate system and measurement logic")
    print("="*100)
    
    for max_len in max_length_candidates:
        print(f"\nValidating max_length={max_len}...")
        
        all_entity_statuses = truncation_analysis[max_len]["all_entity_statuses"]
        
        # Check 1: For records with natural_length <= max_len, no entities should be truncated
        short_record_errors = []
        for (record_idx, entity_idx), status in all_entity_statuses.items():
            natural_length = token_lengths[record_idx]
            if natural_length <= max_len and status != "fully_preserved":
                short_record_errors.append({
                    "record_idx": record_idx,
                    "entity_idx": entity_idx,
                    "natural_length": natural_length,
                    "status": status,
                })
        
        if short_record_errors:
            print(f"  ✗ CRITICAL ERROR: Found {len(short_record_errors)} entities in short records marked as truncated!")
            print(f"      This indicates a bug in coordinate system or truncation detection.")
            for error in short_record_errors[:5]:
                print(f"        Record {error['record_idx']}, Entity {error['entity_idx']}: " \
                      f"natural_length={error['natural_length']} (≤{max_len}), status={error['status']}")
            sys.exit(1)
        else:
            print(f"  ✓ PASS: All entities in short records (natural_length ≤ {max_len}) are fully preserved")
        
        # Check 2: For records with natural_length > max_len, truncation may or may not occur
        long_record_count = 0
        long_records_with_truncation = 0
        for (record_idx, entity_idx), status in all_entity_statuses.items():
            natural_length = token_lengths[record_idx]
            if natural_length > max_len:
                long_record_count += 1
                if status != "fully_preserved":
                    long_records_with_truncation += 1
        
        if long_record_count > 0:
            pct = 100.0 * long_records_with_truncation / long_record_count
            print(f"  ✓ PASS: Long records (natural_length > {max_len}): {long_record_count} entities, " \
                  f"{long_records_with_truncation} truncated ({pct:.1f}%)")
        else:
            print(f"  ✓ PASS: No records exceed max_length={max_len} (all natural_lengths ≤ {max_len})")
    
    # =========================================================================
    # REPORT
    # =========================================================================
    print("\n" + "="*100)
    print("TOKENIZER AUDIT RESULTS")
    print("="*100)
    
    print(f"\nModel: {model_name}")
    print(f"Data: {len(records)} NER training records from {ner_data_path}")
    print(f"Tokenizer: fast=True, is_fast={tokenizer.is_fast}")
    print(f"\nMeasurement Method: Token-level survival using ordinal coordinate system")
    print(f"  - Full sequence tokens mapped to non-special ordinal space")
    print(f"  - Entity overlapping tokens identified in full sequence ordinals")
    print(f"  - Truncated sequence survivors determined by ordinal < surviving_non_special_count")
    print(f"  - Prefix preservation guarantees ordinal-based classification correctness")
    
    print(f"\nNATURAL TOKEN LENGTH STATISTICS (no truncation):")
    print(f"  Min:     {stats['min']}")
    print(f"  Max:     {stats['max']}")
    print(f"  Mean:    {stats['mean']:.2f}")
    print(f"  Median:  {stats['median']:.1f}")
    print(f"  Std Dev: {stats['std']:.2f}")
    print(f"  P95:     {stats['p95']:.1f}")
    print(f"  P99:     {stats['p99']:.1f}")
    
    print(f"\n" + "-"*100)
    print("TRUNCATION ANALYSIS (ordinal-based token survival)")
    print("-"*100)
    
    print(f"\n{'max_len':<10} {'Req.Trunc':<12} {'%':<8} {'Entity.Trunc':<12} {'%':<8} {'Preserved':<10} {'Partial':<10} {'Complete':<10}")
    print(f"{'-'*10} {'-'*12} {'-'*8} {'-'*12} {'-'*8} {'-'*10} {'-'*10} {'-'*10}")
    
    for max_len in max_length_candidates:
        analysis = truncation_analysis[max_len]
        print(
            f"{max_len:<10} "
            f"{analysis['records_requiring_truncation']:<12} "
            f"{analysis['pct_records_requiring_truncation']:<8.2f} "
            f"{analysis['records_with_entity_truncation']:<12} "
            f"{analysis['pct_records_with_entity_truncation']:<8.2f} "
            f"{analysis['entities_fully_preserved']:<10} "
            f"{analysis['entities_partially_truncated']:<10} "
            f"{analysis['entities_completely_truncated']:<10}"
        )
    
    # Show examples
    print(f"\n" + "-"*100)
    print("ENTITY TRUNCATION EXAMPLES (ordinal-based token survival)")
    print("-"*100)
    
    for max_len in max_length_candidates:
        analysis = truncation_analysis[max_len]
        examples = analysis["examples"]
        
        if examples:
            print(f"\nExamples at max_length={max_len}:")
            for ex_idx, example in enumerate(examples, 1):
                print(f"\n  Example {ex_idx} ({example['status']}):")
                print(f"    Record index: {example['record_idx']}")
                print(f"    Entity index: {example['entity_idx']}")
                print(f"    Natural token length: {example['natural_length']}")
                print(f"    Entity label: {example['label']}")
                print(f"    Entity text: '{example['span']}'")
                print(f"    Entity char interval: {example['entity_interval']}")
                print(f"    Full text (first 80 chars): '{example['text'][:80]}{'...' if len(example['text']) > 80 else ''}'")
                print(f"    Overlapping token ordinals (full sequence): {example['overlapping_ordinals']}")
                print(f"    Surviving non-special tokens (after truncation): {example['surviving_non_special_count']}")
                print(f"    Total non-special in full: {example['total_non_special_in_full']}")
                print(f"    Surviving overlap: {[o for o in example['overlapping_ordinals'] if o < example['surviving_non_special_count']]}")
        else:
            print(f"\nNo entity truncation at max_length={max_len} ✓")
    
    # =========================================================================
    # ANALYSIS SUMMARY
    # =========================================================================
    print(f"\n" + "="*100)
    print("ANALYSIS SUMMARY")
    print("="*100)
    
    analysis_32 = truncation_analysis[32]
    analysis_64 = truncation_analysis[64]
    
    print(f"\nmax_length=32:")
    print(f"  Records requiring truncation: {analysis_32['records_requiring_truncation']} / {len(records)} ({analysis_32['pct_records_requiring_truncation']:.2f}%)")
    print(f"  Records with entity truncation: {analysis_32['records_with_entity_truncation']} ({analysis_32['pct_records_with_entity_truncation']:.2f}%)")
    print(f"  Entities affected: {analysis_32['entities_truncated']} " \
          f"({analysis_32['entities_partially_truncated']} partial + " \
          f"{analysis_32['entities_completely_truncated']} complete)")
    
    print(f"\nmax_length=64:")
    print(f"  Records requiring truncation: {analysis_64['records_requiring_truncation']} / {len(records)} ({analysis_64['pct_records_requiring_truncation']:.2f}%)")
    print(f"  Records with entity truncation: {analysis_64['records_with_entity_truncation']} ({analysis_64['pct_records_with_entity_truncation']:.2f}%)")
    print(f"  Entities affected: {analysis_64['entities_truncated']} " \
          f"({analysis_64['entities_partially_truncated']} partial + " \
          f"{analysis_64['entities_completely_truncated']} complete)")
    
    return {
        "stats": stats,
        "truncation_analysis": truncation_analysis,
    }


if __name__ == "__main__":
    # Default paths
    ner_data_path = "app/ner_train.jsonl"
    
    if not Path(ner_data_path).exists():
        print(f"ERROR: NER data file not found at {ner_data_path}")
        print(f"Current working directory: {Path.cwd()}")
        print(f"Expected to run from Backend/ directory")
        sys.exit(1)
    
    results = audit_tokenizer_lengths(
        model_name="distilbert-base-cased",
        ner_data_path=ner_data_path,
        max_length_candidates=(32, 64),
    )
    
    print("\n" + "="*100)
    print("✓ Audit complete.")
    print("="*100)
