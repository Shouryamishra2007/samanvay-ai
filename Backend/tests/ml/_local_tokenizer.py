"""
_local_tokenizer.py (test helper only -- not part of the ml package)

This sandbox has no network access to huggingface.co, so
`AutoTokenizer.from_pretrained("distilbert-base-cased")` cannot be resolved
here (confirmed: it raises OSError trying to reach the Hub). That is an
environment limitation, not a property of `ner_dataset.py` -- the alignment
code in that module is written entirely against the generic fast-tokenizer
contract (`return_offsets_mapping`, `return_special_tokens_mask`,
`is_fast`) and does not care which vocabulary is loaded.

To exercise that alignment logic with a *real* fast tokenizer object (real
WordPiece subword splitting, real offset_mapping, real special-tokens
handling) without any network access, this helper trains a small WordPiece
tokenizer from scratch on the actual ner_train.jsonl corpus, using the same
pre-tokenization and normalization scheme BERT/DistilBERT use
(`BertNormalizer(lowercase=False)` + `BertPreTokenizer`), then wraps it as a
`transformers.PreTrainedTokenizerFast`. This is NOT the distilbert-base-cased
vocabulary and must never be used for actual training -- it exists solely so
`test_ner_dataset.py` can validate alignment behavior (subword splits,
special-token masking, offset accuracy) against a real fast tokenizer while
offline. When this pipeline runs somewhere with Hub access, swap in
`AutoTokenizer.from_pretrained("distilbert-base-cased", use_fast=True)`
directly -- no change to ner_dataset.py is required.
"""

from __future__ import annotations

from pathlib import Path

from tokenizers import Tokenizer, normalizers, pre_tokenizers, processors
from tokenizers.models import WordPiece
from tokenizers.trainers import WordPieceTrainer
from transformers import PreTrainedTokenizerFast

_SPECIAL_TOKENS = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]


def build_local_fast_tokenizer(training_texts: list[str], vocab_size: int = 4000) -> PreTrainedTokenizerFast:
    """Train a small BERT-style WordPiece tokenizer on `training_texts` and
    return it wrapped as a `PreTrainedTokenizerFast`, ready to use anywhere
    `ner_dataset.py` expects a fast tokenizer."""
    raw_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
    raw_tokenizer.normalizer = normalizers.BertNormalizer(lowercase=False)
    raw_tokenizer.pre_tokenizer = pre_tokenizers.BertPreTokenizer()

    trainer = WordPieceTrainer(vocab_size=vocab_size, special_tokens=_SPECIAL_TOKENS)
    raw_tokenizer.train_from_iterator(training_texts, trainer=trainer)

    cls_id = raw_tokenizer.token_to_id("[CLS]")
    sep_id = raw_tokenizer.token_to_id("[SEP]")
    raw_tokenizer.post_processor = processors.TemplateProcessing(
        single="[CLS] $A [SEP]",
        pair="[CLS] $A [SEP] $B:1 [SEP]:1",
        special_tokens=[("[CLS]", cls_id), ("[SEP]", sep_id)],
    )

    return PreTrainedTokenizerFast(
        tokenizer_object=raw_tokenizer,
        unk_token="[UNK]",
        cls_token="[CLS]",
        sep_token="[SEP]",
        pad_token="[PAD]",
        mask_token="[MASK]",
    )


def build_tokenizer_from_ner_train(ner_train_path: str | Path, vocab_size: int = 4000) -> PreTrainedTokenizerFast:
    """Convenience wrapper: train the stand-in tokenizer directly on the
    texts in ner_train.jsonl, for realistic domain subword splitting."""
    import json

    texts = []
    with Path(ner_train_path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                texts.append(json.loads(line)["text"])
    return build_local_fast_tokenizer(texts, vocab_size=vocab_size)
