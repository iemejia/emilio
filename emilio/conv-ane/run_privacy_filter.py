#!/usr/bin/env python3
"""
Run the Privacy Filter ANE model on arbitrary text.

Usage:
    python run_privacy_filter.py "Your text here"
    python run_privacy_filter.py --file input.txt
    echo "text" | python run_privacy_filter.py --stdin
"""

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CKPT = "/Users/ismael/.cache/huggingface/hub/models--openai--privacy-filter/snapshots/7ffa9a043d54d1be65afb281eddf0ffbe629385b"
MODEL_PATH = os.path.join(SCRIPT_DIR, "PrivacyFilterANE_128_int8.mlpackage")
SEQ_LEN = 128


def build_rope_tables(max_seq_len):
    """Build RoPE cos/sin tables."""
    sys.path.insert(0, SCRIPT_DIR)
    from validate_privacy_filter import build_rope_tables as _build
    return _build(max_seq_len)


def load_model():
    """Load the CoreML model."""
    import coremltools as ct
    return ct.models.MLModel(MODEL_PATH, compute_units=ct.ComputeUnit.CPU_ONLY)


def detect_pii(text, model, tokenizer, id2label):
    """Detect PII entities in text. Returns list of (start, end, entity_type)."""
    enc = tokenizer(text, return_offsets_mapping=True)
    all_ids = np.array(enc["input_ids"], dtype=np.int32)
    offsets = enc["offset_mapping"]
    total_tokens = len(all_ids)

    rope_cos, rope_sin = build_rope_tables(SEQ_LEN)

    all_preds = []
    for chunk_start in range(0, total_tokens, SEQ_LEN):
        chunk_ids = all_ids[chunk_start : chunk_start + SEQ_LEN]
        real_len = len(chunk_ids)

        padded = np.full((1, SEQ_LEN), 199999, dtype=np.int32)
        padded[0, :real_len] = chunk_ids

        attn_mask = np.full((SEQ_LEN, SEQ_LEN), -10000.0, dtype=np.float16)
        for i in range(SEQ_LEN):
            for j in range(max(0, i - 128), min(SEQ_LEN, i + 129)):
                if j < real_len:
                    attn_mask[i, j] = 0.0

        inputs = {
            "token_ids": padded,
            "rope_cos": rope_cos,
            "rope_sin": rope_sin,
            "attn_mask": attn_mask,
        }
        out = model.predict(inputs)
        logits = out["logits"][0]
        preds = logits[:, :real_len].argmax(axis=0).tolist()
        all_preds.extend(preds)

    # Group contiguous PII tokens into spans
    spans = []
    i = 0
    redact_chars = set()
    for ti, pred in enumerate(all_preds):
        label = id2label.get(str(pred), "O")
        if label != "O":
            start, end = offsets[ti]
            for c in range(start, end):
                redact_chars.add(c)

    i = 0
    while i < len(text):
        if i in redact_chars:
            start = i
            while i < len(text) and i in redact_chars:
                i += 1
            # Find entity type from first token in span
            entity_type = "PII"
            for ti, (ts, te) in enumerate(offsets):
                if ts >= start and ts < i:
                    label = id2label.get(str(all_preds[ti]), "O")
                    if label != "O":
                        entity_type = label[2:] if len(label) > 2 else label
                        break
            spans.append((start, i, entity_type))
        else:
            i += 1

    return spans


def redact(text, spans):
    """Replace PII spans with [TYPE] placeholders."""
    result = []
    prev_end = 0
    for start, end, etype in spans:
        result.append(text[prev_end:start])
        # Preserve leading space before the tag
        span_text = text[start:end]
        if span_text.startswith(" "):
            result.append(" ")
        result.append(f"[{etype.upper()}]")
        prev_end = end
    result.append(text[prev_end:])
    return "".join(result)


def main():
    parser = argparse.ArgumentParser(description="Privacy Filter - PII Detection & Redaction")
    parser.add_argument("text", nargs="?", help="Text to analyze")
    parser.add_argument("--file", "-f", help="Read text from file")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--show-entities", action="store_true", default=True, help="Show detected entities")
    parser.add_argument("--no-redact", action="store_true", help="Only show entities, skip redacted output")
    args = parser.parse_args()

    if args.stdin:
        text = sys.stdin.read()
    elif args.file:
        with open(args.file) as f:
            text = f.read()
    elif args.text:
        text = args.text
    else:
        parser.print_help()
        sys.exit(1)

    text = text.strip()
    if not text:
        print("No text provided.")
        sys.exit(1)

    # Load tokenizer and config
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(CKPT)
    with open(os.path.join(CKPT, "config.json")) as f:
        config = json.load(f)
    id2label = config.get("id2label", {})

    # Load model
    print("Loading model...", file=sys.stderr)
    t0 = time.time()
    model = load_model()
    print(f"Model loaded in {time.time()-t0:.1f}s", file=sys.stderr)

    # Run detection
    t0 = time.time()
    spans = detect_pii(text, model, tokenizer, id2label)
    elapsed = time.time() - t0
    print(f"Inference: {elapsed*1000:.0f}ms ({len(text)} chars)", file=sys.stderr)

    # Output
    if not args.no_redact:
        print("\n--- REDACTED ---")
        print(redact(text, spans))

    if args.show_entities and spans:
        print("\n--- ENTITIES ---")
        for start, end, etype in spans:
            print(f"  [{etype}] \"{text[start:end]}\"")

    if not spans:
        print("\nNo PII detected.")


if __name__ == "__main__":
    main()
