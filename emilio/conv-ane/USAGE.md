# Privacy Filter ANE — Usage Guide

## Overview

The Privacy Filter runs OpenAI's [privacy-filter](https://huggingface.co/openai/privacy-filter) model on Apple Silicon via CoreML. It detects and redacts PII (personally identifiable information) from text.

**Model:** `PrivacyFilterANE_128_int8.mlpackage` (1.4 GB, int8 quantized)  
**Performance:** ~71 tok/s on M-series CPU, ~1.8s per 128-token batch  
**Accuracy:** 100% label agreement with HuggingFace reference  

## Prerequisites

```bash
# Python 3.12 (required for coremltools compatibility)
cd emilio/conv-ane
source ../../.venv312/bin/activate

# Dependencies
pip install coremltools transformers torch numpy
```

## Quick Start

```bash
# Run on inline text
python run_privacy_filter.py "My name is John Smith and my email is john@example.com"

# Run on a file
python run_privacy_filter.py --file document.txt

# Pipe from stdin
echo "Call me at 555-123-4567" | python run_privacy_filter.py --stdin
```

## Example

**Input:**
```
John Michael Anderson, born on March 14, 1985, currently resides at 742 Evergreen
Terrace, Springfield, IL 62704. His Social Security number is 123-45-6789, and his
driver's license number is D1234567 issued in Illinois. You can reach him at
(217) 555-8932 or via email at john.anderson85@gmail.com. He works as a senior
analyst at Midwest Financial Corp, located at 200 Market Street, Chicago, IL. His
bank account number is 987654321 at First National Bank, routing number 071000013.
John is married to Emily Anderson, whose maiden name is Carter, and they have two
children, Lucas and Sophie. His health insurance ID is HZP-556677889, and he
recently visited Dr. Lisa Reynolds at Springfield General Hospital for treatment
related to hypertension.
```

**Command:**
```bash
python run_privacy_filter.py "John Michael Anderson, born on March 14, 1985, ..."
```

**Redacted output:**
```
[PRIVATE_PERSON], born on [PRIVATE_DATE], currently resides at [PRIVATE_ADDRESS].
His Social Security number is [ACCOUNT_NUMBER], and his driver's license number is
[ACCOUNT_NUMBER] issued in Illinois. You can reach him at [PRIVATE_PHONE] or via
email at [PRIVATE_EMAIL]. He works as a senior analyst at Midwest Financial Corp,
located at [PRIVATE_ADDRESS]. His bank account number is [ACCOUNT_NUMBER] at First
National Bank, routing number [ACCOUNT_NUMBER]. [PRIVATE_PERSON] is married to
[PRIVATE_PERSON], whose maiden name is [PRIVATE_PERSON], and they have two children,
[PRIVATE_PERSON] and [PRIVATE_PERSON]. His health insurance ID is [ACCOUNT_NUMBER],
and he recently visited [PRIVATE_PERSON] at Springfield General Hospital for
treatment related to hypertension.
```

**Detected entities:**
```
  [private_person]  "John Michael Anderson"
  [private_date]    "March 14, 1985"
  [private_address] "742 Evergreen Terrace, Springfield, IL 62704"
  [account_number]  "123-45-6789"        (SSN)
  [account_number]  "D1234567"           (driver's license)
  [private_phone]   "(217) 555-8932"
  [private_email]   "john.anderson85@gmail.com"
  [private_address] "200 Market Street, Chicago, IL"
  [account_number]  "987654321"          (bank account)
  [account_number]  "071000013"          (routing number)
  [private_person]  "John"
  [private_person]  "Emily Anderson"
  [private_person]  "Carter"
  [private_person]  "Lucas"
  [private_person]  "Sophie"
  [account_number]  "HZP-556677889"      (insurance ID)
  [private_person]  "Dr. Lisa Reynolds"
```

## PII Categories

| Label | Description |
|-------|-------------|
| `private_person` | Names of individuals |
| `private_date` | Dates of birth, etc. |
| `private_address` | Physical addresses |
| `private_phone` | Phone numbers |
| `private_email` | Email addresses |
| `account_number` | SSNs, bank accounts, license numbers, insurance IDs |

## Model Details

- **Architecture:** 8-layer transformer encoder, 640d, 128 sparse MoE experts (top-4), bidirectional banded attention (window=128)
- **Quantization:** int8 linear symmetric (post-training)
- **Sequence length:** 128 tokens (longer texts are processed in chunks)
- **Tokenizer:** BPE from `openai/privacy-filter` checkpoint
- **Labels:** 33-class BIOES scheme (Begin/Inside/Outside/End/Single)

## Building the Model

If you need to rebuild the `.mlpackage` from the HuggingFace checkpoint:

```bash
# Build seq_len=128 int8 model
python privacy_filter_to_ane.py --seq-len 128 --quantize int8
```

Or manually:

```python
import privacy_filter_to_ane as pf

weights = pf.load_weights("/path/to/checkpoint")
model = pf.PrivacyFilterANE(weights, seq_len=128)
# ... trace, convert, quantize (see source)
```

## Performance

| Variant | Size | Throughput | Notes |
|---------|------|-----------|-------|
| fp32 (seq=32) | 2.6 GB | 72 tok/s | Baseline |
| int8 (seq=32) | 1.3 GB | 100 tok/s | Best for short text |
| int8 (seq=128) | 1.4 GB | 71 tok/s | Production model |

First inference after model load takes ~3-7s (JIT compilation). Subsequent calls: ~1.8s per 128-token chunk.
