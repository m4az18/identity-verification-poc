# OCR and Verification Evaluation Protocol

The repository contains a complete local POC implementation, but it intentionally does not
claim population-level accuracy from one or a few test cards. Accuracy and threshold claims
must come from a representative authorised evaluation set.

## Engines to compare

Run the same authorised samples through:

1. PaddleOCR PP-OCRv5 mobile Arabic/multilingual â€” current local primary candidate;
2. EasyOCR (`en,ar`) â€” retained local comparison baseline;
3. Nanonets â€” optional external benchmark when organisational approval permits the document
   to be sent to a third-party cloud service.

## Per-field measurements

For every engine, record:

- Emirates ID exact extraction accuracy;
- DOB exact extraction accuracy;
- English-name exact and normalized accuracy;
- Arabic-name exact and normalized accuracy;
- nationality accuracy;
- sex accuracy;
- expiry-date accuracy;
- MRZ reconstruction rate;
- MRZ check-pass rate on correctly captured documents;
- frequency with which MRZ fallback is triggered;
- end-to-end latency and stage-level OCR/MRZ latency;
- failure category such as blur, glare, rotation, scan quality, Arabic text or MRZ confusion.

## Identity-pair measurements

For labelled same-identity and different-identity pairs, measure at minimum:

- true same-identity acceptance/consistency rate;
- false same-identity rate;
- manual-review rate;
- insufficient-data rate;
- which field/rule caused each review;
- impact of MRZ fallback on recovered cases.

No probability threshold should be described as calibrated until these measurements exist.

## Data manifest

Use non-identifying sample IDs in repository-safe metadata, for example:

```text
sample_id,document_variant,engine,id_correct,dob_correct,name_en_correct,name_ar_correct,nationality_correct,sex_correct,mrz_valid,latency_ms,failure_type
```

Do not commit real names, Emirates ID numbers, images or other identity PII in an evaluation
CSV. Sensitive ground truth should remain in an approved private location.

## Latency interpretation

Use the API's timing fields to distinguish:

- document preparation;
- visible OCR;
- field mapping/normalization;
- MRZ processing/fallback;
- internal validation;
- final record matching.

This avoids attributing model warm-up or PDF rendering costs to checksum arithmetic and string
comparison.
