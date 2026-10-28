# Evidence Decision Policy

Policy version: `poc-evidence-policy-v2`

## Core principle

The final identity decision uses **hard gates and typed evidence facts**, not a synthetic average of model scores.

OCR confidence, fuzzy-name similarity, MRZ check digits and AuraFace cosine similarity do not have the same statistical meaning. A value such as `0.95 OCR confidence` and `0.68 face similarity` must therefore not be averaged into an invented `81.5% identity confidence`.

## Final states

- `verified` â€” all mandatory document checks pass and face verification returns `match`
- `review_required` â€” usable evidence contains a meaningful contradiction/review signal
- `retry_required` â€” no blocking contradiction exists, but required evidence is missing/unusable
- `unable_to_verify` â€” a required model/service is unavailable

## Precise decision facts

### Document retry conditions

- required Emirates ID number/DOB missing
- no comparable English or Arabic name
- previous/current MRZ cannot be reliably reconstructed
- current expiry cannot be validated

### Document review conditions

- Emirates ID number or DOB differs across previous/current IDs
- MRZ structure exists but a check digit still fails after the available fallback path
- a checksum-protected printed value contradicts the corresponding MRZ value
- a supporting identity field differs/needs review
- current document is expired

### Face retry conditions

- current photo is not usable -> recapture **current photo**
- current-ID portrait is not usable -> re-upload **current Emirates ID**

These are deliberately distinct so the UI tells the user what actually needs to be fixed.

### Face review condition

A usable ID portrait/current-photo pair with similarity below the configured threshold becomes `review_required`, not fraud/rejection. The current threshold is not calibrated for production use.

### Service failure

If the required face model/service is unavailable, the result is `unable_to_verify` even if document evidence is available.

## Precedence

1. Required service failure blocks a complete automated decision.
2. A real contradiction/review signal outranks a simple retry signal.
3. Missing/unusable evidence is never described as a mismatch.
4. Face match never overrides critical document contradiction.
5. High document evidence index never overrides a hard gate.
6. Automatic verification occurs only when all mandatory document checks pass and face status is `match`.

## Document evidence index

The API also exposes a transparent 0â€“100 **Document evidence index** for manager/developer diagnostics.

| Component | Weight | Rationale |
|---|---:|---|
| Critical identity continuity | 40 | Emirates ID number and DOB are the primary historical/current identity keys. |
| MRZ integrity | 20 | Machine-readable structure/check digits provide independent deterministic validation. |
| Printed â†” MRZ consistency | 20 | Checksum-protected same-document cross-checks guard against accepting contradictory sources. |
| Supporting identity continuity | 10 | Names/nationality/sex add support but are more OCR/transliteration-sensitive. |
| Current-document validity | 10 | Expiry is operationally required but is not itself an identity-continuity signal. |

Within critical identity continuity, Emirates ID number contributes 25 points and DOB contributes 15.

This index is **not a probability** and is **not used as an accept/reject threshold**. It exists to make policy evidence strength visible without pretending the weights were learned from labelled genuine/impostor data.

## Representative cases

| Evidence | Result | Why |
|---|---|---|
| all required document checks pass + face match | `verified` | mandatory evidence passed |
| critical ID/DOB mismatch + face match | `review_required` | face cannot override identity contradiction |
| document passes + face below threshold | `review_required` | usable biometric evidence disagrees |
| document passes + bad current photo | `retry_required` | biometric decision was not made |
| document passes + ID portrait not detected | `retry_required` | current ID image must be re-uploaded, not the selfie |
| unreadable current MRZ + face match | `retry_required` | required document validation is incomplete |
| expired current ID + face match | `review_required` | current document validity gate failed |
| face service unavailable | `unable_to_verify` | required evidence family unavailable |

## Future calibrated fusion

A true unified risk/confidence score should only be introduced after representative labelled data exists. At that stage the document and biometric signals can be calibrated against genuine/impostor outcomes and evaluated using appropriate false-accept/false-reject metrics. Until then, the explicit policy is more defensible and auditable.
