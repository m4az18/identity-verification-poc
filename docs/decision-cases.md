# Decision Case Catalogue

This catalogue documents the POC's known operational cases. It is not a claim that every possible real-world identity/fraud scenario is solved; it makes the implemented behavior explicit and testable.

| Case | Evidence interpretation | Final status | User action |
|---|---|---|---|
| Required document checks pass + face match | complete corroborating evidence | `verified` | none |
| Emirates ID number mismatch | critical identity contradiction | `review_required` | manual review |
| DOB mismatch | critical identity contradiction | `review_required` | manual review |
| Emirates ID number/DOB missing | required document evidence unavailable | `retry_required` | re-upload documents |
| No comparable English/Arabic name | required supporting identity evidence unavailable | `retry_required` | re-upload documents |
| Previous MRZ unreadable | previous document validation incomplete | `retry_required` | re-upload previous ID |
| Current MRZ unreadable | current document validation incomplete | `retry_required` | re-upload current ID |
| Previous MRZ check digit still fails after fallback | unresolved document validation concern | `review_required` | re-upload/review previous ID |
| Current MRZ check digit still fails after fallback | unresolved document validation concern | `review_required` | re-upload/review current ID |
| Checksum-protected printed â†” MRZ value conflicts | same-document contradiction | `review_required` | manual review |
| Supporting name/nationality/sex variation | supporting evidence differs | `review_required` | manual review |
| Current ID expired | operational validity failure | `review_required` | manual review / valid current ID |
| Current expiry unreadable | current validity cannot be determined | `retry_required` | re-upload current ID |
| Current photo too blurry/dark/small/multiple faces | live evidence unusable | `retry_required` | recapture current photo |
| Current ID portrait cannot be obtained | document biometric evidence unusable | `retry_required` | re-upload current ID front |
| Usable face pair below threshold | biometric review signal | `review_required` | recapture once + manual review |
| Face model/service unavailable | required specialist service unavailable | `unable_to_verify` | retry service |
| Document mismatch + bad current photo | real contradiction plus quality issue | `review_required` | manual review + recapture photo |
| Document mismatch + face match | face cannot override critical document evidence | `review_required` | manual review |
| High document evidence index + critical mismatch | score cannot override hard gate | `review_required` | manual review |

## Why there is no `rejected` or `fraud` state

The current face threshold is an uncalibrated POC operating point, liveness/PAD is not validated, and the document pipeline verifies consistency rather than full physical-card authenticity. The POC therefore avoids claiming fraud from these signals alone.
