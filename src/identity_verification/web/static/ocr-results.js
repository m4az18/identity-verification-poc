// Document result presentation carried over from main (bc36bbe).
let revealPii = false;
let documentComparisons = [];
const piiToggle = document.querySelector("#pii-toggle");
piiToggle.addEventListener("click", () => {
  revealPii = !revealPii;
  piiToggle.textContent = revealPii ? "Mask document values" : "Reveal document values";
  renderComparisons(documentComparisons);
});

function renderDocumentResult(result) {
  const text = result.text_evidence;
  revealPii = false;
  documentComparisons = text.comparisons || [];
  piiToggle.textContent = "Reveal document values";
  document.querySelector("#document-status").textContent = documentStatus(text);
  document.querySelector("#document-summary").textContent = documentSummary(text);
  document.querySelector("#historical-doc-check").textContent = documentValidationSummary(text.historical);
  document.querySelector("#current-doc-check").textContent = documentValidationSummary(text.current);
  document.querySelector("#current-expiry").textContent = expirySummary(text.current?.internal_checks?.current_expiry_valid);
  document.querySelector("#face-status").textContent = humanStatus(result.face_evidence.status);
  document.querySelector("#ocr-confidence").textContent = averageOcrConfidence(text);
  document.querySelector("#historical-mrz").textContent = mrzSummary(text.historical?.mrz);
  document.querySelector("#current-mrz").textContent = mrzSummary(text.current?.mrz);
  document.querySelector("#document-time").textContent = formatMs(result.timing_ms?.document_pipeline);
  document.querySelector("#face-time").textContent = formatMs(result.timing_ms?.biometric_pipeline);
  const action = primaryAction(result.decision.recommended_actions || [], result.decision.reason_codes || []);
  document.querySelector("#next-action-title").textContent = action.title;
  document.querySelector("#next-action-copy").textContent = action.copy;
  renderComparisons(documentComparisons);
  const reasons = document.querySelector("#reason-list");
  reasons.replaceChildren();
  for (const code of result.decision.reason_codes || []) {
    const item = document.createElement("p");
    item.textContent = reasonLabel(code);
    reasons.append(item);
  }
}

function renderComparisons(comparisons) {
  const body = document.getElementById("comparison-body");
  body.replaceChildren();
  for (const comparison of comparisons) {
    const row = document.createElement("tr");
    row.innerHTML = `<td>${escapeHtml(fieldLabel(comparison.field))}</td><td>${escapeHtml(displayValue(comparison.historical))}</td><td>${escapeHtml(displayValue(comparison.current))}</td><td><span class="status-text ${escapeHtml(comparison.status)}">${escapeHtml(comparisonLabel(comparison))}</span></td>`;
    body.append(row);
  }
}

function displayValue(value) {
  if (!value) return "Not available";
  if (revealPii) return value;
  const compact = String(value).replace(/\s+/g, " ").trim();
  if (compact.length <= 4) return "••••";
  return `${compact.slice(0, 2)}${"•".repeat(Math.min(10, compact.length - 4))}${compact.slice(-2)}`;
}

function documentStatus(text) {
  if (text.status === "consistent") return "Passed";
  if (text.status === "review_required") return "Needs attention";
  return "Needs new document";
}

function documentSummary(text) {
  const criticalMismatch = (text.comparisons || []).some(item => ["id_number", "date_of_birth"].includes(item.field) && item.status === "review");
  if (criticalMismatch) return "A critical identity field differs between the previous and current ID.";
  if (text.status === "consistent") return "The required identity information is consistent across the two IDs.";
  if (text.status === "insufficient_data") return "Required identity information could not be read from both IDs.";
  return "One or more document checks need review.";
}

function primaryAction(actions, reasons) {
  if (reasons.includes("CRITICAL_IDENTITY_MISMATCH")) return { title: "Review the identity mismatch", copy: "The previous and current ID contain different critical identity information. Do not auto-approve this case." };
  if (reasons.includes("DOCUMENT_FACE_NOT_USABLE")) return { title: "Upload a clearer current Emirates ID", copy: "Make sure the ID front and portrait are fully visible, sharp, and not affected by glare." };
  if (reasons.includes("LIVE_FACE_NOT_USABLE")) return { title: "Retake the current photo", copy: "Use one clear face, good lighting, and keep the camera steady." };
  if (reasons.includes("FACE_BELOW_THRESHOLD")) return { title: "Review or recapture the face photo", copy: "The comparison was completed but fell below the POC threshold. Recapture once, then escalate if it remains below threshold." };
  if (reasons.includes("CURRENT_MRZ_UNREADABLE") || reasons.includes("CURRENT_DOCUMENT_EXPIRY_UNREADABLE")) return { title: "Upload a clearer current ID", copy: "The current ID could not be read reliably enough to complete all required checks." };
  if (reasons.includes("HISTORICAL_MRZ_UNREADABLE")) return { title: "Upload a clearer previous ID", copy: "The previous ID could not be read reliably enough to complete all required checks." };
  if (reasons.includes("DOCUMENT_NAME_EVIDENCE_MISSING")) return { title: "Upload clearer ID images", copy: "At least one comparable English or Arabic name is required in addition to the critical identity keys." };
  if (actions.includes("manual_review")) return { title: "Send this case for manual review", copy: "The system found evidence that should not be automatically approved." };
  if (actions.includes("retry_service")) return { title: "Retry the verification", copy: "A required model/service was unavailable during this request." };
  if (actions.includes("reupload_historical_document")) return { title: "Upload a clearer previous ID", copy: "Include both sides with readable text." };
  if (actions.includes("reupload_current_document")) return { title: "Upload a clearer current ID", copy: "Include both sides and a clear portrait." };
  if (actions.includes("reupload_documents")) return { title: "Upload clearer ID documents", copy: "Required information could not be read. Include both sides of each ID." };
  return { title: "No further action required", copy: "All required checks completed successfully for this POC." };
}

function documentValidationSummary(documentResult) {
  const mrz = documentResult?.mrz;
  if (!mrz?.valid_structure) return "Needs clearer upload";
  const values = [mrz.checks?.document_number, mrz.checks?.date_of_birth, mrz.checks?.expiry_date, mrz.checks?.composite].filter(v => v !== null && v !== undefined);
  if (values.some(v => v === false)) return "Needs review";
  return "Passed";
}

function mrzSummary(mrz) {
  if (!mrz?.valid_structure) return "Not validated";
  const checks = mrz.checks || {};
  const values = [checks.document_number, checks.date_of_birth, checks.expiry_date, checks.composite].filter(v => v !== null && v !== undefined);
  if (!values.length) return "Structure read";
  return values.every(Boolean) ? "Validated" : "Check failed";
}

function expirySummary(value) {
  if (value === true) return "Valid";
  if (value === false) return "Expired / invalid";
  return "Not confirmed";
}

function comparisonLabel(item) {
  return { match: "Match", variation: "Needs review", review: "Mismatch", missing: "Not available", not_compared: "Not compared" }[item.status] || humanStatus(item.status);
}

function fieldLabel(value) {
  return { id_number: "Emirates ID number", date_of_birth: "Date of birth", name_en: "English name", name_ar: "Arabic name", nationality: "Nationality", sex: "Sex", expiry_date: "Expiry date" }[value] || humanStatus(value);
}

function reasonLabel(code) {
  const labels = {
    ALL_REQUIRED_EVIDENCE_PASSED: "All required document and face checks passed.",
    CRITICAL_IDENTITY_MISMATCH: "A critical identity field differs between the previous and current ID.",
    DOCUMENT_REQUIRED_FIELDS_MISSING: "A required identity field could not be read from both documents.",
    HISTORICAL_MRZ_UNREADABLE: "The previous ID MRZ could not be reconstructed reliably.",
    CURRENT_MRZ_UNREADABLE: "The current ID MRZ could not be reconstructed reliably.",
    HISTORICAL_MRZ_CHECK_FAILED: "The previous ID MRZ still has a failed check digit after the fallback read.",
    CURRENT_MRZ_CHECK_FAILED: "The current ID MRZ still has a failed check digit after the fallback read.",
    DOCUMENT_INTERNAL_MISMATCH: "A checksum-protected MRZ value conflicts with the printed value on the same ID.",
    SUPPORTING_IDENTITY_VARIATION: "A supporting identity field differs or needs review.",
    DOCUMENT_NAME_EVIDENCE_MISSING: "No comparable English or Arabic name could be read from both documents.",
    DOCUMENT_REVIEW_REQUIRED_OTHER: "The document pipeline reported a review condition that is not safe to auto-approve.",
    CURRENT_DOCUMENT_EXPIRED: "The current Emirates ID is expired.",
    CURRENT_DOCUMENT_EXPIRY_UNREADABLE: "The current ID expiry date could not be validated.",
    FACE_BELOW_THRESHOLD: "The usable face pair fell below the configured POC similarity threshold.",
    LIVE_FACE_NOT_USABLE: "The current photo did not meet the face pipeline's usability requirements.",
    DOCUMENT_FACE_NOT_USABLE: "A usable portrait could not be obtained from the current Emirates ID.",
    FACE_SERVICE_UNAVAILABLE: "The face-verification service was unavailable."
  };
  return labels[code] || humanStatus(code);
}

function averageOcrConfidence(text) {
  const values = [];
  for (const documentResult of [text?.historical, text?.current]) {
    for (const field of Object.values(documentResult?.record || {})) {
      if (field?.confidence != null) values.push(Number(field.confidence));
    }
  }
  if (!values.length) return "Not available";
  return `${Math.round(values.reduce((a, b) => a + b, 0) / values.length * 100)}%`;
}

function humanStatus(value) { return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, c => c.toUpperCase()); }

function escapeHtml(value) { return String(value ?? "").replace(/[&<>'"]/g, char => ({"&":"&amp;","<":"&lt;",">":"&gt;","'":"&#39;",'"':"&quot;"}[char])); }
