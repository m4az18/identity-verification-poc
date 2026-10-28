const form = document.querySelector("#verify-form");
const button = document.querySelector("#verify-button");
const statusPanel = document.querySelector("#status-panel");
const resultPanel = document.querySelector("#result-panel");

const FIELD_LABELS = {
  id_number: "Emirates ID number",
  date_of_birth: "Date of birth",
  name_en: "Name (English)",
  name_ar: "Name (Arabic)",
  nationality: "Nationality",
  sex: "Sex",
  expiry_date: "Expiry date",
};

const VERDICTS = {
  consistent: {
    word: "Identity data is consistent",
    why: "The critical older/current identity fields agree and no unresolved document-validation issue requires review.",
  },
  review_required: {
    word: "Needs review",
    why: "At least one extraction, MRZ-validation or supporting-field issue remains unresolved.",
  },
  insufficient_data: {
    word: "Not enough evidence",
    why: "At least one critical identity field could not be obtained from both documents.",
  },
};

const CHECK_LABELS = {
  document_number: "MRZ document-number check",
  date_of_birth: "MRZ date-of-birth check",
  expiry_date: "MRZ expiry-date check",
  composite: "MRZ composite check",
};

const MRZ_METHOD_LABELS = {
  existing_text: "Initial full-page OCR text",
  mrz_crop_tesseract: "Tesseract re-read of the MRZ crop",
  mrz_crop_ocr: "Configured OCR engine re-read of the MRZ crop",
  repaired: "MRZ structural repair candidate",
};

bindSelection("historical-document", "historical-selection");
bindSelection("current-document", "current-selection");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const problem = validateInputs();
  if (problem) {
    setStatus(problem, true);
    return;
  }

  button.disabled = true;
  button.textContent = "Comparing…";
  resultPanel.classList.add("hidden");
  setStatus(
    "Processing visible fields, validating each MRZ, cross-checking printed and machine-readable evidence, then comparing the older and current records.",
    false,
  );

  try {
    const response = await fetch("/api/v1/text/verify", {
      method: "POST",
      body: new FormData(form),
    });
    const body = await response.json();
    if (!response.ok) {
      throw new Error(body.detail || "The comparison could not be completed.");
    }

    render(body);
    statusPanel.classList.add("hidden");
  } catch (error) {
    setStatus(error.message || "The comparison could not be completed.", true);
  } finally {
    button.disabled = false;
    button.textContent = "Compare documents";
  }
});

function bindSelection(inputId, targetId) {
  const input = document.querySelector(`#${inputId}`);
  const target = document.querySelector(`#${targetId}`);
  input.addEventListener("change", () => {
    const names = Array.from(input.files).map((file) => file.name);
    target.textContent = names.length ? names.join(", ") : "Nothing selected";
  });
}

function validateInputs() {
  const inputs = [
    ["historical-document", "older"],
    ["current-document", "current"],
  ];

  for (const [id, role] of inputs) {
    const files = Array.from(document.querySelector(`#${id}`).files);
    if (!files.length) return `Select a file for the ${role} document.`;
    if (files.length > 2) {
      return `The ${role} document accepts one PDF, or two images representing front and back.`;
    }
    if (files.length === 2 && files.some((file) => file.type === "application/pdf")) {
      return `For the ${role} document, upload one PDF containing both pages. Two-file selection is for front/back images.`;
    }
  }
  return null;
}

function setStatus(message, isError) {
  statusPanel.textContent = message;
  statusPanel.classList.remove("hidden", "error");
  if (isError) statusPanel.classList.add("error");
}

function render(result) {
  renderVerdict(result);
  renderComparisons(result);
  renderDocument("#historical-evidence", "Older document", result.historical);
  renderDocument("#current-evidence", "Current document", result.current);
  renderRuntime(result);

  const total = result.timing_ms?.total;
  document.querySelector("#runtime-summary").textContent = total == null
    ? ""
    : `Case ${result.case_id.slice(0, 8)} completed in ${(total / 1000).toFixed(1)} seconds using ${result.historical?.ocr_engine || "the configured OCR engine"}.`;

  // The browser already received this structured response from the API. Showing it here
  // adds auditability without writing another PII-bearing file to disk.
  document.querySelector("#result-json").textContent = JSON.stringify(result, null, 2);

  resultPanel.classList.remove("hidden");
}

function renderVerdict(result) {
  const verdict = VERDICTS[result.status] || { word: result.status, why: "" };
  document.querySelector("#verdict-word").textContent = verdict.word;

  const reasons = (result.reasons || []).filter(Boolean);
  document.querySelector("#verdict-why").textContent = reasons.length
    ? reasons.join(" ")
    : verdict.why;

  document.querySelector("#verdict").dataset.status = result.status;
}

function renderComparisons(result) {
  document.querySelector("#comparison-body").innerHTML = result.comparisons
    .map((item) => {
      const olderField = result.historical?.record?.[item.field];
      const currentField = result.current?.record?.[item.field];
      return `<tr>
        <td>${escapeHtml(FIELD_LABELS[item.field] || item.field)}</td>
        <td>${renderFieldValue(item.historical, olderField)}</td>
        <td>${renderConfidence(olderField)}</td>
        <td>${renderFieldValue(item.current, currentField)}</td>
        <td>${renderConfidence(currentField)}</td>
        <td>${renderOutcome(item)}</td>
      </tr>`;
    })
    .join("");
}

function renderConfidence(field) {
  // A field can have a high-confidence OCR candidate that is deliberately rejected by
  // normalization. Show that state explicitly instead of displaying a confidence next
  // to an empty value with no explanation.
  if (field?.raw && !field?.normalized && field.confidence != null) {
    const percentage = Math.round(field.confidence * 100);
    return `<span class="confidence moderate">${percentage}% — OCR candidate read<br><small>Candidate was not accepted as a valid normalized field value</small></span>`;
  }

  if (!field || field.confidence == null) {
    const source = field?.sources?.[0] || "";
    if (source.startsWith("mrz:")) {
      return `<span class="confidence deterministic">MRZ-derived<br><small>checksum-protected MRZ evidence</small></span>`;
    }
    return `<span class="confidence absent">Not available</span>`;
  }

  const percentage = Math.round(field.confidence * 100);
  let label = "Lower";
  let cls = "low";
  if (field.confidence >= 0.90) {
    label = "High";
    cls = "high";
  } else if (field.confidence >= 0.75) {
    label = "Moderate";
    cls = "moderate";
  }

  return `<span class="confidence ${cls}">${percentage}% — ${label}<br><small>OCR extraction confidence</small></span>`;
}

function renderOutcome(item) {
  if (item.status === "match") {
    return `<span class="outcome match">Exact / normalized match</span>`;
  }
  if (item.status === "variation") {
    const similarity = item.similarity == null ? "" : ` (${item.similarity}% similarity)`;
    return `<span class="outcome variation">Name variation${escapeHtml(similarity)}</span>${renderNote(item.note)}`;
  }
  if (item.status === "review") {
    return `<span class="outcome review">Different after normalization</span>${renderNote(item.note)}`;
  }
  if (item.status === "missing") {
    return `<span class="outcome missing">Not available from both documents</span>`;
  }
  return `<span class="outcome not_compared">Not used for historical identity matching</span>${renderNote(item.note)}`;
}

function renderNote(note) {
  return note ? `<small class="outcome-note">${escapeHtml(note)}</small>` : "";
}

function renderFieldValue(value, field) {
  if (value) return renderValue(value);

  // Preserve traceability when OCR did read a candidate but conservative normalization
  // rejected it. The raw candidate is evidence for debugging/review, not a canonical
  // identity value and therefore is not used by the matcher.
  if (field?.raw) {
    return `<span class="value absent">Not resolved</span><br><small class="raw-candidate">OCR candidate: ${escapeHtml(field.raw)}</small>`;
  }

  return `<span class="value absent">—</span>`;
}

function renderValue(value) {
  if (!value) return `<span class="value absent">—</span>`;
  const isArabic = /[\u0600-\u06FF]/.test(value);
  const attrs = isArabic ? ' dir="rtl" lang="ar"' : "";
  return `<span class="value"${attrs}>${escapeHtml(value)}</span>`;
}

function renderDocument(selector, title, documentResult) {
  const mrz = documentResult.mrz || {};
  const target = document.querySelector(selector);

  const rawMethod = mrz.extraction_method;
  const method = MRZ_METHOD_LABELS[rawMethod] || rawMethod || "Not available";

  const checkNames = Object.keys(CHECK_LABELS);
  const availableChecks = checkNames.filter(
    (name) => mrz?.checks?.[name] !== null && mrz?.checks?.[name] !== undefined,
  );
  const passedChecks = availableChecks.filter((name) => mrz?.checks?.[name] === true);
  const mrzSummary = mrz?.valid_structure
    ? `${passedChecks.length}/${availableChecks.length} available TD1 checks passed`
    : "MRZ could not be reliably reconstructed";

  const checks = mrz?.valid_structure
    ? checkNames.map((name) => renderCheck(name, mrz.checks?.[name])).join("")
    : `<div class="check-detail fail">
         <div class="check fail">Machine-readable zone <span>not reconstructed</span></div>
         <small>The MRZ cannot currently be used as validated supporting evidence.</small>
       </div>`;

  const attempts = renderMrzAttempts(mrz.attempts || []);
  const supportingMrz = renderSupportingMrzFields(mrz);
  const crossChecks = renderCrossChecks(documentResult.internal_checks || {});
  const warnings = renderWarnings(documentResult.warnings || []);
  const seconds = (documentResult.timing_ms?.total || 0) / 1000;

  target.innerHTML = `
    <h4>${escapeHtml(title)}</h4>
    <div class="doc-summary-grid">
      <p><b>Visible OCR engine</b><span>${escapeHtml(documentResult.ocr_engine || "Not available")}</span></p>
      <p><b>Selected MRZ path</b><span>${escapeHtml(method)}</span></p>
      <p><b>MRZ validation</b><span>${escapeHtml(mrzSummary)}</span></p>
      <p><b>Document processing time</b><span>${seconds.toFixed(1)} seconds</span></p>
    </div>

    <div class="doc-section">
      <h5>Final selected MRZ checks</h5>
      <div class="checks">${checks}</div>
    </div>

    <div class="doc-section">
      <h5>MRZ reading attempts</h5>
      ${attempts}
    </div>

    <div class="doc-section">
      <h5>Parsed MRZ identity fields</h5>
      ${supportingMrz}
    </div>

    <div class="doc-section">
      <h5>Printed field ↔ MRZ cross-check</h5>
      ${crossChecks || '<p class="quiet">No validated printed-versus-MRZ cross-check was available.</p>'}
    </div>

    ${warnings}
  `;
}

function renderCheck(name, state) {
  const cls = state === true ? "pass" : state === false ? "fail" : "";
  const word = state === true ? "passed" : state === false ? "failed" : "not available";
  let meaning = "This check could not be evaluated from the available MRZ data.";
  if (state === true) {
    meaning = "The MRZ character sequence covered by this check is mathematically consistent with its stored check digit.";
  } else if (state === false) {
    meaning = "The covered MRZ sequence did not validate. This can indicate an OCR reading error and is not proof of forgery.";
  }

  return `<div class="check-detail ${cls}">
    <div class="check ${cls}">${escapeHtml(CHECK_LABELS[name])}<span>${escapeHtml(word)}</span></div>
    <small>${escapeHtml(meaning)}</small>
  </div>`;
}

function renderMrzAttempts(attempts) {
  if (!attempts.length) {
    return `<p class="quiet">No separate MRZ attempt history was recorded.</p>`;
  }

  return `<div class="attempts">
    ${attempts.map((attempt, index) => {
      const label = MRZ_METHOD_LABELS[attempt.method] || attempt.method;
      const selected = attempt.selected ? '<span class="selected-badge">Selected</span>' : "";
      const linePreview = (attempt.raw_lines || []).map((line) => escapeHtml(line)).join("<br>");
      return `<article class="attempt ${attempt.selected ? "selected" : ""}">
        <div class="attempt-head">
          <b>Attempt ${index + 1}: ${escapeHtml(label)}</b>
          ${selected}
        </div>
        <p>${attempt.passing_checks ?? 0}/4 MRZ checks passed${attempt.valid_structure ? "; TD1 structure valid" : "; TD1 structure unresolved"}.</p>
        ${linePreview ? `<details><summary>Show MRZ lines from this attempt</summary><code class="mrz-lines">${linePreview}</code></details>` : ""}
        ${attempt.error ? `<small class="attempt-error">${escapeHtml(attempt.error)}</small>` : ""}
      </article>`;
    }).join("")}
  </div>`;
}


function renderSupportingMrzFields(mrz) {
  if (!mrz?.valid_structure) {
    return `<p class="quiet">No structured MRZ fields are available because the MRZ could not be reconstructed.</p>`;
  }

  const rows = [
    ["Emirates ID candidate", mrz.emirates_id_candidate, "Hard evidence only when the composite check passes."],
    ["Date of birth (YYMMDD)", mrz.date_of_birth_raw, "Hard evidence only when the DOB check passes."],
    ["Expiry date (YYMMDD)", mrz.expiry_date_raw, "Hard evidence only when the expiry check passes."],
    ["Nationality", mrz.nationality, "Supporting MRZ evidence; nationality has no dedicated TD1 check digit."],
    ["Sex", mrz.sex, "Supporting MRZ evidence; sex has no dedicated TD1 check digit."],
    ["Name", mrz.name, "Supporting MRZ evidence; the TD1 name line is not independently checksum-protected."],
  ];

  return `<div class="mrz-field-list">${rows.map(([label, value, note]) => `
    <div class="mrz-field-row">
      <b>${escapeHtml(label)}</b>
      <code>${escapeHtml(value || "—")}</code>
      <small>${escapeHtml(note)}</small>
    </div>`).join("")}</div>`;
}

function renderCrossChecks(internalChecks) {
  return Object.entries(internalChecks)
    .filter(([key, value]) => key.endsWith("_visible_vs_mrz") && value && typeof value === "object")
    .map(([key, value]) => {
      const field = key.replace(/_visible_vs_mrz$/, "").replace(/_/g, " ");
      const supporting = value.severity === "supporting";
      const cls = value.pass === true ? "pass" : supporting ? "supporting" : "fail";
      const heading = value.pass === true
        ? `Printed ${field} agrees with MRZ evidence`
        : supporting
          ? `Printed ${field} differs from MRZ representation — supporting only`
          : `Printed ${field} contradicts checksum-protected MRZ evidence`;
      const explanation = supporting
        ? "This difference is shown for review but does not independently fail same-document validation."
        : value.pass === true
          ? "The printed extraction and the independently read MRZ support the same value."
          : "This is treated as a hard contradiction only because the relevant MRZ checksum passed.";

      return `<div class="cross-check ${cls}">
        <b>${escapeHtml(heading)}</b>
        <p>Printed: <code>${escapeHtml(value.visible ?? "—")}</code></p>
        <p>MRZ: <code>${escapeHtml(value.mrz ?? "—")}</code></p>
        <small>${escapeHtml(explanation)}</small>
      </div>`;
    })
    .join("");
}

function renderWarnings(warnings) {
  if (!warnings.length) {
    return `<div class="processing-notes"><b>Processing notes</b><p>No additional processing warnings.</p></div>`;
  }
  return `<div class="processing-notes"><b>Processing notes</b><ul>${warnings
    .map((warning) => `<li>${escapeHtml(warning)}</li>`)
    .join("")}</ul></div>`;
}

function renderRuntime(result) {
  const historical = result.historical?.timing_ms || {};
  const current = result.current?.timing_ms || {};
  const total = result.timing_ms || {};

  document.querySelector("#runtime-details").innerHTML = `
    <div><b>Older document</b><br>${formatTiming(historical)}</div>
    <div><b>Current document</b><br>${formatTiming(current)}</div>
    <div><b>Final old-versus-current comparison</b><br>${formatMs(total.record_matching)}<br><small>Normalization, checksum arithmetic and deterministic comparison are normally negligible beside OCR inference.</small></div>
  `;
}

function formatTiming(timing) {
  const rows = [
    ["Document preparation", timing.document_preparation],
    ["Visible OCR", timing.visible_text_extraction],
    ["Field mapping / normalization", timing.field_mapping_and_normalization],
    ["MRZ processing / fallback", timing.mrz_processing],
    ["Internal validation", timing.internal_validation],
  ];
  return rows
    .filter(([, value]) => value != null)
    .map(([name, value]) => `${escapeHtml(name)}: ${formatMs(value)}`)
    .join("<br>");
}

function formatMs(value) {
  if (value == null) return "—";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Number(value).toFixed(1)} ms`;
}

function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    "'": "&#39;",
    '"': "&quot;",
  })[char]);
}
