const form = document.querySelector("#verify-form");
const button = document.querySelector("#verify-button");
const statusPanel = document.querySelector("#status-panel");
const resultPanel = document.querySelector("#result-panel");
const piiToggle = document.querySelector("#pii-toggle");
const backendSelect = document.querySelector("#extraction-backend");

let lastResult = null;
let revealPii = false;

const FIELD_LABELS = {
  id_number: "Emirates ID number",
  date_of_birth: "Date of birth",
  name_en: "English name",
  name_ar: "Arabic name",
  nationality: "Nationality",
  sex: "Sex",
  expiry_date: "Expiry date"
};

bindSelection("historical-document", "historical-selection");
bindSelection("current-document", "current-selection");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const problem = validateInputs();
  if (problem) return setStatus(problem, true);

  button.disabled = true;
  button.textContent = "Verifying…";
  resultPanel.classList.add("hidden");

  setStatus(
    `Verifying with ${backendSelect.options[backendSelect.selectedIndex].text}…`,
    false
  );

  try {
    const backend = backendSelect.value;

    const response = await fetch(
      `/api/v1/text/verify?backend=${encodeURIComponent(backend)}`,
      {
        method: "POST",
        body: new FormData(form)
      }
    );

    const body = await response.json();

    if (!response.ok) {
      throw new Error(
        body.detail ||
        "Document verification could not be completed."
      );
    }

    lastResult = body;
    revealPii = false;
    piiToggle.textContent = "Reveal document values";

    render(body);

    statusPanel.classList.add("hidden");
    resultPanel.classList.remove("hidden");

    resultPanel.scrollIntoView({
      behavior: "smooth",
      block: "start"
    });

  } catch (error) {
    setStatus(
      error.message ||
      "Document verification could not be completed.",
      true
    );

  } finally {
    button.disabled = false;
    button.textContent = "Verify documents →";
  }
});

piiToggle.addEventListener("click", () => {
  revealPii = !revealPii;

  piiToggle.textContent = revealPii
    ? "Mask document values"
    : "Reveal document values";

  if (lastResult) {
    renderTable(lastResult);
  }
});

function bindSelection(inputId, targetId) {
  const input = document.querySelector(`#${inputId}`);
  const target = document.querySelector(`#${targetId}`);

  input.addEventListener("change", () => {
    const names = [...input.files].map(file => file.name);

    target.textContent = names.length
      ? names.join(" · ")
      : "No file selected";
  });
}

function validateInputs() {
  for (const [id, role] of [
    ["historical-document", "previous"],
    ["current-document", "current"]
  ]) {
    const files = [
      ...document.querySelector(`#${id}`).files
    ];

    if (!files.length) {
      return `Select the ${role} Emirates ID.`;
    }

    if (files.length > 2) {
      return `The ${role} ID accepts one PDF or at most two images.`;
    }

    if (
      files.length === 2 &&
      files.some(file => file.type === "application/pdf")
    ) {
      return `For the ${role} ID, use one PDF or two image files.`;
    }
  }

  return null;
}

function setStatus(message, isError) {
  statusPanel.textContent = message;
  statusPanel.classList.remove("hidden", "error");

  if (isError) {
    statusPanel.classList.add("error");
  }
}

function render(result) {
  const copy = userFacingVerdict(result);

  document.querySelector("#verdict").dataset.status = result.status;
  document.querySelector("#verdict-word").textContent = copy[0];
  document.querySelector("#verdict-why").textContent = copy[1];

  const comps = Object.fromEntries(
    (result.comparisons || []).map(
      item => [item.field, item]
    )
  );

  document.querySelector("#id-result").textContent =
    comparisonShort(comps.id_number);

  document.querySelector("#dob-result").textContent =
    comparisonShort(comps.date_of_birth);

  document.querySelector("#english-name-result").textContent =
    comparisonShort(comps.name_en);

  document.querySelector("#arabic-name-result").textContent =
    comparisonShort(comps.name_ar);

  document.querySelector("#nationality-result").textContent =
    comparisonShort(comps.nationality);

  document.querySelector("#sex-result").textContent =
    comparisonShort(comps.sex);

  document.querySelector("#old-mrz").textContent =
    documentValidationSummary(result.historical?.mrz);

  document.querySelector("#current-mrz").textContent =
    documentValidationSummary(result.current?.mrz);

  document.querySelector("#expiry-result").textContent =
    expirySummary(
      result.current?.internal_checks?.current_expiry_valid
    );

  document.querySelector("#backend-result").textContent =
    backendLabel(result.historical?.ocr_engine);

  document.querySelector("#ocr-confidence").textContent =
    averageOcrConfidence(result);

  document.querySelector("#runtime-summary").textContent =
    formatMs(benchmarkTotal(result));

  renderTable(result);
  renderRuntime(result);
  renderReasons(result.reasons || []);
  renderOcrEvidence(result);

  document.querySelector("#result-json").textContent =
    JSON.stringify(
      resultForRawDisplay(result),
      null,
      2
    );
}

function renderTable(result) {
  const body = document.querySelector("#comparison-body");
  body.replaceChildren();

  for (const item of result.comparisons || []) {
    const oldField =
      result.historical?.record?.[item.field];

    const curField =
      result.current?.record?.[item.field];

    const tr = document.createElement("tr");

    tr.innerHTML =
      `<td>${esc(FIELD_LABELS[item.field] || human(item.field))}</td>` +
      `<td>${esc(display(item.historical))}</td>` +
      `<td>${confidence(oldField)}</td>` +
      `<td>${esc(display(item.current))}</td>` +
      `<td>${confidence(curField)}</td>` +
      `<td><b class="outcome ${esc(item.status)}">${esc(comparisonShort(item))}</b></td>`;

    body.append(tr);
  }
}

function renderRuntime(result) {
  const root = document.querySelector("#runtime-details");

  const old = result.historical?.timing_ms || {};
  const cur = result.current?.timing_ms || {};

  root.innerHTML =
    `<div class="metric">` +
      `<span>Previous ID</span>` +
      `<b>${formatMs(old.total)}</b>` +
    `</div>` +

    `<div class="metric">` +
      `<span>Current ID</span>` +
      `<b>${formatMs(cur.total)}</b>` +
    `</div>` +

    `<div class="metric">` +
      `<span>Field comparison</span>` +
      `<b>${formatMs(result.timing_ms?.record_matching)}</b>` +
    `</div>` +

    `<div class="metric">` +
      `<span>Total</span>` +
      `<b>${formatMs(benchmarkTotal(result))}</b>` +
    `</div>`;
}

function renderReasons(reasons) {
  const root = document.querySelector("#reason-list");
  root.replaceChildren();

  const values = reasons.length
    ? reasons
    : ["No additional review reason."];

  values.forEach(reason => {
    const p = document.createElement("p");
    p.textContent = reason;
    root.append(p);
  });
}

function renderOcrEvidence(result) {
  const root = document.querySelector("#ocr-evidence");

  if (!root) {
    return;
  }

  root.replaceChildren();

  const documents = [
    ["Previous ID", result.historical],
    ["Current ID", result.current]
  ];

  let found = false;

  for (const [title, documentResult] of documents) {
    const pages = documentResult?.pages || [];

    if (
      !pages.some(
        page =>
          page.preview_data_url ||
          Array.isArray(page.ocr_lines)
      )
    ) {
      continue;
    }

    found = true;

    const heading = document.createElement("h3");
    heading.className = "ocr-document-title";
    heading.textContent = title;
    root.append(heading);

    const grid = document.createElement("div");
    grid.className = "ocr-grid";

    pages.forEach((page, index) => {
      const card = document.createElement("article");
      card.className = "ocr-card";

      const cardTitle = document.createElement("h4");

      cardTitle.textContent =
        `${index === 0 ? "Front" : "Back"} · ` +
        `${page.ocr_lines?.length || 0} detections`;

      card.append(cardTitle);

      if (page.preview_data_url) {
        const visual = document.createElement("div");
        visual.className = "ocr-image";

        const img = document.createElement("img");
        img.src = page.preview_data_url;
        img.alt =
          `${title} ${index === 0 ? "front" : "back"}`;

        visual.append(img);

        if (page.ocr_width && page.ocr_height) {
          for (const line of page.ocr_lines || []) {
            if (
              !Array.isArray(line.bbox) ||
              !line.bbox.length
            ) {
              continue;
            }

            const xs = line.bbox
              .map(point => Number(point?.[0]))
              .filter(Number.isFinite);

            const ys = line.bbox
              .map(point => Number(point?.[1]))
              .filter(Number.isFinite);

            if (!xs.length || !ys.length) {
              continue;
            }

            const left = Math.max(
              0,
              Math.min(...xs)
            );

            const right = Math.min(
              page.ocr_width,
              Math.max(...xs)
            );

            const top = Math.max(
              0,
              Math.min(...ys)
            );

            const bottom = Math.min(
              page.ocr_height,
              Math.max(...ys)
            );

            if (
              right <= left ||
              bottom <= top
            ) {
              continue;
            }

            const box = document.createElement("span");
            box.className = "ocr-box";

            box.style.left =
              `${left / page.ocr_width * 100}%`;

            box.style.top =
              `${top / page.ocr_height * 100}%`;

            box.style.width =
              `${(right - left) / page.ocr_width * 100}%`;

            box.style.height =
              `${(bottom - top) / page.ocr_height * 100}%`;

            box.title =
              line.confidence != null
                ? `${line.text} · ${Math.round(Number(line.confidence) * 100)}%`
                : line.text;

            visual.append(box);
          }
        }

        card.append(visual);

      } else {
        const note = document.createElement("p");
        note.className = "ocr-empty";
        note.textContent = "Page preview unavailable.";
        card.append(note);
      }

      grid.append(card);
    });

    root.append(grid);
  }

  if (!found) {
    const note = document.createElement("p");
    note.className = "ocr-empty";

    note.textContent =
      result.historical?.ocr_engine === "nanonets-ocr2-3b"
        ? "Nanonets currently returns structured fields rather than line-level OCR boxes."
        : "No OCR evidence returned. Set DEBUG_INCLUDE_RAW_TEXT=true and run verification again.";

    root.append(note);
  }
}

function benchmarkTotal(result) {
  return (
    Number(
      result.historical?.timing_ms?.total || 0
    ) +
    Number(
      result.current?.timing_ms?.total || 0
    ) +
    Number(
      result.timing_ms?.record_matching || 0
    )
  );
}

function resultForRawDisplay(result) {
  const copy = structuredClone(result);

  for (const documentResult of [
    copy.historical,
    copy.current
  ]) {
    for (const page of documentResult?.pages || []) {
      if (page.preview_data_url) {
        page.preview_data_url =
          "[debug image omitted from raw display]";
      }
    }
  }

  return copy;
}

function averageOcrConfidence(result) {
  const values = [];

  for (const documentResult of [
    result.historical,
    result.current
  ]) {
    for (
      const field
      of Object.values(
        documentResult?.record || {}
      )
    ) {
      if (field?.confidence != null) {
        values.push(
          Number(field.confidence)
        );
      }
    }
  }

  if (!values.length) {
    return "Not available";
  }

  return `${
    Math.round(
      values.reduce(
        (a, b) => a + b,
        0
      ) /
      values.length *
      100
    )
  }%`;
}

function confidence(field) {
  if (field?.confidence != null) {
    return `${Math.round(field.confidence * 100)}%`;
  }

  if (
    (field?.sources || []).some(
      source =>
        String(source).startsWith("mrz:")
    )
  ) {
    return "MRZ";
  }

  return "—";
}

function display(value) {
  if (!value) {
    return "Not available";
  }

  if (revealPii) {
    return value;
  }

  const stringValue = String(value)
    .replace(/\s+/g, " ")
    .trim();

  if (stringValue.length <= 4) {
    return "••••";
  }

  return (
    stringValue.slice(0, 2) +
    "•".repeat(
      Math.min(
        10,
        stringValue.length - 4
      )
    ) +
    stringValue.slice(-2)
  );
}

function comparisonShort(item) {
  if (!item) {
    return "Not available";
  }

  return {
    match: "Match",
    review: "Mismatch",
    variation: "Needs review",
    missing: "Not available",
    not_compared: "Not compared"
  }[item.status] || human(item.status);
}

function documentValidationSummary(mrz) {
  if (!mrz?.valid_structure) {
    return "Needs clearer upload";
  }

  const checks = mrz.checks || {};

  const values = [
    checks.document_number,
    checks.date_of_birth,
    checks.expiry_date,
    checks.composite
  ].filter(
    value => value != null
  );

  return values.some(
    value => value === false
  )
    ? "Needs review"
    : "Passed";
}

function expirySummary(value) {
  return value === true
    ? "Valid"
    : value === false
      ? "Expired / invalid"
      : "Not confirmed";
}

function formatMs(value) {
  if (value == null) {
    return "—";
  }

  return value >= 1000
    ? `${(value / 1000).toFixed(2)} s`
    : `${Math.round(value)} ms`;
}

function human(value) {
  return String(value || "")
    .replaceAll("_", " ")
    .replace(
      /\b\w/g,
      character =>
        character.toUpperCase()
    );
}

function esc(value) {
  return String(value ?? "")
    .replace(
      /[&<>'"]/g,
      character => ({
        "&": "&amp;",
        "<": "&lt;",
        ">": "&gt;",
        "'": "&#39;",
        '"': "&quot;"
      })[character]
    );
}

function backendLabel(value) {
  return {
    "paddleocr-ppocrv5-arabic-mobile":
      "PaddleOCR — PP-OCRv5 mobile",

    "easyocr":
      "EasyOCR — Arabic + English",

    "tesseract-eng-ara":
      "Tesseract — Arabic + English",

    "nanonets-ocr2-3b":
      "Nanonets OCR2-3B — document VLM"

  }[value] ||
    human(
      value ||
      "Not available"
    );
}

function userFacingVerdict(result) {
  const comps = Object.fromEntries(
    (result.comparisons || []).map(
      item => [item.field, item]
    )
  );

  if (result.status === "consistent") {
    return [
      "Identity information matches",
      "The previous and current Emirates IDs contain consistent identity information and the required document checks passed."
    ];
  }

  if (comps.id_number?.status === "review") {
    return [
      "Emirates ID numbers do not match",
      "The previous and current documents contain different Emirates ID numbers. Review the submitted documents before continuing."
    ];
  }

  if (comps.date_of_birth?.status === "review") {
    return [
      "Dates of birth do not match",
      "The previous and current documents contain different dates of birth. Review the submitted documents before continuing."
    ];
  }

  if (
    result.current
      ?.internal_checks
      ?.current_expiry_valid === false
  ) {
    return [
      "Current Emirates ID is expired",
      "The current document cannot be automatically verified because its extracted expiry date has passed."
    ];
  }

  if (
    !result.historical
      ?.mrz
      ?.valid_structure
  ) {
    return [
      "Previous ID could not be fully validated",
      "The machine-readable information on the previous ID could not be read reliably. Upload a clearer previous ID and try again."
    ];
  }

  if (
    !result.current
      ?.mrz
      ?.valid_structure
  ) {
    return [
      "Current ID could not be fully validated",
      "The machine-readable information on the current ID could not be read reliably. Upload a clearer current ID and try again."
    ];
  }

  if (
    result.historical
      ?.mrz
      ?.checks
      ?.document_number === false ||
    result.historical
      ?.mrz
      ?.checks
      ?.date_of_birth === false ||
    result.historical
      ?.mrz
      ?.checks
      ?.expiry_date === false ||
    result.historical
      ?.mrz
      ?.checks
      ?.composite === false
  ) {
    return [
      "Previous ID needs review",
      "One or more machine-readable validation checks on the previous ID did not pass after the available reading attempts."
    ];
  }

  if (
    result.current
      ?.mrz
      ?.checks
      ?.document_number === false ||
    result.current
      ?.mrz
      ?.checks
      ?.date_of_birth === false ||
    result.current
      ?.mrz
      ?.checks
      ?.expiry_date === false ||
    result.current
      ?.mrz
      ?.checks
      ?.composite === false
  ) {
    return [
      "Current ID needs review",
      "One or more machine-readable validation checks on the current ID did not pass after the available reading attempts."
    ];
  }

  if (result.status === "insufficient_data") {
    return [
      "More document information is needed",
      "A required identity field could not be read from both documents. Upload clearer document images and try again."
    ];
  }

  if (
    [
      comps.name_en,
      comps.name_ar
    ].some(
      item =>
        item?.status === "variation" ||
        item?.status === "review"
    )
  ) {
    return [
      "Name information needs review",
      "The name evidence is not an exact normalized match. Review the compared values before continuing."
    ];
  }

  return [
    "Document needs review",
    "A document check produced an inconsistency or unresolved validation issue. Open the compared fields for more information."
  ];
}