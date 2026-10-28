const form = document.querySelector("#face-form");
const idInput = document.querySelector("#id-photo");
const historicalInput = document.querySelector("#historical-document");
const modeInput = document.querySelector("#verification-mode");
const liveInput = document.querySelector("#live-file");
const video = document.querySelector("#camera-video");
const livePreview = document.querySelector("#live-preview");
const cameraEmpty = document.querySelector("#camera-empty");
const startCameraButton = document.querySelector("#start-camera");
const captureButton = document.querySelector("#capture-button");
const compareButton = document.querySelector("#compare-button");
const canvas = document.querySelector("#capture-canvas");
const working = document.querySelector("#working");
const errorPanel = document.querySelector("#error");
const resultPanel = document.querySelector("#result");
let liveBlob = null;
let cameraStream = null;
let livePreviewUrl = null;
let submitting = false;

idInput.addEventListener("change", () => {
  document.querySelector("#id-file-name").textContent = selectionText(idInput);
});
historicalInput.addEventListener("change", () => {
  document.querySelector("#historical-file-name").textContent = selectionText(historicalInput);
});
modeInput.addEventListener("change", () => {
  const identity = modeInput.value === "identity";
  document.querySelector("#historical-upload").classList.toggle("hidden", !identity);
  historicalInput.required = identity;
  idInput.multiple = identity;
  idInput.accept = identity ? "application/pdf,image/png,image/jpeg,.pdf,.png,.jpg,.jpeg" : "image/png,image/jpeg,.png,.jpg,.jpeg";
  document.querySelector("#current-step").textContent = identity ? "2" : "1";
  document.querySelector("#face-step").textContent = identity ? "3" : "2";
  document.querySelector("#current-hint").textContent = identity ? "One PDF or front and back images, including the portrait" : "Use one image of the side that shows the portrait";
  compareButton.textContent = identity ? "Verify identity" : "Compare faces";
  resultPanel.classList.add("hidden");
  hideError();
});

if (document.body.dataset.faceOnly === "true") {
  modeInput.value = "face";
  modeInput.querySelector('option[value="identity"]').remove();
  modeInput.dispatchEvent(new Event("change"));
  modeInput.closest(".verification-mode").classList.add("hidden");
  document.querySelector('nav[aria-label="Verification options"]').classList.add("hidden");
}

liveInput.addEventListener("change", () => {
  const file = liveInput.files[0];
  if (!file) return;
  liveBlob = file;
  showLivePreview(file);
  stopCamera();
});

startCameraButton.addEventListener("click", async () => {
  hideError();
  if (!navigator.mediaDevices?.getUserMedia) return showError("Camera access is not supported. Choose a photo instead.");
  try {
    stopCamera();
    cameraStream = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "user", width: { ideal: 1280 } }, audio: false });
    video.srcObject = cameraStream;
    await video.play();
    video.classList.remove("hidden");
    livePreview.classList.add("hidden");
    cameraEmpty.classList.add("hidden");
    captureButton.classList.remove("hidden");
    startCameraButton.textContent = "Restart camera";
  } catch (_) {
    stopCamera();
    showError("Camera access was blocked. Allow permission or choose a photo.");
  }
});

captureButton.addEventListener("click", async () => {
  if (!cameraStream || !video.videoWidth) return showError("Start the camera first.");
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext("2d").drawImage(video, 0, 0);
  liveBlob = await new Promise(resolve => canvas.toBlob(resolve, "image/jpeg", 0.94));
  if (!liveBlob) return showError("The photo could not be captured.");
  showLivePreview(liveBlob);
  stopCamera();
  form.requestSubmit();
});

form.addEventListener("submit", async event => {
  event.preventDefault();
  if (submitting) return;
  hideError();
  resultPanel.classList.add("hidden");
  const identity = modeInput.value === "identity";
  if (identity && !validDocument(historicalInput, "previous")) return;
  if (!validDocument(idInput, "current", !identity)) return;
  if (!liveBlob) return showError("Take or choose a current face photo.");

  const body = new FormData();
  if (identity) {
    for (const file of historicalInput.files) body.append("historical_document", file);
    for (const file of idInput.files) body.append("current_document", file);
  } else {
    body.append("id_photo", idInput.files[0]);
  }
  body.append("live_photo", liveBlob, liveBlob.name || "live-photo.jpg");
  submitting = true;
  setBusy(true);
  working.classList.remove("hidden");
  try {
    const endpoint = identity ? "/api/v1/identity/verify" : "/api/v1/face/verify";
    const response = await fetch(endpoint, { method: "POST", body });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof payload.detail === "string" ? payload.detail : `Verification failed (${response.status}). Check the uploaded files and retry.`);
    renderResult(payload);
    resultPanel.classList.remove("hidden");
    resultPanel.scrollIntoView({ behavior: "smooth", block: "start" });
  } catch (error) {
    showError(error.message || "Verification could not be completed.");
  } finally {
    submitting = false;
    setBusy(false);
    working.classList.add("hidden");
  }
});

document.querySelector("#again-button").addEventListener("click", () => {
  resultPanel.classList.add("hidden");
  form.scrollIntoView({ behavior: "smooth" });
});
window.addEventListener("beforeunload", stopCamera);

function renderResult(payload) {
  const identity = Boolean(payload.decision);
  const result = identity ? payload.face_evidence : payload;
  document.querySelector("#document-result").classList.toggle("hidden", !identity);
  document.querySelector("#document-diagnostics").classList.toggle("hidden", !identity);
  const labels = { match: "Faces match", no_match: "Faces do not match", retry_required: "New photo needed", unavailable: "Verification unavailable" };
  document.querySelector("#result-banner").dataset.status = result.status;
  document.querySelector("#result-title").textContent = labels[result.status] || result.status;
  document.querySelector("#result-message").textContent = result.user_message || result.detail || "Verification finished.";
  document.querySelector("#similarity").textContent = result.similarity == null ? "—" : result.similarity.toFixed(4);
  document.querySelector("#threshold").textContent = result.threshold == null ? "" : `Threshold ${result.threshold.toFixed(2)}`;
  renderIdPipeline(result.document_preprocessing_previews || []);
  setImage("aligned-live-face", result.live_face_preview_data_url);
  document.querySelector("#live-operations").textContent = operationText(result.live_preprocessing);
  document.querySelector("#technical-status").textContent = humanize(result.status);
  document.querySelector("#detector").textContent = result.detector || "YOLOv8 Face";
  document.querySelector("#recognizer").textContent = result.recognizer || "AuraFace";
  document.querySelector("#processing-time").textContent = formatMs(payload.timing_ms?.total);
  if (identity) {
    document.querySelector("#result-banner").dataset.status = payload.decision.status;
    document.querySelector("#result-title").textContent = payload.decision.title;
    document.querySelector("#result-message").textContent = payload.decision.explanation;
    document.querySelector("#technical-status").textContent = humanize(payload.decision.status);
    renderDocumentResult(payload);
  }
}

function selectionText(input) {
  return Array.from(input.files).map(file => file.name).join(" + ") || "Choose";
}

function validDocument(input, label, imageOnly = false) {
  const files = Array.from(input.files);
  const pdfCount = files.filter(file => file.type === "application/pdf" || /\.pdf$/i.test(file.name)).length;
  if (!files.length) { showError(`Choose the ${label} Emirates ID first.`); return false; }
  if (imageOnly && (files.length !== 1 || pdfCount)) { showError("Face only requires one ID image. Use Documents + face for PDFs or front/back images."); return false; }
  if (files.length > 2 || (pdfCount && files.length !== 1)) { showError(`The ${label} ID must be one PDF or at most two images (front and back).`); return false; }
  return true;
}

function setBusy(busy) {
  for (const control of form.querySelectorAll("input, button, select")) control.disabled = busy;
}

function renderIdPipeline(phases) {
  const pipeline = document.querySelector("#id-pipeline");
  pipeline.replaceChildren();
  if (!phases.length) {
    const empty = document.createElement("p");
    empty.className = "pipeline-empty";
    empty.textContent = "No ID preprocessing images were produced for this request.";
    pipeline.append(empty);
    return;
  }
  phases.forEach((phase, index) => {
    const figure = document.createElement("figure");
    if (phase.image_data_url?.startsWith("data:image/png")) {
      figure.classList.add("transparent-output");
    }
    const imageBox = document.createElement("div");
    imageBox.className = "image-box";
    const image = document.createElement("img");
    image.src = phase.image_data_url;
    image.alt = phase.label || `ID preprocessing phase ${index + 1}`;
    const caption = document.createElement("figcaption");
    const title = document.createElement("b");
    title.textContent = `${index + 1}. ${phase.label}`;
    const operation = document.createElement("span");
    operation.textContent = phase.operation;
    imageBox.append(image);
    caption.append(title, operation);
    figure.append(imageBox, caption);
    pipeline.append(figure);
  });
}

function setImage(id, source) {
  const image = document.getElementById(id);
  image.src = source || "";
  image.closest("figure").classList.toggle("missing", !source);
}

function operationText(items) {
  return items?.length ? items.map(humanize).join(" · ") : "No correction needed";
}

function showLivePreview(blob) {
  if (livePreviewUrl) URL.revokeObjectURL(livePreviewUrl);
  livePreviewUrl = URL.createObjectURL(blob);
  livePreview.src = livePreviewUrl;
  livePreview.classList.remove("hidden");
  video.classList.add("hidden");
  cameraEmpty.classList.add("hidden");
}

function stopCamera() {
  if (cameraStream) cameraStream.getTracks().forEach(track => track.stop());
  cameraStream = null;
  video.srcObject = null;
  captureButton.classList.add("hidden");
  startCameraButton.textContent = liveBlob ? "Retake" : "Use camera";
}

function showError(message) { errorPanel.textContent = message; errorPanel.classList.remove("hidden"); }
function hideError() { errorPanel.classList.add("hidden"); }
function humanize(value) { return String(value || "").replaceAll("_", " ").replace(/\b\w/g, char => char.toUpperCase()); }
function formatMs(value) { return value == null ? "—" : value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`; }
