from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image


@dataclass(slots=True)
class NanonetsPageExtraction:
    """Structured output from one Nanonets OCR2 page inference."""

    fields: dict[str, str | None]
    mrz_lines: list[str] = field(default_factory=list)
    raw_output: str = ""
    warnings: list[str] = field(default_factory=list)


class NanonetsOCR2Extractor:
    """Lazy local adapter for ``nanonets/Nanonets-OCR2-3B``."""

    PROMPT = """You are extracting one visible page of a UAE Emirates ID for a benchmark.
Return ONLY one valid JSON object with exactly these keys:
{
  "id_number": string or null,
  "name_en": string or null,
  "name_ar": string or null,
  "date_of_birth": string or null,
  "nationality": string or null,
  "sex": string or null,
  "expiry_date": string or null,
  "mrz_lines": array of strings
}
Rules:
- Copy only values that are visibly present on this page. Do not infer, guess, translate, or invent values.
- Preserve names and dates as printed; local deterministic normalization happens after extraction.
- For mrz_lines, return only actual machine-readable-zone lines visible on the page, in top-to-bottom order.
- Preserve every '<' filler character in MRZ text exactly when visible.
- If a field is not clearly visible, use null. If no MRZ is visible, use an empty array.
- Do not add Markdown fences, prose, comments, confidence scores, or extra keys.
"""

    # Caps image tokens; huge scans are the main cause of slow inference.
    DEFAULT_MAX_PIXELS = 1280 * 28 * 28

    def __init__(
        self,
        *,
        model_id: str,
        cache_dir: Path,
        max_new_tokens: int = 1024,
        device: str | None = None,
        max_pixels: int | None = DEFAULT_MAX_PIXELS,
    ) -> None:
        self.model_id = model_id
        self.cache_dir = cache_dir
        self.max_new_tokens = max_new_tokens
        self.requested_device = device
        self.max_pixels = max_pixels
        self._model = None
        self._processor = None
        self._device: str | None = None

    @property
    def name(self) -> str:
        return "nanonets-ocr2-3b"

    @property
    def device(self) -> str | None:
        return self._device

    def _get_runtime(self):
        if self._model is None or self._processor is None:
            try:
                import torch
                from transformers import AutoModelForImageTextToText, AutoProcessor
            except ImportError as exc:
                raise RuntimeError(
                    "Nanonets OCR2 dependencies are not installed. Run `uv sync --extra ocr-nanonets`."
                ) from exc

            device = _resolve_device(self.requested_device, torch)
            dtype = _dtype_for_device(device, torch)

            model_cache = self.cache_dir / "huggingface"
            model_cache.mkdir(parents=True, exist_ok=True)

            processor_kwargs = {}
            if self.max_pixels is not None:
                processor_kwargs["max_pixels"] = self.max_pixels

            self._processor = AutoProcessor.from_pretrained(
                self.model_id,
                cache_dir=str(model_cache),
                **processor_kwargs,
            )
            self._model = AutoModelForImageTextToText.from_pretrained(
                self.model_id,
                cache_dir=str(model_cache),
                torch_dtype=dtype,
                attn_implementation="sdpa",
            ).to(device)
            self._model.eval()
            self._device = device
            print(f"[nanonets-ocr2] loaded on {device} ({dtype})")
        return self._model, self._processor

    def warm_up(self) -> None:
        dummy = np.full((640, 1000, 3), 255, dtype=np.uint8)
        self._generate_text(dummy, max_new_tokens=8)

    def extract_page(self, image_rgb: np.ndarray) -> NanonetsPageExtraction:
        generated = self._generate_text(image_rgb, max_new_tokens=self.max_new_tokens)
        return parse_nanonets_output(generated)

    def _generate_text(self, image_rgb: np.ndarray, *, max_new_tokens: int) -> str:
        model, processor = self._get_runtime()
        image = Image.fromarray(np.asarray(image_rgb, dtype=np.uint8)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": self.PROMPT},
                ],
            }
        ]

        chat_text = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )
        inputs = processor(
            text=[chat_text],
            images=[image],
            padding=True,
            return_tensors="pt",
        ).to(model.device)

        import torch

        with torch.inference_mode():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        prompt_length = inputs["input_ids"].shape[-1]
        return processor.decode(
            output_ids[0][prompt_length:],
            skip_special_tokens=True,
        ).strip()


def _resolve_device(requested: str | None, torch) -> str:
    if requested:
        if requested == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        if requested.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _dtype_for_device(device: str, torch):
    if device.startswith("cuda") or device == "mps":
        return torch.bfloat16
    return torch.float32


def parse_nanonets_output(text: str) -> NanonetsPageExtraction:
    """Parse the model's JSON response without silently accepting invented structure."""

    warnings: list[str] = []
    payload = _load_first_json_object(text)
    if payload is None:
        return NanonetsPageExtraction(
            fields=_empty_fields(),
            mrz_lines=[],
            raw_output=text,
            warnings=["Nanonets OCR2 did not return valid JSON for this page."],
        )

    fields: dict[str, str | None] = {}
    for key in _empty_fields():
        value = payload.get(key)
        if value is None:
            fields[key] = None
        elif isinstance(value, str):
            cleaned = value.strip()
            fields[key] = cleaned or None
        else:
            warnings.append(f"Nanonets OCR2 returned a non-string value for '{key}'; ignored.")
            fields[key] = None

    raw_mrz = payload.get("mrz_lines", [])
    mrz_lines: list[str] = []
    if isinstance(raw_mrz, list):
        mrz_lines = [str(line).strip() for line in raw_mrz if str(line).strip()]
    elif raw_mrz not in (None, ""):
        warnings.append("Nanonets OCR2 returned 'mrz_lines' in an unexpected format; ignored.")

    return NanonetsPageExtraction(
        fields=fields,
        mrz_lines=mrz_lines,
        raw_output=text,
        warnings=warnings,
    )


def _load_first_json_object(text: str) -> dict | None:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.removeprefix("```json").removeprefix("```").strip()
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3].strip()

    start = cleaned.find("{")
    if start < 0:
        return None

    try:
        payload, _ = json.JSONDecoder().raw_decode(cleaned[start:])
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def _empty_fields() -> dict[str, None]:
    return {
        "id_number": None,
        "name_en": None,
        "name_ar": None,
        "date_of_birth": None,
        "nationality": None,
        "sex": None,
        "expiry_date": None,
    }
