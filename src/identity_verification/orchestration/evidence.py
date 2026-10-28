from __future__ import annotations

from identity_verification.features.text_verification.domain.models import (
    ComparisonStatus,
    VerificationResult,
)
from identity_verification.orchestration.models import (
    EvidenceComponent,
    PolicyEvidenceScore,
)

# Policy weights are intentionally *not* probabilities. They make the relative strength
# of document evidence explicit for diagnostics while hard gates in DecisionPolicy still
# determine whether the case can be verified. A high score can never override a critical
# identity contradiction, an unusable required input, or a biometric failure.
COMPONENT_WEIGHTS = {
    "critical_identity_continuity": 40.0,
    "mrz_integrity": 20.0,
    "visible_mrz_consistency": 20.0,
    "supporting_identity_continuity": 10.0,
    "current_document_validity": 10.0,
}


def build_document_evidence_score(result: VerificationResult) -> PolicyEvidenceScore:
    components = [
        _critical_identity(result),
        _mrz_integrity(result),
        _visible_mrz_consistency(result),
        _supporting_identity(result),
        _current_validity(result),
    ]
    earned = round(sum(item.earned for item in components), 2)
    available_weight = round(sum(item.weight for item in components if item.available), 2)
    return PolicyEvidenceScore(
        score=earned,
        maximum=100.0,
        available_weight=available_weight,
        components=components,
        label="Document evidence index",
        disclaimer=(
            "Policy evidence index only; it is not a probability, model confidence, or "
            "identity-risk score and it never overrides hard verification gates."
        ),
    )


def _critical_identity(result: VerificationResult) -> EvidenceComponent:
    weight = COMPONENT_WEIGHTS["critical_identity_continuity"]
    weights = {"id_number": 25.0, "date_of_birth": 15.0}
    selected = [item for item in result.comparisons if item.field in weights]
    available = all(item.status != ComparisonStatus.MISSING for item in selected) and len(selected) == 2
    earned = sum(weights[item.field] for item in selected if item.status == ComparisonStatus.MATCH)
    if not available:
        detail = "Required identity keys were not available from both documents."
    elif earned == weight:
        detail = "Emirates ID number and date of birth match across previous and current IDs."
    else:
        detail = "At least one critical identity key differs across the two IDs."
    return EvidenceComponent(
        key="critical_identity_continuity",
        label="Critical identity continuity",
        weight=weight,
        earned=round(earned, 2),
        available=available,
        passed=available and earned == weight,
        detail=detail,
    )


def _mrz_integrity(result: VerificationResult) -> EvidenceComponent:
    weight = COMPONENT_WEIGHTS["mrz_integrity"]
    earned = 0.0
    available = True
    passing: list[str] = []
    failing: list[str] = []
    for label, document in (("previous", result.historical), ("current", result.current)):
        mrz = document.mrz
        ok = bool(mrz and mrz.valid_structure and mrz.checks.all_available_pass is True)
        if ok:
            earned += weight / 2
            passing.append(label)
        else:
            if mrz is None or not mrz.valid_structure or mrz.checks.all_available_pass is None:
                available = False
            failing.append(label)
    detail = (
        "MRZ structure and available check digits validate on both documents."
        if earned == weight
        else f"MRZ validation is incomplete or unresolved for: {', '.join(failing)}."
    )
    return EvidenceComponent(
        key="mrz_integrity",
        label="MRZ integrity",
        weight=weight,
        earned=round(earned, 2),
        available=available,
        passed=earned == weight,
        detail=detail,
    )


def _visible_mrz_consistency(result: VerificationResult) -> EvidenceComponent:
    weight = COMPONENT_WEIGHTS["visible_mrz_consistency"]
    checks: list[bool] = []
    for document in (result.historical, result.current):
        for key, value in document.internal_checks.items():
            if not key.endswith("_visible_vs_mrz") or not isinstance(value, dict):
                continue
            if value.get("severity") != "critical":
                continue
            passed = value.get("pass")
            if isinstance(passed, bool):
                checks.append(passed)

    if not checks:
        return EvidenceComponent(
            key="visible_mrz_consistency",
            label="Printed ↔ MRZ consistency",
            weight=weight,
            earned=0.0,
            available=False,
            passed=None,
            detail="No checksum-protected printed↔MRZ cross-check was available.",
        )

    ratio = sum(1 for value in checks if value) / len(checks)
    earned = weight * ratio
    return EvidenceComponent(
        key="visible_mrz_consistency",
        label="Printed ↔ MRZ consistency",
        weight=weight,
        earned=round(earned, 2),
        available=True,
        passed=all(checks),
        detail=(
            "Available checksum-protected printed fields agree with their MRZ values."
            if all(checks)
            else "At least one validated printed field conflicts with its MRZ value."
        ),
    )


def _supporting_identity(result: VerificationResult) -> EvidenceComponent:
    weight = COMPONENT_WEIGHTS["supporting_identity_continuity"]
    supporting = [item for item in result.comparisons if item.field in {"name_en", "name_ar", "nationality", "sex"}]
    comparable = [item for item in supporting if item.status != ComparisonStatus.MISSING]
    if not comparable:
        return EvidenceComponent(
            key="supporting_identity_continuity",
            label="Supporting identity continuity",
            weight=weight,
            earned=0.0,
            available=False,
            passed=None,
            detail="No supporting identity field was comparable across both documents.",
        )

    matches = sum(item.status == ComparisonStatus.MATCH for item in comparable)
    earned = weight * (matches / len(comparable))
    return EvidenceComponent(
        key="supporting_identity_continuity",
        label="Supporting identity continuity",
        weight=weight,
        earned=round(earned, 2),
        available=True,
        passed=matches == len(comparable),
        detail=(
            "All comparable supporting identity fields match."
            if matches == len(comparable)
            else "One or more supporting identity fields differ or require review."
        ),
    )


def _current_validity(result: VerificationResult) -> EvidenceComponent:
    weight = COMPONENT_WEIGHTS["current_document_validity"]
    value = result.current.internal_checks.get("current_expiry_valid")
    if value is True:
        return EvidenceComponent(
            key="current_document_validity",
            label="Current document validity",
            weight=weight,
            earned=weight,
            available=True,
            passed=True,
            detail="The current document has a valid, non-expired expiry date.",
        )
    if value is False:
        return EvidenceComponent(
            key="current_document_validity",
            label="Current document validity",
            weight=weight,
            earned=0.0,
            available=True,
            passed=False,
            detail="The current document expiry check failed.",
        )
    return EvidenceComponent(
        key="current_document_validity",
        label="Current document validity",
        weight=weight,
        earned=0.0,
        available=False,
        passed=None,
        detail="The current document expiry date could not be validated.",
    )
