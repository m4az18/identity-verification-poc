from types import SimpleNamespace

from identity_verification.features.text_verification.domain.models import ComparisonStatus
from identity_verification.orchestration.evidence import build_document_evidence_score


def _c(field, status):
    return SimpleNamespace(field=field, status=status)


def _doc(*, mrz_ok=True, expiry=True, checks=None):
    mrz = SimpleNamespace(
        valid_structure=mrz_ok is not None,
        checks=SimpleNamespace(all_available_pass=mrz_ok),
    )
    return SimpleNamespace(
        mrz=mrz,
        internal_checks={"current_expiry_valid": expiry, **(checks or {})},
    )


def _result():
    return SimpleNamespace(
        comparisons=[
            _c("id_number", ComparisonStatus.MATCH),
            _c("date_of_birth", ComparisonStatus.MATCH),
            _c("name_en", ComparisonStatus.MATCH),
            _c("name_ar", ComparisonStatus.MATCH),
            _c("nationality", ComparisonStatus.MATCH),
            _c("sex", ComparisonStatus.MATCH),
        ],
        historical=_doc(
            checks={
                "id_number_visible_vs_mrz": {"pass": True, "severity": "critical"},
                "date_of_birth_visible_vs_mrz": {"pass": True, "severity": "critical"},
            }
        ),
        current=_doc(
            checks={
                "id_number_visible_vs_mrz": {"pass": True, "severity": "critical"},
                "date_of_birth_visible_vs_mrz": {"pass": True, "severity": "critical"},
            }
        ),
    )


def test_full_document_evidence_index_is_100_when_all_policy_components_pass():
    score = build_document_evidence_score(_result())
    assert score.score == 100.0
    assert score.available_weight == 100.0
    assert "not a probability" in score.disclaimer


def test_critical_mismatch_lowers_index_but_score_itself_does_not_decide():
    result = _result()
    result.comparisons[0].status = ComparisonStatus.REVIEW
    score = build_document_evidence_score(result)
    assert score.score == 75.0
    critical = next(x for x in score.components if x.key == "critical_identity_continuity")
    assert critical.passed is False
