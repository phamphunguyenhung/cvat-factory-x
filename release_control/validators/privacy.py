"""Privacy validator: enforces privacy_policy on faces and PII risk metadata."""

from __future__ import annotations

from typing import Any

from .models import ValidationIssue, resolve_image_id, resolve_image_relpath


def _extract_privacy_flags(container: dict[str, Any]) -> tuple[bool, bool, bool]:
    """Extract (contains_faces, pii_risk, privacy_approved) from a dict or its metadata/privacy subdicts."""
    privacy_sub = container.get("privacy") if isinstance(container.get("privacy"), dict) else {}
    meta_sub = container.get("metadata") if isinstance(container.get("metadata"), dict) else {}

    contains_faces = bool(
        container.get("contains_faces")
        or privacy_sub.get("contains_faces")
        or meta_sub.get("contains_faces")
    )
    pii_risk = bool(
        container.get("pii_risk")
        or privacy_sub.get("pii_risk")
        or meta_sub.get("pii_risk")
    )
    privacy_approved = bool(
        container.get("privacy_approved")
        or privacy_sub.get("privacy_approved")
        or meta_sub.get("privacy_approved")
    )
    return contains_faces, pii_risk, privacy_approved


def validate_privacy(
    manifest: dict[str, Any],
    contract: dict[str, Any] | None = None,
) -> list[ValidationIssue]:
    """Check privacy_policy for unapproved faces or PII risk in manifest and image metadata."""
    contract = contract or {}
    policy = contract.get("privacy_policy") if isinstance(contract.get("privacy_policy"), dict) else {}
    block_unapproved_faces = policy.get("block_unapproved_faces", True)
    block_unapproved_pii = policy.get("block_unapproved_pii", True)

    issues: list[ValidationIssue] = []

    m_faces, m_pii, m_approved = _extract_privacy_flags(manifest)
    if ((m_faces and block_unapproved_faces) or (m_pii and block_unapproved_pii)) and not m_approved:
        issues.append(
            ValidationIssue(
                code="PRIVACY_POLICY_VIOLATION",
                fault_type="PRIVACY",
                severity="CRITICAL",
                message=(
                    "Dataset manifest flags privacy risk (contains_faces/pii_risk) "
                    "without privacy_approved=True."
                ),
                details={
                    "contains_faces": m_faces,
                    "pii_risk": m_pii,
                    "privacy_approved": m_approved,
                },
            )
        )

    images = manifest.get("images", [])
    if isinstance(images, list):
        for idx, entry in enumerate(images):
            if not isinstance(entry, dict):
                continue
            i_faces, i_pii, i_approved = _extract_privacy_flags(entry)
            effective_approved = i_approved or m_approved
            if ((i_faces and block_unapproved_faces) or (i_pii and block_unapproved_pii)) and not effective_approved:
                image_id = resolve_image_id(entry, idx)
                issues.append(
                    ValidationIssue(
                        code="PRIVACY_POLICY_VIOLATION",
                        fault_type="PRIVACY",
                        severity="CRITICAL",
                        message=(
                            f"Image '{image_id}' flags privacy risk (contains_faces/pii_risk) "
                            "without privacy_approved=True."
                        ),
                        file_path=resolve_image_relpath(entry),
                        image_id=image_id,
                        details={
                            "contains_faces": i_faces,
                            "pii_risk": i_pii,
                            "privacy_approved": effective_approved,
                        },
                    )
                )

    return issues
