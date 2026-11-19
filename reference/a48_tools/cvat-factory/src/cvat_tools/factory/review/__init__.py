"""Phase 4 Automated VS Code Review package."""

from .dashboard import record_decision, render_review_dashboard, save_decisions
from .models import ReviewBatch, ReviewCase, ReviewProposal, VSCodeOpenResult
from .pipeline import AutomatedReviewManager, open_review
from .renderer import enhance_for_display, render_case_evidence
from .vscode import open_in_vscode, resolve_vscode_cli

__all__ = [
    "AutomatedReviewManager",
    "ReviewBatch",
    "ReviewCase",
    "ReviewProposal",
    "VSCodeOpenResult",
    "enhance_for_display",
    "open_in_vscode",
    "open_review",
    "record_decision",
    "render_case_evidence",
    "render_review_dashboard",
    "resolve_vscode_cli",
    "save_decisions",
]
