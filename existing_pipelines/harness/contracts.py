"""Typed V3 data contracts and compact C-order RLE mask codecs."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

import numpy as np


class LabelCategory(str, Enum):
    THING = "thing"
    STUFF = "stuff"
    STRUCTURE = "structure"
    DERIVED_AREA = "derived_area"


class ElementStatus(str, Enum):
    ACCEPTED = "accepted"
    REVIEW = "review"
    SUPPRESSED = "suppressed"
    DROPPED = "dropped"


class TrackStateKind(str, Enum):
    TENTATIVE = "tentative"
    CONFIRMED = "confirmed"
    OCCLUDED = "occluded"
    LOST = "lost"
    TERMINATED = "terminated"


@dataclass(frozen=True)
class RleMask:
    """Compact row-major (C-order) uncompressed RLE starting with run of zeros."""

    size: tuple[int, int]  # (height, width)
    counts: tuple[int, ...]

    @property
    def height(self) -> int:
        return self.size[0]

    @property
    def width(self) -> int:
        return self.size[1]

    @property
    def area(self) -> int:
        return int(sum(self.counts[1::2]))

    def to_dict(self) -> dict[str, Any]:
        return {"size": [int(self.size[0]), int(self.size[1])], "counts": list(self.counts)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "RleMask":
        sz = data["size"]
        return cls(size=(int(sz[0]), int(sz[1])), counts=tuple(int(x) for x in data["counts"]))


def encode_rle(mask: np.ndarray) -> RleMask:
    """Encode 2D binary mask into row-major C-order RLE starting with 0-run."""
    if mask.ndim != 2:
        raise ValueError(f"Expected 2D mask, got shape {mask.shape}")
    h, w = int(mask.shape[0]), int(mask.shape[1])
    flat = (mask.reshape(-1) > 0).astype(np.uint8)
    n = flat.size
    if n == 0:
        return RleMask(size=(h, w), counts=())
    changes = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.concatenate(([0], changes))
    ends = np.concatenate((changes, [n]))
    lengths = (ends - starts).tolist()
    if flat[0] == 1:
        lengths.insert(0, 0)
    return RleMask(size=(h, w), counts=tuple(int(v) for v in lengths))


def decode_rle(rle: RleMask | dict[str, Any] | None) -> np.ndarray:
    """Decode row-major C-order RLE into uint8 binary mask (0/1)."""
    if rle is None:
        return np.zeros((0, 0), dtype=np.uint8)
    obj = rle if isinstance(rle, RleMask) else RleMask.from_dict(rle)
    h, w = obj.size
    total = h * w
    out = np.zeros(total, dtype=np.uint8)
    pos = 0
    val = 0
    for run in obj.counts:
        nxt = min(total, pos + run)
        if val == 1 and nxt > pos:
            out[pos:nxt] = 1
        pos = nxt
        val ^= 1
    return out.reshape((h, w))


def bbox_to_rle(box_xyxy: list[float] | tuple[float, float, float, float], height: int, width: int) -> RleMask:
    """Rasterize clipped xyxy box into compact RleMask."""
    x1 = max(0, min(width, int(round(box_xyxy[0]))))
    y1 = max(0, min(height, int(round(box_xyxy[1]))))
    x2 = max(x1, min(width, int(round(box_xyxy[2]))))
    y2 = max(y1, min(height, int(round(box_xyxy[3]))))
    m = np.zeros((height, width), dtype=np.uint8)
    if x2 > x1 and y2 > y1:
        m[y1:y2, x1:x2] = 1
    return encode_rle(m)


def rle_bbox(rle: RleMask) -> list[float] | None:
    m = decode_rle(rle)
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    return [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def rle_intersection_area(a: RleMask, b: RleMask) -> int:
    ma = decode_rle(a)
    mb = decode_rle(b)
    if ma.shape != mb.shape or ma.size == 0:
        return 0
    return int(np.count_nonzero((ma > 0) & (mb > 0)))


def rle_iou(a: RleMask, b: RleMask) -> float:
    ma = decode_rle(a)
    mb = decode_rle(b)
    if ma.shape != mb.shape or ma.size == 0:
        return 0.0
    pa = ma > 0
    pb = mb > 0
    inter = int(np.count_nonzero(pa & pb))
    if inter == 0:
        return 0.0
    union = int(np.count_nonzero(pa | pb))
    return inter / float(union) if union > 0 else 0.0


@dataclass
class FrameProfile:
    frame_id: int
    width: int
    height: int
    mean_luma: float
    is_night: bool
    contrast_std: float
    glare_ratio: float
    blur_score: float
    continuity_group: str = "independent"
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ContinuityReport:
    is_continuous: bool
    reason: str
    median_hist_corr: float | None = None
    frame_count: int = 0
    per_pair_corr: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class RouteDecision:
    label: str
    category: LabelCategory
    export_shape_type: str  # "rectangle" | "polygon" | "polyline"
    z_order: int
    primary_model: str
    fallback_models: tuple[str, ...] = ()
    run_sam_on_overlap: bool = False
    use_night_enhance: bool = False
    accept_conf: float = 0.25
    review_conf_floor: float = 0.08
    notes: str = ""


@dataclass
class FrameRoutePlan:
    frame_id: int
    mode: str
    is_night: bool
    tracking_enabled: bool
    run_stuff: bool
    run_structure: bool
    decisions: dict[str, RouteDecision] = field(default_factory=dict)


@dataclass
class InstanceCandidate:
    candidate_id: str
    frame_id: int
    label: str
    bbox_xyxy: list[float]
    conf: float
    source_model: str
    mask_rle: RleMask | None = None
    mask_source: str = "none"  # "none" | "yolo_seg" | "sam_crop" | "bbox_fallback"
    occluded: bool = False
    truncated: bool = False
    occlusion_ratio: float = 0.0
    uncertainty: float = 0.0
    status: ElementStatus = ElementStatus.ACCEPTED
    duplicate_group: str | None = None
    track_id: int | None = None
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def width(self) -> float:
        return max(0.0, self.bbox_xyxy[2] - self.bbox_xyxy[0])

    @property
    def height(self) -> float:
        return max(0.0, self.bbox_xyxy[3] - self.bbox_xyxy[1])

    @property
    def area(self) -> float:
        return self.width * self.height


@dataclass
class StuffRegion:
    region_id: str
    frame_id: int
    label: str
    raw_mask_rle: RleMask
    prob_mean: float
    prob_p90: float
    boundary_uncertainty: float
    component_count: int
    simplified_polygons: list[list[float]] = field(default_factory=list)
    source_model: str = "segformer_b2"
    status: ElementStatus = ElementStatus.ACCEPTED
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class StructureGeometry:
    structure_id: str
    frame_id: int
    label: str
    geometry_type: str  # "polygon" | "polyline"
    points: list[float]
    mask_rle: RleMask | None = None
    conf: float = 0.5
    continuity_score: float = 1.0
    glare_risk: float = 0.0
    source_model: str = "classical_cv"
    status: ElementStatus = ElementStatus.ACCEPTED
    provenance: dict[str, Any] = field(default_factory=dict)


@dataclass
class TrackState:
    track_id: int
    label: str
    state: TrackStateKind
    history: dict[int, str] = field(default_factory=dict)  # frame_id -> candidate_id
    boxes: dict[int, list[float]] = field(default_factory=dict)  # frame_id -> bbox
    velocity_xy: tuple[float, float] = (0.0, 0.0)
    last_seen_frame: int = -1
    misses: int = 0
    reid_events: list[dict[str, Any]] = field(default_factory=list)
    interpolated_frames: list[int] = field(default_factory=list)


@dataclass
class PanopticElement:
    element_id: str
    frame_id: int
    label: str
    category: LabelCategory
    export_shape_type: str
    points: list[float]
    z_order: int
    raw_mask_rle: RleMask | None = None
    visible_mask_rle: RleMask | None = None
    instance_id: str | None = None
    track_id: int | None = None
    occluded: bool = False
    truncated: bool = False
    occlusion_ratio: float = 0.0
    occluded_by: list[str] = field(default_factory=list)
    conf: float = 1.0
    uncertainty: float = 0.0
    status: ElementStatus = ElementStatus.ACCEPTED
    review_flags: list[str] = field(default_factory=list)
    provenance: dict[str, Any] = field(default_factory=dict)

    def to_cvat_shape(self, label_id_map: dict[str, int] | None = None) -> dict[str, Any]:
        shape: dict[str, Any] = {
            "frame": int(self.frame_id),
            "label": self.label,
            "type": self.export_shape_type,
            "points": [round(float(v), 2) for v in self.points],
            "z_order": int(self.z_order),
            "occluded": bool(self.occluded),
            "source": "manual",
            "attributes": [
                {"name": "occluded", "value": "true" if self.occluded else "false"},
                {"name": "truncated", "value": "true" if self.truncated else "false"},
            ]
            if self.category == LabelCategory.THING
            else [],
            "_conf": round(float(self.conf), 4),
            "_status": self.status.value,
            "_provenance": self.provenance,
        }
        if label_id_map and self.label in label_id_map:
            shape["label_id"] = int(label_id_map[self.label])
        return shape


@dataclass
class ReviewItem:
    frame_id: int
    element_id: str
    label: str
    priority_score: float
    reason_code: str
    message: str
    bbox_xyxy: list[float] | None = None
    evidence: dict[str, Any] = field(default_factory=dict)


@dataclass
class ComposedScene:
    job_id: int
    frame_id: int
    profile: FrameProfile
    elements: list[PanopticElement] = field(default_factory=list)
    review_items: list[ReviewItem] = field(default_factory=list)
    timings_s: dict[str, float] = field(default_factory=dict)
    peak_rss_mb: float = 0.0


@dataclass
class QualityReport:
    scope: str
    metrics: dict[str, Any] = field(default_factory=dict)
    stop_gates: list[dict[str, Any]] = field(default_factory=list)
    honest_gaps: list[str] = field(default_factory=list)


@dataclass
class ExperimentRecord:
    experiment_id: str
    timestamp: str
    hypothesis: str
    baseline_id: str
    config: dict[str, Any]
    metrics: dict[str, Any]
    latency_s_per_frame: float
    peak_rss_mb: float
    verdict: str  # "accept" | "reject" | "inconclusive"
    notes: str = ""


def to_serializable(obj: Any) -> Any:
    """Convert dataclasses, Enums, NumPy scalars cleanly for JSON export."""
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, RleMask):
        return obj.to_dict()
    if hasattr(obj, "__dataclass_fields__"):
        return {k: to_serializable(v) for k, v in asdict(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_serializable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    return obj
