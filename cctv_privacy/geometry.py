"""All boxes are half-open [x1, y1, x2, y2] in prepared-video pixels."""

from __future__ import annotations
import math
import numpy as np


def clip_box(box, width, height):
    x1, y1, x2, y2 = map(float, box)
    if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
        return None
    x1, y1 = max(0, math.floor(x1)), max(0, math.floor(y1))
    x2, y2 = min(width, math.ceil(x2)), min(height, math.ceil(y2))
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def area(box):
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def intersection(a, b):
    return max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))


def iou(a, b):
    inter = intersection(a, b)
    return inter / max(1e-9, area(a) + area(b) - inter)


def expand(box, width, height, fraction=0.35, minimum=5):
    dx = max(minimum, (box[2] - box[0]) * fraction)
    dy = max(minimum, (box[3] - box[1]) * fraction)
    return clip_box([box[0] - dx, box[1] - dy, box[2] + dx, box[3] + dy], width, height)


def inverse_rotated_box(box, k, width, height):
    """Undo np.rot90(image, k). Pixel EDGE coordinates, not pixel centers."""
    pts = np.array(
        [[box[0], box[1]], [box[2], box[1]], [box[0], box[3]], [box[2], box[3]]], dtype=float
    )
    u, v = pts[:, 0], pts[:, 1]
    if k % 4 == 0:
        x, y = u, v
    elif k % 4 == 1:
        x, y = width - v, u
    elif k % 4 == 2:
        x, y = width - u, height - v
    else:
        x, y = v, height - u
    return [float(x.min()), float(y.min()), float(x.max()), float(y.max())]


def conservative_merge(detections, threshold=0.35):
    """Merge duplicate boxes using their UNION so padding is never shrunk by NMS."""
    pending = sorted(detections, key=lambda d: d["score"], reverse=True)
    output = []
    while pending:
        item = dict(pending.pop(0))
        changed = True
        while changed:
            changed = False
            keep = []
            for other in pending:
                if iou(item["box"], other["box"]) >= threshold:
                    a, b = item["box"], other["box"]
                    item["box"] = [
                        min(a[0], b[0]),
                        min(a[1], b[1]),
                        max(a[2], b[2]),
                        max(a[3], b[3]),
                    ]
                    item["score"] = max(item["score"], other["score"])
                    changed = True
                else:
                    keep.append(other)
            pending = keep
        output.append(item)
    return output
