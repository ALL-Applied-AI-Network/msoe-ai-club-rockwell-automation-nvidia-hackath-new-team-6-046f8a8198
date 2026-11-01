"""Conservative position predictions from legacy YOLO labels and image appearance."""
from pathlib import Path

import numpy as np


LEGACY = {3: "missing_cap", 4: "damaged_vial", 5: "no_vial", 6: "crooked_cap"}
PRIORITY = {"pass": 0, "missing_cap": 1, "crooked_cap": 2,
            "damaged_vial": 3, "no_vial": 4}


def _yellow_score(rgb, center, radius=34):
    x, y = (int(round(v)) for v in center)
    h, w = rgb.shape[:2]
    patch = rgb[max(0, y-radius):min(h, y+radius+1),
                max(0, x-radius):min(w, x+radius+1)].astype(np.float32)
    if not patch.size:
        return 0.0
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    yellow = (r > 55) & (g > 45) & (b < .72*np.minimum(r, g)) & (r > 1.02*g)
    return float(yellow.mean())


def _scrap_evidence(rgb, slot):
    x, y = (int(round(v)) for v in slot["center"])
    h, w = rgb.shape[:2]
    patch = rgb[max(0, y-42):min(h, y+43),
                max(0, x-18):min(w, x+155)].astype(np.float32)
    if not patch.size:
        return 0.0
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    yellow = ((r > 55) & (g > 45) & (b < .72*np.minimum(r, g))).mean()
    gray = patch.mean(axis=2)
    edge = (np.abs(np.diff(gray, axis=1)) > 28).mean() if patch.shape[1] > 1 else 0
    neutral = ((gray > 105) & ((patch.max(axis=2)-patch.min(axis=2)) < 55)).mean()
    return float(5.0*yellow + 0.45*edge + 0.15*neutral)


def read_yolo(path, image_size):
    width, height = image_size
    detections = []
    for line_no, line in enumerate(Path(path).read_text().splitlines(), 1):
        fields = line.split()
        if not fields:
            continue
        if len(fields) != 5:
            raise ValueError(f"{path}:{line_no}: expected five YOLO fields")
        class_id = int(fields[0])
        if class_id not in LEGACY:
            continue
        x, y, w, h = (float(v) for v in fields[1:])
        if not all(0 <= v <= 1 for v in (x, y, w, h)) or w == 0 or h == 0:
            raise ValueError(f"{path}:{line_no}: invalid normalized box")
        detections.append({"class": LEGACY[class_id], "center": [x*width, y*height],
                           "box": [(x-w/2)*width, (y-h/2)*height,
                                   (x+w/2)*width, (y+h/2)*height]})
    return detections


def _assign_detections(slots, detections):
    assigned = {}
    for det in detections:
        cx, cy = det["center"]
        candidates = []
        for slot in slots:
            sx, sy = slot["center"]
            distance = ((sx-cx)**2 + (sy-cy)**2)**.5
            if slot["area"] == "scrap":
                distance = abs(sy-cy) + 0.18*abs(sx-cx)
            candidates.append((distance, slot["id"]))
        distance, slot_id = min(candidates)
        previous = assigned.get(slot_id)
        item = {**det, "distance": float(distance)}
        if previous is None or PRIORITY[item["class"]] > PRIORITY[previous["class"]]:
            assigned[slot_id] = item
    return assigned


def predict(result, pil_image, yolo_path, auto_confirm=False):
    """Populate all 60 slots and retain prediction confidence/provenance."""
    rgb = np.asarray(pil_image.convert("RGB"))
    detections = read_yolo(yolo_path, result["image_size"])
    assigned = _assign_detections(result["slots"], detections)
    scrap = [s for s in result["slots"] if s["area"] == "scrap"]
    evidence = {s["id"]: _scrap_evidence(rgb, s) for s in scrap}
    # The legacy annotations are sparse, but any scrap-slot annotation proves that the
    # robot reached that five-position column. Rounding that anchor to the column end is
    # more robust than raw brightness under the dark/green camera regime. Appearance is
    # retained as a fallback for images with no scrap annotations at all.
    anchors = [s["position"] for s in scrap if s["id"] in assigned]
    if anchors:
        scrap_through = min(30, ((max(anchors)+4)//5)*5)
    else:
        strong = [s["position"] for s in scrap if evidence[s["id"]] >= .50]
        scrap_through = max(strong, default=0)

    for slot in result["slots"]:
        yellow = _yellow_score(rgb, slot["center"])
        det = assigned.get(slot["id"])
        if det is not None:
            predicted = det["class"]
            confidence = max(.70, min(.99, 1.0-det["distance"]/220.0))
            source = "legacy_yolo"
        elif slot["area"] == "good":
            predicted = "pass" if yellow >= .012 else "no_vial"
            confidence = min(.96, .60 + abs(yellow-.012)*8)
            source = "cap_appearance"
        else:
            predicted = "pass" if slot["position"] <= scrap_through else "no_vial"
            confidence = .72 if predicted == "pass" else .82
            source = "fill_sequence_appearance"
        slot.update({"class": predicted, "presence": predicted != "no_vial",
                     "review_status": "confirmed" if auto_confirm else "draft",
                     "suggested_class": predicted,
                     "prediction_confidence": round(float(confidence), 4),
                     "prediction_source": source,
                     "reason": f"auto:{source}; confidence={confidence:.3f}"})
    result["backend"] = "legacy_yolo_plus_appearance"
    result["context"]["prediction"] = {"legacy_yolo": str(Path(yolo_path).name),
        "auto_confirmed": bool(auto_confirm), "scrap_fill_through": scrap_through,
        "scrap_evidence": {key: round(value, 5) for key, value in evidence.items()}}
    return result
