from ultralytics import YOLO
import numpy as np
import cv2
from shapely.geometry import Polygon, box
from src.models import Detection, PredictionType, Segmentation
from src.config import get_settings

SETTINGS = get_settings()


def match_gun_bbox(segment: list[list[int]], bboxes: list[list[int]], max_distance: int = 10) -> list[int] | None:
    matched_box = None
    if not segment or not bboxes:
        return None
    try:
        seg_poly = Polygon(segment)
        if not seg_poly.is_valid:
            seg_poly = seg_poly.buffer(0)
    except Exception:
        return None

    min_dist = float("inf")
    for box_coords in bboxes:
        try:
            x1, y1, x2, y2 = box_coords
        except Exception:
            continue
        gun_box = box(x1, y1, x2, y2)
        try:
            dist = seg_poly.distance(gun_box)
        except Exception:
            continue
        if dist < min_dist:
            min_dist = dist
            matched_box = [int(x1), int(y1), int(x2), int(y2)]
    if min_dist <= max_distance:
        return matched_box
    return None

def annotate_detection(image_array: np.ndarray, detection: Detection) -> np.ndarray:
    ann_color = (255, 0, 0)
    annotated_img = image_array.copy()
    for label, conf, box in zip(detection.labels, detection.confidences, detection.boxes):
        x1, y1, x2, y2 = box
        cv2.rectangle(annotated_img, (x1, y1), (x2,y2), ann_color, 3)
        cv2.putText(
            annotated_img,
            f"{label}: {conf:.1f}",
            (x1, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            ann_color,
            2,
        )
    return annotated_img


def annotate_segmentation(image_array: np.ndarray, segmentation: Segmentation, draw_boxes: bool = True) -> np.ndarray:
    if segmentation is None:
        return image_array
    annotated_img = image_array.copy()
    h, w = annotated_img.shape[:2]
    for idx, poly in enumerate(segmentation.polygons or []):
        try:
            label = segmentation.labels[idx] if segmentation.labels and idx < len(segmentation.labels) else "safe"
        except Exception:
            label = "safe"
        if str(label).lower() == "danger":
            color = (0, 0, 255)
        else:
            color = (0, 255, 0)
        try:
            pts = np.array(poly, dtype=np.int32)
            if pts.ndim == 2 and pts.shape[0] >= 3:
                overlay = annotated_img.copy()
                cv2.fillPoly(overlay, [pts], color)
                alpha = 0.4
                annotated_img = cv2.addWeighted(overlay, alpha, annotated_img, 1 - alpha, 0)
        except Exception:
            continue

        if draw_boxes and segmentation.boxes and idx < len(segmentation.boxes):
            try:
                x1, y1, x2, y2 = [int(v) for v in segmentation.boxes[idx]]
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                cv2.putText(
                    annotated_img,
                    str(label),
                    (max(0, x1), max(15, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    color,
                    2,
                )
            except Exception:
                pass

    return annotated_img


class GunDetector:
    def __init__(self) -> None:
        print(f"loading od model: {SETTINGS.od_model_path}")
        self.od_model = YOLO(SETTINGS.od_model_path)
        print(f"loading seg model: {SETTINGS.seg_model_path}")
        self.seg_model = YOLO(SETTINGS.seg_model_path)

    def detect_guns(self, image_array: np.ndarray, threshold: float = 0.5):
        results = self.od_model(image_array, conf=threshold)[0]
        labels = results.boxes.cls.tolist()
        indexes = [
            i for i in range(len(labels)) if labels[i] in [3, 4]
        ]  # 0 = "person"
        boxes = [
            [int(v) for v in box]
            for i, box in enumerate(results.boxes.xyxy.tolist())
            if i in indexes
        ]
        confidences = [
            c for i, c in enumerate(results.boxes.conf.tolist()) if i in indexes
        ]
        labels_txt = [
            results.names[labels[i]] for i in indexes
        ]
        return Detection(
            pred_type=PredictionType.object_detection,
            n_detections=len(boxes),
            boxes=boxes,
            labels=labels_txt,
            confidences=confidences,
        )
    
    def segment_people(self, image_array: np.ndarray, threshold: float = 0.5, max_distance: int = 10):
        detection = self.detect_guns(image_array, threshold)
        gun_boxes = detection.boxes if detection and detection.boxes else []
        results = self.seg_model(image_array, conf=threshold)[0]

        polygons: list[list[list[int]]] = []
        boxes: list[list[int]] = []
        labels: list[str] = []
        try:
            seg_labels = results.boxes.cls.tolist()
            seg_boxes_xy = results.boxes.xyxy.tolist()
        except Exception:
            seg_labels = []
            seg_boxes_xy = []
        mask_polys = None
        if hasattr(results, "masks"):
            if hasattr(results.masks, "xy"):
                try:
                    mask_polys = [p.tolist() for p in results.masks.xy]
                except Exception:
                    mask_polys = None
            elif hasattr(results.masks, "xys"):
                try:
                    mask_polys = [p.tolist() for p in results.masks.xys]
                except Exception:
                    mask_polys = None
        for i, cls in enumerate(seg_labels):
            try:
                if int(cls) != 0:
                    continue
            except Exception:
                continue
            poly_pts: list[list[int]] | None = None
            if mask_polys is not None and i < len(mask_polys):
                try:
                    poly_pts = [[int(x), int(y)] for x, y in mask_polys[i]]
                except Exception:
                    poly_pts = None
            if poly_pts is None and i < len(seg_boxes_xy):
                x1, y1, x2, y2 = seg_boxes_xy[i]
                poly_pts = [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]]

            if poly_pts is None:
                continue
            xs = [p[0] for p in poly_pts]
            ys = [p[1] for p in poly_pts]
            bbox = [min(xs), min(ys), max(xs), max(ys)]

            matched = match_gun_bbox(poly_pts, gun_boxes, max_distance=max_distance)
            label = "danger" if matched is not None else "safe"

            polygons.append(poly_pts)
            boxes.append([int(v) for v in bbox])
            labels.append(label)

        return Segmentation(
            pred_type=PredictionType.segmentation,
            n_detections=len(labels),
            polygons=polygons,
            boxes=boxes,
            labels=labels,
        )       

