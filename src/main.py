import io
import cv2
from fastapi import FastAPI, UploadFile, File, HTTPException, status, Depends
from fastapi.responses import Response
import numpy as np
from functools import cache
from PIL import Image, UnidentifiedImageError
from src.predictor import GunDetector, Detection, Segmentation, annotate_detection, annotate_segmentation, match_gun_bbox
from src.models import Gun, PixelLocation, GunType, Person, PersonType
from src.config import get_settings

SETTINGS = get_settings()

app = FastAPI(title=SETTINGS.api_name, version=SETTINGS.revision)


@cache
def get_gun_detector() -> GunDetector:
    print("Creating model...")
    return GunDetector()


def detect_uploadfile(detector: GunDetector, file, threshold) -> tuple[Detection, np.ndarray]:
    img_stream = io.BytesIO(file.file.read())
    if file.content_type.split("/")[0] != "image":
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Not an image"
        )
    # convertir a una imagen de Pillow
    try:
        img_obj = Image.open(img_stream)
    except UnidentifiedImageError:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, detail="Image format not suported"
        )
    # crear array de numpy
    img_array = np.array(img_obj)
    return detector.detect_guns(img_array, threshold), img_array


@app.get("/model_info")
def get_model_info(detector: GunDetector = Depends(get_gun_detector)):
    return {
        "model_name": "Gun detector",
        "gun_detector_model": detector.od_model.model.__class__.__name__,
        "semantic_segmentation_model": detector.seg_model.model.__class__.__name__,
        "input_type": "image",
    }


@app.post("/detect_guns")
def detect_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Detection:
    results, _ = detect_uploadfile(detector, file, threshold)

    return results


@app.post("/detect_people")
def detect_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Segmentation:
    """Retorna la segmentación de personas etiquetadas como 'safe' o 'danger'."""
    _, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold, max_distance=max_distance)
    return segmentation


@app.post("/detect")
def detect_both(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> dict:
    """Retorna un diccionario con la detección de armas y la segmentación de personas."""
    detection, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold, max_distance=max_distance)
    return {"detection": detection, "segmentation": segmentation}


@app.post("/annotate_people")
def annotate_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    """Retorna una imagen con anotaciones de segmentación."""
    _, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold, max_distance=max_distance)
    annotated_img = annotate_segmentation(img, segmentation, draw_boxes=draw_boxes)
    img_rgb = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/annotate_guns")
def annotate_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    detection, img = detect_uploadfile(detector, file, threshold)
    annotated_img = annotate_detection(img, detection)

    img_pil = Image.fromarray(annotated_img)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")


@app.post("/annotate")
def annotate(
    threshold: float = 0.5,
    max_distance: int = 10,
    draw_boxes: bool = True,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> Response:
    """Anota ambas detecciones (armas) y segmentaciones (personas) en la misma imagen."""
    detection, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold, max_distance=max_distance)
    annotated = annotate_detection(img.copy(), detection)
    annotated = annotate_segmentation(annotated, segmentation, draw_boxes=draw_boxes)
    try:
        annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    except Exception:
        annotated_rgb = annotated
    img_pil = Image.fromarray(annotated_rgb)
    image_stream = io.BytesIO()
    img_pil.save(image_stream, format="JPEG")
    image_stream.seek(0)
    return Response(content=image_stream.read(), media_type="image/jpeg")

@app.post("/guns")
def get_guns(
    threshold: float = 0.5,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Gun]:
    """Retorna una lista de armas con tipo y ubicación en píxeles."""
    detection, _ = detect_uploadfile(detector, file, threshold)
    guns: list[Gun] = []
    for label, box in zip(detection.labels, detection.boxes):
        try:
            x1, y1, x2, y2 = [int(v) for v in box]
        except Exception:
            continue
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        try:
            gun_type = GunType(label)
        except Exception:
            try:
                gun_type = GunType(str(label).lower())
            except Exception:
                continue
        guns.append(Gun(gun_type=gun_type, location=PixelLocation(x=cx, y=cy)))
    return guns
    

@app.post("/people")
def get_people(
    threshold: float = 0.5,
    max_distance: int = 10,
    file: UploadFile = File(...),
    detector: GunDetector = Depends(get_gun_detector),
) -> list[Person]:
    """Retorna una lista de personas con categoría, ubicación central y área en píxeles."""
    _, img = detect_uploadfile(detector, file, threshold)
    segmentation = detector.segment_people(img, threshold, max_distance=max_distance)

    people: list[Person] = []
    for idx, poly in enumerate(segmentation.polygons or []):
        try:
            pts = np.array(poly, dtype=np.int32)
            if pts.ndim != 2 or pts.shape[0] < 3:
                continue
            cx = int(np.mean(pts[:, 0]))
            cy = int(np.mean(pts[:, 1]))
            area = int(abs(cv2.contourArea(pts)))
            label = segmentation.labels[idx] if segmentation.labels and idx < len(segmentation.labels) else "safe"
            try:
                person_type = PersonType(label)
            except Exception:
                try:
                    person_type = PersonType(str(label).lower())
                except Exception:
                    person_type = PersonType.safe
            people.append(
                Person(person_type=person_type, location=PixelLocation(x=cx, y=cy), area=area)
            )
        except Exception:
            continue
    return people
    
# Es un metodo para probar solamente la funcion match_gun_bbox, como debugger
@app.get("/test_match_gun_bbox")
def test_match(detector: GunDetector = Depends(get_gun_detector)):
    segment = [[100, 100], [150, 100], [150, 150], [100, 150]]
    bboxes = [
        [200, 200, 250, 250],
        [155, 120, 180, 140],
    ]
    result = match_gun_bbox(segment, bboxes, max_distance=20)
    return {"matched_box": result}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("src.app:app", port=8080, host="0.0.0.0")
