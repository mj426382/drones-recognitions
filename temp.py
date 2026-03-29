"""
System detekcji obiektów powietrznych — dron / ptak / człowiek / niegroźny obiekt.

Dwie opcje:
  1) TRYB SZYBKI (--mode quick)  — używa YOLOv8x + mapowanie etykiet COCO
  2) TRYB TRENING (--mode train)  — pobiera dataset z Roboflow, trenuje własny model
  3) TRYB INFERENCJA (--mode infer) — używa wytrenowanego modelu z trybu train

Użycie:
  python temp.py --mode quick               # szybka detekcja na test.webp
  python temp.py --mode train --api-key KLUCZ  # trening własnego modelu
  python temp.py --mode infer               # inferencja wytrenowanym modelem
"""

import argparse
import os
import sys
from pathlib import Path

# Fix: konflikt OpenMP (libomp.dll vs libiomp5md.dll) na Windows
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import cv2
from ultralytics import YOLO

# ──────────────────────────────────────────────
# Konfiguracja
# ──────────────────────────────────────────────
IMAGE_PATH = "test6.webp"
TRAINED_MODEL_DIR = "runs/detect/drone_detector"
TRAINED_MODEL_PATH = os.path.join(TRAINED_MODEL_DIR, "weights", "best.pt")

# Klasy zagrożeń
THREAT_LEVELS = {
    "drone":    ("⚠️  ZAGROŻENIE",  "DRON (możliwy Shahed / UAV kamikaze)"),
    "shahed":   ("🔴 WYSOKIE ZAGROŻENIE", "DRON SHAHED-136 — dron kamikaze!"),
    "bird":     ("✅ BRAK ZAGROŻENIA",    "PTAK — obiekt niegroźny"),
    "person":   ("⚠️  UWAGA",             "CZŁOWIEK wykryty w strefie"),
    "airplane": ("⚠️  UWAGA",             "SAMOLOT — wymaga identyfikacji"),
    "kite":     ("⚠️  ZAGROŻENIE",        "LATAWIEC/DRON — możliwy UAV!"),
}

# Mapowanie klas COCO → nasze kategorie (tryb quick)
COCO_TO_CATEGORY = {
    "bird":     "bird",
    "kite":     "drone",     # YOLO myli Shahedy z latawcami
    "airplane": "airplane",
    "person":   "person",
}


def assess_threat(label: str) -> tuple[str, str]:
    """Zwraca (poziom_zagrożenia, opis) dla danej etykiety."""
    label_lower = label.lower().strip()
    for key, value in THREAT_LEVELS.items():
        if key in label_lower:
            return value
    return ("✅ BRAK ZAGROŻENIA", f"{label} — obiekt niegroźny")


def print_header(title: str):
    print("\n" + "=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ──────────────────────────────────────────────
# TRYB 1: Szybka detekcja z YOLOv8x + mapowanie
# ──────────────────────────────────────────────
def quick_detect(image_path: str):
    print_header("TRYB SZYBKI — YOLOv8x + mapowanie klas COCO")

    model = YOLO("yolov8x.pt")
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"❌ Nie udało się wczytać obrazu: {image_path}")
        return

    results = model(frame)

    print(f"\n📷 Obraz: {image_path}")
    print("-" * 60)

    detections = []
    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            coco_label = model.names[cls_id]

            # Mapuj na nasze kategorie
            category = COCO_TO_CATEGORY.get(coco_label)
            if category:
                threat_level, description = assess_threat(category)
                detections.append((category, confidence, threat_level, description))
                print(f"  {threat_level}: {description}")
                print(f"     Pewność: {confidence:.1%} | COCO label: '{coco_label}'")
                print()

    if not detections:
        print("  Nie wykryto obiektów pasujących do kategorii dron/ptak/człowiek.")
        print("  Spróbuj trybu 'train' dla lepszych wyników.\n")

    # Pokaż obraz z adnotacjami
    annotated_frame = results[0].plot()
    cv2.imshow("Detekcja zagrozen powietrznych", annotated_frame)
    print("Naciśnij dowolny klawisz w oknie obrazu, aby zamknąć.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# TRYB 2: Trening własnego modelu
# ──────────────────────────────────────────────
def train_model(api_key: str):
    print_header("TRYB TRENINGU — Pobieranie datasetu i trening modelu")

    try:
        from roboflow import Roboflow
    except ImportError:
        print("❌ Brak pakietu 'roboflow'. Zainstaluj:")
        print("   pip install roboflow")
        sys.exit(1)

    # Pobierz dataset "Drone vs Bird Detection" z Roboflow
    print("\n📥 Pobieranie datasetu 'Drone vs Bird Detection'...")
    print("   (9.85k obrazów, klasy: Drones, Birds)\n")

    rf = Roboflow(api_key=api_key)
    project = rf.workspace("drone-detection-project").project("drone-vs-bird-detection")
    version = project.version(5)
    dataset = version.download("yolov8", location="datasets/drone_bird")

    # Trenuj model YOLOv8x na pobranym datasecie
    print("\n🏋️ Rozpoczynam trening YOLOv8x...")
    print("   To może potrwać od 30 min do kilku godzin.\n")

    model = YOLO("yolov8x.pt")
    model.train(
        data=os.path.join("datasets", "drone_bird", "data.yaml"),
        epochs=50,
        imgsz=640,
        batch=8,
        name="drone_detector",
        patience=10,
        save=True,
        plots=True,
    )

    print(f"\n✅ Model wytrenowany! Zapisano w: {TRAINED_MODEL_DIR}/")
    print(f"   Użyj: python temp.py --mode infer")


# ──────────────────────────────────────────────
# TRYB 3: Inferencja wytrenowanym modelem
# ──────────────────────────────────────────────
def infer_trained(image_path: str):
    print_header("TRYB INFERENCJI — Wytrenowany model dron/ptak")

    if not os.path.exists(TRAINED_MODEL_PATH):
        print(f"❌ Nie znaleziono wytrenowanego modelu: {TRAINED_MODEL_PATH}")
        print("   Najpierw uruchom: python temp.py --mode train --api-key KLUCZ")
        return

    model = YOLO(TRAINED_MODEL_PATH)
    frame = cv2.imread(image_path)

    if frame is None:
        print(f"❌ Nie udało się wczytać obrazu: {image_path}")
        return

    results = model(frame)

    print(f"\n📷 Obraz: {image_path}")
    print("-" * 60)

    for result in results:
        for box in result.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            label = model.names[cls_id]

            threat_level, description = assess_threat(label)
            print(f"  {threat_level}: {description}")
            print(f"     Pewność: {confidence:.1%}")
            print()

    # Pokaż obraz z adnotacjami
    annotated_frame = results[0].plot()
    cv2.imshow("Wytrenowany model - Detekcja zagrozen", annotated_frame)
    print("Naciśnij dowolny klawisz w oknie obrazu, aby zamknąć.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="System detekcji obiektów powietrznych")
    parser.add_argument(
        "--mode",
        choices=["quick", "train", "infer"],
        default="quick",
        help="Tryb: quick (szybki z COCO), train (trening), infer (wytrenowany model)",
    )
    parser.add_argument("--image", default=IMAGE_PATH, help="Ścieżka do obrazu")
    parser.add_argument("--api-key", default=None, help="Klucz API Roboflow (do trybu train)")

    args = parser.parse_args()

    if args.mode == "quick":
        quick_detect(args.image)
    elif args.mode == "train":
        if not args.api_key:
            print("❌ Tryb 'train' wymaga klucza API Roboflow.")
            print("   Zarejestruj się na https://app.roboflow.com/ (darmowe)")
            print("   Użycie: python temp.py --mode train --api-key TWOJ_KLUCZ")
            sys.exit(1)
        train_model(args.api_key)
    elif args.mode == "infer":
        infer_trained(args.image)
