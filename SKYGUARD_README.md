# 🛡️ SkyGuard — System Detekcji Zagrożeń Powietrznych

Detekcja dronów (Shahed-136, UAV kamikaze), ptaków i ludzi na obrazach z wykorzystaniem modeli YOLOv8.

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-purple?logo=yolo)
![OpenCV](https://img.shields.io/badge/OpenCV-4.x-green?logo=opencv)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## 📋 Opis

SkyGuard to narzędzie do wykrywania i klasyfikacji obiektów powietrznych z automatyczną oceną poziomu zagrożenia. System rozróżnia:

| Obiekt | Ocena zagrożenia |
|--------|-----------------|
| 🔴 Shahed-136 | **WYSOKIE ZAGROŻENIE** — dron kamikaze |
| ⚠️ Dron / UAV | **ZAGROŻENIE** — możliwy UAV bojowy |
| ⚠️ Samolot | **UWAGA** — wymaga identyfikacji |
| ⚠️ Człowiek | **UWAGA** — osoba wykryta w strefie |
| ✅ Ptak | **BRAK ZAGROŻENIA** — obiekt niegroźny |

---

## 🚀 Szybki start

### Instalacja

```bash
pip install opencv-python ultralytics roboflow
```

### Użycie

Skrypt obsługuje **3 tryby pracy**:

```bash
# 1. Szybka detekcja (działa od razu, bez treningu)
python temp.py --mode quick --image zdjecie.webp

# 2. Trening własnego modelu na datasecie drone/bird
python temp.py --mode train --api-key TWOJ_KLUCZ_ROBOFLOW

# 3. Inferencja wytrenowanym modelem (po treningu)
python temp.py --mode infer --image zdjecie.webp
```

---

## 🔧 Tryby pracy

### 1️⃣ `--mode quick` — Szybka detekcja

Używa pretrenowanego modelu **YOLOv8x** (zbiór COCO, 80 klas) z inteligentnym mapowaniem etykiet:

- `kite` (latawiec) → **DRON** (YOLO myli Shahedy z latawcami ze względu na kształt delta)
- `bird` → **PTAK**
- `airplane` → **SAMOLOT**
- `person` → **CZŁOWIEK**

**Zalety:** działa natychmiast, bez treningu  
**Wady:** ograniczona dokładność — model nie był trenowany na dronach

```bash
python temp.py --mode quick
python temp.py --mode quick --image moje_zdjecie.jpg
```

### 2️⃣ `--mode train` — Trening dedykowanego modelu

Pobiera dataset **„Drone vs Bird Detection"** z Roboflow (~9.85k obrazów) i trenuje YOLOv8x od zera.

**Wymagania:**
- Darmowe konto na [Roboflow](https://app.roboflow.com/) → klucz API
- Czas: 30 min – kilka godzin (zależy od CPU/GPU)
- Miejsce na dysku: ~2 GB (dataset + model)

```bash
python temp.py --mode train --api-key TWOJ_KLUCZ_ROBOFLOW
```

**Parametry treningu:**

| Parametr | Wartość | Opis |
|----------|---------|------|
| Model bazowy | `yolov8x.pt` | Extra Large — najdokładniejszy |
| Epoki | 50 | Ilość przejść przez dataset |
| Rozmiar obrazu | 640×640 | Rozdzielczość wejściowa |
| Batch size | 8 | Dostosowany do CPU/8GB RAM |
| Patience | 10 | Early stopping — zatrzymanie jeśli brak poprawy |

Wytrenowany model zostaje zapisany w `runs/detect/drone_detector/weights/best.pt`.

### 3️⃣ `--mode infer` — Inferencja wytrenowanym modelem

Używa modelu z trybu `train` — najdokładniejsze wyniki.

```bash
python temp.py --mode infer
python temp.py --mode infer --image inne_zdjecie.png
```

---

## 📁 Struktura projektu

```
.
├── temp.py                     # Główny skrypt
├── test.webp                   # Przykładowy obraz testowy
├── datasets/
│   └── drone_bird/             # Dataset (pobierany automatycznie)
│       ├── data.yaml
│       ├── train/
│       ├── valid/
│       └── test/
└── runs/
    └── detect/
        └── drone_detector/     # Wyniki treningu
            ├── weights/
            │   ├── best.pt     # Najlepszy model
            │   └── last.pt     # Ostatni checkpoint
            ├── results.png     # Wykresy metryk
            └── labels.jpg      # Rozkład klas
```

---

## ⚙️ Konfiguracja

Edytuj stałe na górze `temp.py`:

```python
IMAGE_PATH = "test.webp"                              # Domyślny obraz
TRAINED_MODEL_DIR = "runs/detect/drone_detector"      # Katalog modelu
```

### Dodawanie własnych klas zagrożeń

```python
THREAT_LEVELS = {
    "drone":    ("⚠️  ZAGROŻENIE",        "DRON (możliwy Shahed / UAV kamikaze)"),
    "shahed":   ("🔴 WYSOKIE ZAGROŻENIE", "DRON SHAHED-136 — dron kamikaze!"),
    "bird":     ("✅ BRAK ZAGROŻENIA",    "PTAK — obiekt niegroźny"),
    "person":   ("⚠️  UWAGA",            "CZŁOWIEK wykryty w strefie"),
    # Dodaj własne:
    "helicopter": ("⚠️  UWAGA",          "HELIKOPTER — wymaga identyfikacji"),
}
```

---

## 🐛 Znane problemy i rozwiązania

### Błąd OpenMP na Windows
```
OMP: Error #15: Initializing libomp.dll, but found libiomp5md.dll already initialized.
```
**Rozwiązanie:** Już obsłużone w skrypcie (`KMP_DUPLICATE_LIB_OK=TRUE`). Typowy problem z conda + PyTorch na Windows.

### YOLO klasyfikuje drony jako „kite" (latawiec)
**Przyczyna:** Model COCO nie zna kategorii „dron". Trójkątny kształt Shahed-136 przypomina latawiec.  
**Rozwiązanie:** Użyj trybu `train` do wytrenowania dedykowanego modelu.

### Wolny trening na CPU
**Rozwiązanie:** Zmniejsz parametry w `train_model()`:
```python
model.train(
    epochs=20,      # mniej epok
    imgsz=416,      # mniejszy obraz
    batch=4,        # mniejszy batch
)
```
Lub użyj GPU (CUDA):
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## 📦 Wymagania

| Pakiet | Wersja | Opis |
|--------|--------|------|
| Python | ≥ 3.10 | |
| ultralytics | ≥ 8.0 | YOLOv8 |
| opencv-python | ≥ 4.0 | Przetwarzanie obrazów |
| roboflow | ≥ 1.0 | Pobieranie datasetu (tylko tryb train) |

### Instalacja pełna

```bash
pip install opencv-python ultralytics roboflow
```

### Instalacja minimalna (tylko tryb quick)

```bash
pip install opencv-python ultralytics
```

---

## 📊 Dataset

Tryb `train` korzysta z datasetu **[Drone vs Bird Detection](https://universe.roboflow.com/drone-detection-project/drone-vs-bird-detection)** z Roboflow Universe:

- **9 850 obrazów** (train/valid/test)
- **2 klasy:** Drones, Birds
- **Licencja:** CC BY 4.0
- **Format:** YOLOv8

Aby użyć innego datasetu (np. z klasą Shahed-136), zmień `project` i `version` w funkcji `train_model()`.

Rekomendowane datasety na Roboflow:
- [Shahed (7.6k img, bird/notshahed)](https://universe.roboflow.com/e-yjnj4/shahed-y4fsd) — rozpoznaje Shahedy vs ptaki
- [Shahed 136 (5k img)](https://universe.roboflow.com/horus-al939/shahed-136-m2zkt) — dedykowany Shahed-136
- [Drone Detection (34k img)](https://universe.roboflow.com/dronedetectionpitt-nwyps/drone-detection-yhkcr) — ogólna detekcja dronów

---

## 🗺️ Roadmap

- [ ] Detekcja w czasie rzeczywistym z kamery / streamu RTSP
- [ ] Integracja z systemem alarmowym (webhook / MQTT)
- [ ] Śledzenie obiektów (tracking) — identyfikacja trajektorii lotu
- [ ] Eksport modelu do ONNX / TensorRT dla edge devices (Jetson Nano, Raspberry Pi)
- [ ] Multi-dataset training — połączenie datasetu dron/ptak/Shahed
- [ ] REST API do integracji z innymi systemami

---

## 📄 Licencja

MIT License — używaj, modyfikuj, dystrybuuj bez ograniczeń.

---

## 🙏 Źródła i podziękowania

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [Roboflow Universe](https://universe.roboflow.com/) — datasety
- [Drone vs Bird Detection Dataset](https://universe.roboflow.com/drone-detection-project/drone-vs-bird-detection) (CC BY 4.0)
