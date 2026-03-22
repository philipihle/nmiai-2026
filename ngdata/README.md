# NorgesGruppen Data — Object Detection (Oppgave 3)

NM i AI 2026. Object detection av dagligvareprodukter på hyllebilder.

## Resultater

| Submission | Modell | Teknikk | Score |
|---|---|---|---|
| submission-legacy | YOLO11x + 5-tile + NMS + conf=0.15 | Baseline | **0.8774** |
| submission-v8 | YOLO11x (all data) + classifier + NMS | EfficientNet-B3 reclassifier | 0.8647 |
| submission-v9 | YOLO11x (all data) + NMS + conf=0.10 | Lavere terskel | 0.8626 |
| submission-v11 | YOLO11x + WBF + TTA flip + conf=0.15 | WBF + TTA | 0.8690 |
| submission-v10 | YOLO11x + NMS + conf=0.10 | Lavere terskel, gammel modell | 0.8547 |

Beste score: **0.8774** (leder: 0.9255)

## Arkitektur

- **Detektor**: YOLO11x trent på 248 treningsbilder, eksportert til ONNX (IR v8)
- **Tiling**: Full bilde + 4 overlappende kvadranter (20% overlap)
- **NMS**: Hard NMS (klasse-agnostisk) — ga bedre resultater enn WBF og Soft-NMS
- **Classifier** (forsøkt): EfficientNet-B3 fintunt på 18420 ekte crops — skadet score konsekvent

## Filer

### Inference (submission)
- `run_v7.py` — Tiling + NMS (baseline)
- `run_v8.py` — + EfficientNet-B3 classifier
- `run_v9.py` — Uten classifier, conf=0.10
- `run_v11.py` — WBF (ensemble_boxes) + TTA horizontal flip
- `run_ensemble.py` — To-modell ensemble (FP16)

### Trening
- `train.py` — YOLO trening (Ultralytics)
- `train_classifier.py` — EfficientNet-B3 på produktbilder
- `train_classifier_v2.py` — EfficientNet-B3 på ekte crops fra treningsbilder
- `extract_crops.py` — Ekstraher crops fra YOLO ground truth labels

### Prosjektstruktur
- `detector/` — Katalog, COCO-konvertering, inference, validering, visualisering
- `tests/` — Enhetstester

## Lærdom

1. **Klassifikator skadet alltid** — YOLO trent end-to-end var bedre enn post-hoc EfficientNet-B3
2. **Lavere conf-terskel skadet** — Flere falske positiver > flere riktige deteksjoner
3. **WBF + TTA skadet** — Sannsynligvis fordi tile-basert WBF ikke er optimalt (WBF er designet for multi-modell, ikke multi-tile)
4. **Enkel pipeline vant** — Beste score var den enkleste konfigurasjonen
