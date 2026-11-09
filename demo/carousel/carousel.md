# Vehicle detection · carousel

Spec: film/specs/capstone1-vehicle-detection.json · 8 slides, 1080x1350.

## 1/8 Hook

- One frame, one box, one name.

## 2/8 The question

- Car is 59% of every frame.
- Will it still see a bus?
- car 528

## 3/8 What it does

- Vehicle detection
- Capstone 1 · Intelligent transport systems
- A box and a name in about 60 ms
- max@baluyot ~ $ curl -s -F file=@data/capstone1/Images/00000001.jpg localhost:8101/predict

## 4/8 How it is built

- 1. Frames and boxes — pandas · pillow
- 2. MobileNetV2 features — tensorflow
- 3. Class head + box head — tensorflow
- 4. POST /predict — FastAPI
- 5. Two pages — HTML + ui.css
- Weight rare classes so it won't just say car

## 5/8 The result

- 3 models on the same holdout
- Majority class 58.7%
- Notebook CNN 64.5%
- MobileNetV2 + two heads 73.0% (winner)

## 6/8 The proof

- Detect: bus, one hundred percent, boxed
- screen: capstone1-vehicle-detection/demo/poster.png

## 7/8 The answer and the limit

- 73.0%
- val accuracy · 900 held-out frames · baseline 58.7%
- One box per frame. IoU 0.25: boxes are loose.
- Notebook 64.5 → 73.0 here. And it ships.

## 8/8 Try it

- Try it: sample 42, the near-empty road.
- max@baluyot ~ $ git clone https://github.com/Keshigami/capstone1-vehicle-detection.git
- max@baluyot ~ $ just setup
- max@baluyot ~ $ just up
- github.com/Keshigami/capstone1-vehicle-detection
- Maximus Friedrich Baluyot
- AI & ML Specialist · Applied AI · ML Engineering · Automation

_Maximus Friedrich Baluyot · Caltech CTME · PGC AIML_
