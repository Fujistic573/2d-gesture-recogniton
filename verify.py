"""Verification script for multi-stroke recognition and GUI."""
import ast
import json
import math
import numpy as np

for f in ["preprocessing.py", "knn_classifier.py", "generate_dataset.py", "app.py"]:
    with open(f, encoding="utf-8") as fh:
        ast.parse(fh.read())
    print(f"  syntax OK: {f}")

from preprocessing import normalize_strokes
from knn_classifier import KNNClassifier
import app

with open("data/dataset.json", encoding="utf-8") as fh:
    data = json.load(fh)
print(f"  dataset entries: {len(data)}")

clf = KNNClassifier(k=5, dataset_path="data/dataset.json")

tests = [
    ("Circle CW from top",   [[(200 + 100 * math.cos(a), 200 + 100 * math.sin(a)) for a in np.linspace(-math.pi / 2, -math.pi / 2 - 2 * math.pi, 50)]], "circle"),
    ("Circle CCW from left", [[(200 + 100 * math.cos(a), 200 + 100 * math.sin(a)) for a in np.linspace(math.pi, 3 * math.pi, 50)]], "circle"),
    ("Square 4 strokes",     [[(0, 0), (100, 0)], [(100, 0), (100, 100)], [(100, 100), (0, 100)], [(0, 100), (0, 0)]], "square"),
    ("Triangle 3 strokes",   [[(50, 0), (100, 90)], [(100, 90), (0, 90)], [(0, 90), (50, 0)]], "triangle"),
    ("Cross (+) 2 strokes",  [[(0, 50), (100, 50)], [(50, 0), (50, 100)]], "cross"),
    ("X-shape 2 strokes",    [[(0, 0), (100, 100)], [(100, 0), (0, 100)]], "x_shape"),
    ("Line H right-to-left", [[(200, 100), (50, 100)]], "line_h"),
    ("Line V bottom-to-top", [[(100, 200), (100, 50)]], "line_v"),
    ("Checkmark",            [[(10, 50), (40, 90), (120, 10)]], "check"),
    ("Diamond",              [[(50, 0), (100, 50), (50, 100), (0, 50), (50, 0)]], "diamond"),
]

passed = 0
for name, strokes, expected in tests:
    pred, conf = clf.predict(normalize_strokes(strokes))
    ok = pred == expected
    if ok:
        passed += 1
    tag = "OK" if ok else f"FAIL expected {expected}"
    print(f"  {name:24s} -> {pred:12s} ({conf:.0%})  [{tag}]")

print(f"Score: {passed}/{len(tests)}")

# GUI smoke test (multi-stroke + undo + predict)
a = app.GestureApp()
a._strokes = [[(0, 50), (100, 50)], [(50, 0), (50, 100)]]
a._redraw_all_strokes()
a._draw_preview()
a._predict()
a.update_idletasks()
print("GUI multi-stroke status:", a._status.cget("text"))
a._undo()
a.update_idletasks()
print("GUI after undo status:", a._status.cget("text"))
a.destroy()
