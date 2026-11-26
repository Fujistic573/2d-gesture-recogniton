"""
generate_dataset.py
~~~~~~~~~~~~~~~~~~~
Generates a synthetic training dataset of 24 shapes (geometric + non-geometric)
with realistic noise, scale variation and slight rotation per sample.

Run once before starting the app:
    python generate_dataset.py

Each shape gets ``SAMPLES_PER_CLASS`` examples written to data/dataset.json.
The file is fully replaced on every run.
"""

from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Callable

import numpy as np

from preprocessing import normalize, N_POINTS

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SAMPLES_PER_CLASS = 40          # examples stored per shape
NOISE_STD         = 0.010       # Gaussian noise on normalised vector
OUTPUT_PATH       = Path("data/dataset.json")
RANDOM_SEED       = 42

CLOSED_SHAPES = {
    "circle", "ellipse", "square", "rectangle", "parallelogram", "trapezoid",
    "triangle", "pentagon", "hexagon", "diamond",
    "star", "star6", "heart", "infinity", "crescent", "cloud", "hourglass", "crown",
}

BIDIRECTIONAL_SHAPES = {
    "line_h", "line_v", "zigzag", "wave", "v_shape",
    "u_shape", "l_shape", "s_shape", "check", "cross", "x_shape", "lightning",
}

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ---------------------------------------------------------------------------
# Stroke generators  (each returns a list of (x,y) in arbitrary pixel coords)
# ---------------------------------------------------------------------------

Stroke = list[tuple[float, float]]


def _densify(pts: Stroke, per_seg: int = 12) -> Stroke:
    """Subdivide each straight segment so start-point shifts are smooth."""
    out: Stroke = []
    for i in range(len(pts) - 1):
        x0, y0 = pts[i]
        x1, y1 = pts[i + 1]
        for j in range(per_seg):
            t = j / per_seg
            out.append((x0 + t * (x1 - x0), y0 + t * (y1 - y0)))
    out.append(pts[-1])
    return out


def _poly(n: int, r: float = 100.0) -> Stroke:
    """Regular polygon with n sides, centred at origin."""
    verts = [
        (r * math.cos(2 * math.pi * i / n - math.pi / 2),
         r * math.sin(2 * math.pi * i / n - math.pi / 2))
        for i in range(n + 1)
    ]
    return _densify(verts)


def _arc(start: float, end: float, r: float = 100.0, steps: int = 80) -> Stroke:
    angles = np.linspace(start, end, steps)
    return [(r * math.cos(a), r * math.sin(a)) for a in angles]


def gen_circle() -> Stroke:
    return _arc(0, 2 * math.pi)


def gen_ellipse() -> Stroke:
    rx, ry = 120, 70
    angles = np.linspace(0, 2 * math.pi, 80)
    return [(rx * math.cos(a), ry * math.sin(a)) for a in angles]


def gen_square() -> Stroke:
    return _densify([(-100, -100), (100, -100), (100, 100), (-100, 100), (-100, -100)])


def gen_triangle() -> Stroke:
    return _poly(3)


def gen_pentagon() -> Stroke:
    return _poly(5)


def gen_hexagon() -> Stroke:
    return _poly(6)


def gen_diamond() -> Stroke:
    r = 100
    return [(0, -r), (r, 0), (0, r), (-r, 0), (0, -r)]


def gen_star(pentagram: bool = False) -> Stroke:
    """5-pointed star (outline or intersecting pentagram)."""
    if pentagram:
        # 5-stroke intersecting star starting from bottom-left -> top -> ...
        verts = [
            (-59, 81), (0, -100), (59, 81),
            (-95, -31), (95, -31), (-59, 81),
        ]
        return _densify(verts, per_seg=10)
    pts: Stroke = []
    for i in range(11):
        angle = -math.pi / 2 + i * math.pi / 5
        r = 100 if i % 2 == 0 else 40
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return _densify(pts, per_seg=6)


def gen_star6() -> Stroke:
    """6-pointed star (Star of David outline)."""
    pts: Stroke = []
    for i in range(13):
        angle = math.pi / 2 + i * math.pi / 6
        r = 100 if i % 2 == 0 else 50
        pts.append((r * math.cos(angle), r * math.sin(angle)))
    return pts


def gen_arrow_right() -> Stroke:
    return _densify([
        (-100, 0), (100, 0), (45, -45), (100, 0), (45, 45),
    ], per_seg=8)


def gen_arrow_up() -> Stroke:
    return _densify([
        (0, 100), (0, -100), (-45, -45), (0, -100), (45, -45),
    ], per_seg=8)


def gen_check() -> Stroke:
    """Checkmark / tick."""
    return [(-80, 0), (-30, 70), (90, -80)]


def gen_cross() -> Stroke:
    """Plus / cross (+)."""
    return [
        (0, -100), (0, 0), (100, 0), (0, 0),
        (0, 100), (0, 0), (-100, 0),
    ]


def gen_x_shape() -> Stroke:
    """Letter-X diagonal cross."""
    return [
        (-80, -80), (0, 0), (80, 80), (0, 0),
        (80, -80), (0, 0), (-80, 80),
    ]


def gen_zigzag(n_segs: int = 5, amp: float = 55.0) -> Stroke:
    pts: Stroke = []
    for i in range(n_segs + 1):
        x = -100 + i * 200 / n_segs
        y = -amp if i % 2 == 0 else amp
        pts.append((x, y))
    return _densify(pts, per_seg=8)


def gen_wave(cycles: float = 2.0, amp: float = 50.0) -> Stroke:
    xs = np.linspace(-120, 120, 80)
    return [(float(x), amp * math.sin(x * math.pi * cycles / 120.0)) for x in xs]


def gen_spiral(turns: float = 2.0, cw: bool = True, phase: float = 0.0) -> Stroke:
    max_a = 2 * math.pi * turns
    angles = np.linspace(0, max_a, 100)
    sign = 1.0 if cw else -1.0
    return [
        ((a / max_a) * 100 * math.cos(sign * a + phase),
         (a / max_a) * 100 * math.sin(sign * a + phase))
        for a in angles
    ]


def gen_infinity() -> Stroke:
    """Figure-eight / lemniscate."""
    t = np.linspace(0, 2 * math.pi, 100)
    scale = 100
    xs = scale * np.cos(t) / (1 + np.sin(t) ** 2)
    ys = scale * np.sin(t) * np.cos(t) / (1 + np.sin(t) ** 2)
    return list(zip(xs.tolist(), ys.tolist()))


def gen_heart() -> Stroke:
    t = np.linspace(0, 2 * math.pi, 100)
    xs = 80 * (np.sin(t) ** 3)
    ys = -80 * (0.8125 * np.cos(t) - 0.3125 * np.cos(2*t)
                - 0.125 * np.cos(3*t) - 0.0625 * np.cos(4*t))
    return list(zip(xs.tolist(), ys.tolist()))


def gen_line_h() -> Stroke:
    return [(-100, 0), (100, 0)]


def gen_line_v() -> Stroke:
    return [(0, -100), (0, 100)]


def gen_v_shape() -> Stroke:
    return [(-80, -80), (0, 60), (80, -80)]


def gen_u_shape(width: float = 75.0, height: float = 80.0, boxy: bool = False) -> Stroke:
    if boxy:
        pts = [
            (-width, -height), (-width, height * 0.5), (-width * 0.4, height),
            (width * 0.4, height), (width, height * 0.5), (width, -height),
        ]
        return _densify(pts, per_seg=8)
    pts = [(-width, -height)]
    for a in np.linspace(math.pi, 0, 40):
        pts.append((-width * math.cos(a), height * 0.25 + width * 0.75 * math.sin(a)))
    pts.append((width, -height))
    return _densify(pts, per_seg=2)


def gen_l_shape() -> Stroke:
    return [(-60, -100), (-60, 80), (80, 80)]


def gen_s_shape(x_amp: float = 55.0) -> Stroke:
    # Realistic hand-drawn letter S: top-right -> top-left -> middle -> bottom-right -> bottom-left
    pts = [
        (x_amp * 0.8, -75), (0.0, -95), (-x_amp * 0.9, -60),
        (-x_amp * 0.5, -15), (x_amp * 0.5, 15), (x_amp * 0.9, 60),
        (0.0, 95), (-x_amp * 0.8, 75),
    ]
    return _densify(pts, per_seg=8)


def gen_pi() -> Stroke:
    return [(-90, -80), (90, -80), (90, -80),
            (40, -80), (40, 80), (40, -80),
            (-20, -80), (-20, 80)]


def gen_rectangle() -> Stroke:
    return _densify([(-120, -65), (120, -65), (120, 65), (-120, 65), (-120, -65)])


def gen_parallelogram() -> Stroke:
    return _densify([(-70, -70), (110, -70), (70, 70), (-110, 70), (-70, -70)])


def gen_trapezoid() -> Stroke:
    return _densify([(-60, -70), (60, -70), (110, 70), (-110, 70), (-60, -70)])


def gen_octagon() -> Stroke:
    return _poly(8)


def gen_arrow_left() -> Stroke:
    return _densify([
        (100, 0), (-100, 0), (-45, -45), (-100, 0), (-45, 45),
    ], per_seg=8)


def gen_arrow_down() -> Stroke:
    return _densify([
        (0, -100), (0, 100), (-45, 45), (0, 100), (45, 45),
    ], per_seg=8)


def gen_crescent() -> Stroke:
    outer = _arc(-math.pi * 0.65, math.pi * 0.65, r=100, steps=45)
    inner = [(x * 0.65 + 32, y * 0.65) for x, y in reversed(outer)]
    return outer + inner + [outer[0]]


def gen_cloud() -> Stroke:
    pts: Stroke = []
    bumps = [
        (-55, 20, 40, math.pi * 0.5, math.pi * 1.65),
        (-15, -20, 45, math.pi * 1.1, math.pi * 1.95),
        (40, -10, 38, math.pi * 1.25, math.pi * 2.2),
        (60, 25, 32, math.pi * 1.7, math.pi * 2.5),
    ]
    for cx, cy, r, a0, a1 in bumps:
        for a in np.linspace(a0, a1, 18):
            pts.append((cx + r * math.cos(a), cy + r * math.sin(a)))
    pts.append((-55, 60))
    pts.append(pts[0])
    return pts


def gen_cloud_variant(idx: int) -> Stroke:
    """Generate a closed cloud outline with a varied number of soft bumps."""
    bump_count = 3 + (idx % 4)
    width = random.uniform(150.0, 210.0)
    baseline = random.uniform(25.0, 45.0)
    step = width / bump_count
    pts: Stroke = []

    for i in range(bump_count):
        radius = step * random.uniform(0.48, 0.68)
        cx = -width / 2 + step * (i + 0.5)
        cy = baseline
        angles = np.linspace(math.pi, 0.0, 18)
        pts.extend((cx + radius * math.cos(a), cy - radius * math.sin(a)) for a in angles)

    pts.extend([(width / 2, baseline), (width / 2, baseline + 28),
                (-width / 2, baseline + 28), pts[0]])
    return pts


def gen_house() -> Stroke:
    return _densify([
        (-80, 0), (0, -90), (80, 0), (80, 90), (-80, 90), (-80, 0), (80, 0),
    ])


def gen_tree() -> Stroke:
    return _densify([
        (0, -100), (65, -20), (25, -20), (85, 50),
        (15, 50), (15, 100), (-15, 100), (-15, 50),
        (-85, 50), (-25, -20), (-65, -20), (0, -100),
    ])


def gen_lightning() -> Stroke:
    return _densify([(20, -100), (-50, 0), (10, 0), (-30, 100)])


def gen_music_note() -> Stroke:
    head = [(x - 35, y + 60) for x, y in _arc(0, 2 * math.pi, r=32, steps=36)]
    stem = _densify([(-3, 60), (-3, -90), (55, -65), (55, -35), (-3, -60)])
    return head + stem


def gen_hourglass() -> Stroke:
    return _densify([
        (-80, -90), (80, -90), (-80, 90), (80, 90), (-80, -90),
    ])


def gen_envelope() -> Stroke:
    return _densify([
        (-100, -65), (100, -65), (100, 65), (-100, 65),
        (-100, -65), (0, 15), (100, -65),
    ])


def gen_crown() -> Stroke:
    return _densify([
        (-95, 65), (-95, -45), (-45, 15), (0, -75),
        (45, 15), (95, -45), (95, 65), (-95, 65),
    ])


def gen_crown_variant(idx: int) -> Stroke:
    """Generate a crown with varied peak count, height, and valley depth."""
    peak_count = 3 + (idx % 4)
    width = random.uniform(160.0, 210.0)
    base_y = random.uniform(48.0, 70.0)
    peak_xs = np.linspace(-width / 2, width / 2, peak_count)
    pts: Stroke = [(-width / 2, base_y), (float(peak_xs[0]), -random.uniform(35.0, 90.0))]

    for left_x, right_x in zip(peak_xs, peak_xs[1:]):
        valley_x = float((left_x + right_x) / 2)
        pts.append((valley_x, base_y - random.uniform(8.0, 34.0)))
        pts.append((float(right_x), -random.uniform(35.0, 90.0)))

    pts.extend([(width / 2, base_y), (-width / 2, base_y)])
    return _densify(pts, per_seg=8)


def gen_butterfly() -> Stroke:
    t = np.linspace(0, 2 * math.pi, 140)
    r = np.exp(np.cos(t)) - 2 * np.cos(4 * t) + np.sin(t / 12) ** 5
    xs = 28 * np.sin(t) * r
    ys = -28 * np.cos(t) * r
    return list(zip(xs.tolist(), ys.tolist()))


def gen_butterfly_strokes(idx: int = 0) -> list[Stroke]:
    """Generate a multi-stroke butterfly with varied wing proportions/details."""
    style = idx % 4
    upper_rx = random.uniform(42.0, 58.0)
    upper_ry = random.uniform(34.0, 49.0)
    lower_rx = random.uniform(30.0, 45.0)
    lower_ry = random.uniform(32.0, 48.0)
    upper_y = random.uniform(-34.0, -22.0)
    lower_y = random.uniform(22.0, 36.0)
    lobe = (0.08, 0.18, 0.28, 0.12)[style]

    def wing(cx: float, cy: float, rx: float, ry: float, mirror: float) -> Stroke:
        result: Stroke = []
        for angle in np.linspace(0, 2 * math.pi, 34):
            contour = 1.0 + lobe * math.cos(2 * angle) + 0.06 * math.sin(3 * angle)
            result.append((cx + mirror * rx * math.cos(angle) * contour,
                           cy + ry * math.sin(angle) * contour))
        result.append(result[0])
        return result

    strokes = [
        wing(-22.0, upper_y, upper_rx, upper_ry, 1.0),
        wing(22.0, upper_y, upper_rx, upper_ry, -1.0),
        wing(-18.0, lower_y, lower_rx, lower_ry, 1.0),
        wing(18.0, lower_y, lower_rx, lower_ry, -1.0),
        [(0.0, -48.0), (random.uniform(-3, 3), 0.0), (0.0, 66.0)],
    ]
    if idx % 3 != 0:
        strokes.extend([
            [(0.0, upper_y + 3), (-upper_rx * 0.72, upper_y - upper_ry * 0.48)],
            [(0.0, upper_y + 3), (upper_rx * 0.72, upper_y - upper_ry * 0.48)],
            [(0.0, lower_y - 2), (-lower_rx * 0.72, lower_y + lower_ry * 0.48)],
            [(0.0, lower_y - 2), (lower_rx * 0.72, lower_y + lower_ry * 0.48)],
        ])
    if idx % 4 != 1:
        strokes.extend([
            _densify([(0.0, -45.0), (-12.0, -72.0), (-25.0, -80.0)], per_seg=5),
            _densify([(0.0, -45.0), (12.0, -72.0), (25.0, -80.0)], per_seg=5),
        ])

    angle = random.uniform(-0.12, 0.12)
    fx, fy = random.uniform(0.88, 1.12), random.uniform(0.88, 1.12)
    cosine, sine = math.cos(angle), math.sin(angle)
    transformed: list[Stroke] = []
    for stroke in strokes:
        pts = []
        for x, y in stroke:
            x, y = x * fx, y * fy
            pts.append((cosine * x - sine * y + random.gauss(0, 2.0),
                        sine * x + cosine * y + random.gauss(0, 2.0)))
        if idx % 2:
            pts.reverse()
        transformed.append(pts)
    return transformed


def butterfly_vector(idx: int) -> np.ndarray:
    """Create one normalized, lightly noised multi-stroke butterfly sample."""
    from preprocessing import normalize_strokes

    strokes = gen_butterfly_strokes(idx)
    vector = normalize_strokes(strokes, N_POINTS)
    return vector + np.random.normal(0, NOISE_STD, vector.shape)


def gen_flower() -> Stroke:
    t = np.linspace(0, 2 * math.pi, 140)
    r = 65 + 35 * np.cos(5 * t)
    xs = r * np.cos(t)
    ys = r * np.sin(t)
    return list(zip(xs.tolist(), ys.tolist()))


def gen_cube() -> Stroke:
    return _densify([
        (-70, -30), (30, -30), (30, 70), (-70, 70), (-70, -30),
        (-30, -70), (70, -70), (70, 30), (30, 70),
        (70, 30), (70, -70), (30, -30),
    ])


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

SHAPES: dict[str, Callable[[], Stroke]] = {
    "circle":        gen_circle,
    "ellipse":       gen_ellipse,
    "square":        gen_square,
    "rectangle":     gen_rectangle,
    "parallelogram": gen_parallelogram,
    "trapezoid":     gen_trapezoid,
    "triangle":      gen_triangle,
    "pentagon":      gen_pentagon,
    "hexagon":       gen_hexagon,
    "diamond":       gen_diamond,
    "star":          gen_star,
    "star6":         gen_star6,
    "arrow_right":   gen_arrow_right,
    "arrow_left":    gen_arrow_left,
    "arrow_up":      gen_arrow_up,
    "arrow_down":    gen_arrow_down,
    "check":         gen_check,
    "cross":         gen_cross,
    "x_shape":       gen_x_shape,
    "zigzag":        gen_zigzag,
    "wave":          gen_wave,
    "spiral":        gen_spiral,
    "infinity":      gen_infinity,
    "heart":         gen_heart,
    "crescent":      gen_crescent,
    "cloud":         gen_cloud,
    "house":         gen_house,
    "tree":          gen_tree,
    "lightning":     gen_lightning,
    "music_note":    gen_music_note,
    "hourglass":     gen_hourglass,
    "envelope":      gen_envelope,
    "crown":         gen_crown,
    "butterfly":     gen_butterfly,
    "flower":        gen_flower,
    "cube":          gen_cube,
    "line_h":        gen_line_h,
    "line_v":        gen_line_v,
    "v_shape":       gen_v_shape,
    "u_shape":       gen_u_shape,
    "l_shape":       gen_l_shape,
    "s_shape":       gen_s_shape,
}


# ---------------------------------------------------------------------------
# Augmentation helpers
# ---------------------------------------------------------------------------

def _rotate(pts: Stroke, angle_rad: float) -> Stroke:
    c, s = math.cos(angle_rad), math.sin(angle_rad)
    return [(c * x - s * y, s * x + c * y) for x, y in pts]


def _jitter(pts: Stroke, px: float = 4.0) -> Stroke:
    return [(x + random.gauss(0, px), y + random.gauss(0, px)) for x, y in pts]


def _scale_aug(pts: Stroke, fx: float, fy: float) -> Stroke:
    return [(x * fx, y * fy) for x, y in pts]


def _shift_closed_start(pts: Stroke, frac: float) -> Stroke:
    """Shift the starting point of a closed loop by *frac* ∈ [0, 1)."""
    if len(pts) < 3:
        return pts
    # Closed shapes repeat the first point at the end; strip and re-close.
    loop = pts[:-1] if pts[0] == pts[-1] else pts
    idx = int(frac * len(loop)) % len(loop)
    shifted = loop[idx:] + loop[:idx]
    return shifted + [shifted[0]]


def augment(pts: Stroke, label: str, idx: int, total: int) -> Stroke:
    """Apply direction, start-point, rotation, stretch and jitter."""
    if label == "star" and idx % 2 == 1:
        pts = gen_star(pentagram=True)
    elif label == "spiral":
        pts = gen_spiral(
            turns=random.uniform(1.4, 2.6),
            cw=(idx % 2 == 0),
            phase=random.uniform(0, 2 * math.pi),
        )
    elif label == "zigzag":
        pts = gen_zigzag(
            n_segs=random.choice([4, 5, 6, 7]),
            amp=random.choice([-1.0, 1.0]) * random.uniform(35.0, 75.0),
        )
    elif label == "wave":
        pts = gen_wave(
            cycles=random.choice([1.5, 2.0, 2.5]),
            amp=random.choice([-1.0, 1.0]) * random.uniform(35.0, 65.0),
        )
    elif label == "u_shape":
        pts = gen_u_shape(
            width=random.uniform(55.0, 100.0),
            height=random.uniform(55.0, 95.0),
            boxy=(idx % 2 == 1),
        )
    elif label == "s_shape":
        pts = gen_s_shape(x_amp=random.uniform(35.0, 75.0))
    elif label == "cloud":
        pts = gen_cloud_variant(idx)
    elif label == "crown":
        pts = gen_crown_variant(idx)

    if label in CLOSED_SHAPES:
        if idx % 2 == 1:
            pts = list(reversed(pts))
        start_slot = (idx // 2) % 8
        pts = _shift_closed_start(pts, start_slot / 8.0)
    elif label in BIDIRECTIONAL_SHAPES and idx % 2 == 1:
        pts = list(reversed(pts))

    # Mix mild and rougher ("uglier") human variations
    rough = idx >= total // 2
    rot_lim = 0.28 if rough else 0.16
    jit_px  = 6.5  if rough else 3.5
    s_lo, s_hi = (0.80, 1.20) if rough else (0.90, 1.10)

    pts = _rotate(pts, random.uniform(-rot_lim, rot_lim))
    pts = _scale_aug(pts, random.uniform(s_lo, s_hi), random.uniform(s_lo, s_hi))
    pts = _jitter(pts, px=jit_px)
    return pts


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    for label, gen_fn in SHAPES.items():
        for i in range(SAMPLES_PER_CLASS):
            if i == 0:
                # Keep sample 0 clean so it serves as a crisp UI prototype icon.
                if label == "butterfly":
                    from preprocessing import normalize_strokes
                    vec = normalize_strokes(gen_butterfly_strokes(i), N_POINTS)
                else:
                    vec = normalize(gen_fn(), N_POINTS)
            elif label == "butterfly":
                vec = butterfly_vector(i)
            else:
                raw = augment(gen_fn(), label, i, SAMPLES_PER_CLASS)
                vec = normalize(raw, N_POINTS)
                vec = vec + np.random.normal(0, NOISE_STD, vec.shape)
            records.append({"label": label, "vector": vec.tolist()})
        print(f"  ✓  {label:16s}  ({SAMPLES_PER_CLASS} samples)")

    with OUTPUT_PATH.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, indent=2, ensure_ascii=False)

    total = len(records)
    print(f"\nDataset saved → {OUTPUT_PATH}")
    print(f"Classes: {len(SHAPES)}   Total samples: {total}")


def self_train(
    dataset_path: str | Path = OUTPUT_PATH,
    rounds: int = 2,
    probes_per_class: int = 8,
) -> int:
    """Hard-example self-training pass.

    Generates rough hand-drawn probes for every shape, evaluates the current
    k-NN classifier, and appends misclassified or low-confidence (< 0.80)
    examples to *dataset_path*. Returns the number of newly learned samples.
    """
    from knn_classifier import KNNClassifier

    clf = KNNClassifier(k=5, dataset_path=dataset_path)
    added = 0
    for _ in range(rounds):
        new_samples: list[tuple[str, np.ndarray]] = []
        for label, gen_fn in SHAPES.items():
            for idx in range(probes_per_class):
                sample_idx = idx + SAMPLES_PER_CLASS // 2
                if label == "butterfly":
                    vec = butterfly_vector(sample_idx)
                else:
                    raw = augment(gen_fn(), label, sample_idx, SAMPLES_PER_CLASS)
                    vec = normalize(raw, N_POINTS) + np.random.normal(0, NOISE_STD, N_POINTS * 2)
                res = clf.predict(vec)
                if res is None or res[0] != label or res[1] < 0.80:
                    new_samples.append((label, vec))
        clf.add_samples(new_samples)
        added += len(new_samples)
        if not new_samples:
            break
    print(f"Self-training complete: +{added} hard examples (total: {clf.n_samples})")
    return added


if __name__ == "__main__":
    generate()
    self_train()
