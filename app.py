"""
app.py
~~~~~~
Modern Tkinter GUI for 2-D gesture recording and real-time recognition.
Supports multi-stroke drawing (continue drawing after releasing the mouse)
and renders crisp vector graphics for every shape instead of emojis.
"""

from __future__ import annotations

import json
import queue
import random
import threading
import tkinter as tk
from tkinter import messagebox
from pathlib import Path
from typing import Optional

import numpy as np

from preprocessing import normalize_strokes, normalize_strokes_to_array, N_POINTS
from knn_classifier import KNNClassifier

# ── Palette ────────────────────────────────────────────────────────────────
C = {
    "bg":         "#0F172A",
    "surface":    "#1E293B",
    "surface2":   "#273548",
    "border":     "#334155",
    "primary":    "#6366F1",
    "primary_dk": "#4F46E5",
    "success":    "#10B981",
    "warn":       "#F59E0B",
    "danger":     "#EF4444",
    "text":       "#F1F5F9",
    "text2":      "#94A3B8",
    "stroke":     "#818CF8",
    "dot_line":   "#475569",
    "axis":       "#1E3A5F",
}

# ── Sizes ──────────────────────────────────────────────────────────────────
DRAW_W, DRAW_H = 580, 500
PREV_W, PREV_H = 280, 500
SIDE_W         = 290
TOOLBAR_H      = 72

DATASET_PATH  = Path("data/dataset.json")
SCORES_PATH   = Path("data/scores.json")
K             = 5
ROUND_SECONDS = 30

BG_NAMES: dict[str, str] = {
    "circle":        "Кръг",
    "ellipse":       "Елипса",
    "square":        "Квадрат",
    "rectangle":     "Правоъгълник",
    "parallelogram": "Успоредник",
    "trapezoid":     "Трапец",
    "triangle":      "Триъгълник",
    "pentagon":      "Петоъгълник",
    "hexagon":       "Шестоъгълник",
    "diamond":       "Ромб",
    "star":          "Звезда",
    "star6":         "Шестолъчна звезда",
    "arrow_right":   "Стрелка надясно",
    "arrow_left":    "Стрелка наляво",
    "arrow_up":      "Стрелка нагоре",
    "arrow_down":    "Стрелка надолу",
    "check":         "Тикче",
    "cross":         "Кръст (+)",
    "x_shape":       "Хикс (X)",
    "zigzag":        "Зигзаг",
    "wave":          "Вълна",
    "spiral":        "Спирала",
    "infinity":      "Безкрайност",
    "heart":         "Сърце",
    "crescent":      "Полумесец",
    "cloud":         "Облак",
    "house":         "Къща",
    "tree":          "Дърво",
    "lightning":     "Светкавица",
    "music_note":    "Музикална нота",
    "hourglass":     "Пясъчен часовник",
    "envelope":      "Плик",
    "crown":         "Корона",
    "butterfly":     "Пеперуда",
    "flower":        "Цвете",
    "cube":          "Куб",
    "line_h":        "Хоризонтална линия",
    "line_v":        "Вертикална линия",
    "v_shape":       "V-форма",
    "u_shape":       "U-форма",
    "l_shape":       "L-форма",
    "s_shape":       "S-форма",
}

EASY_SHAPES: set[str] = {
    "circle", "square", "triangle", "diamond", "star", "heart",
    "house", "arrow_right", "arrow_left", "arrow_up", "arrow_down",
    "check", "cross", "x_shape", "line_h", "line_v",
}

DIFFICULTIES: dict[str, tuple[int, bool]] = {
    "Лесно":  (35, True),
    "Средно": (30, True),
    "Трудно": (18, False),
}


def _display_name(label: str) -> str:
    """Return Bulgarian display name for *label* if known."""
    return BG_NAMES.get(label, label)


# ── Helpers ────────────────────────────────────────────────────────────────

def _btn(parent: tk.Widget, text: str, color: str, hover: str,
         command=None, **kw) -> tk.Button:
    """Flat button with hover effect."""
    b = tk.Button(
        parent, text=text, bg=color, fg=C["text"],
        activebackground=hover, activeforeground=C["text"],
        relief="flat", bd=0, cursor="hand2",
        font=("Segoe UI", 9, "bold"),
        padx=12, pady=6, command=command, **kw,
    )
    b.bind("<Enter>", lambda _: b.config(bg=hover))
    b.bind("<Leave>", lambda _: b.config(bg=color))
    return b


def _draw_shape_icon(
    canvas: tk.Canvas,
    pts: Optional[np.ndarray],
    size: int = 28,
    color: str = C["stroke"],
) -> None:
    """Render a miniature vector graphic of a normalised (N, 2) shape."""
    canvas.delete("all")
    if pts is None or len(pts) < 2:
        return

    pad = 4
    half = (size - 2 * pad) / 2.0
    cx = cy = size / 2.0
    coords: list[float] = []
    for nx, ny in pts:
        coords.extend((cx + float(nx) * half, cy + float(ny) * half))

    if len(coords) >= 4:
        canvas.create_line(
            coords, fill=color, width=1.8,
            capstyle=tk.ROUND, joinstyle=tk.ROUND,
        )


# ── Main application ───────────────────────────────────────────────────────

class GestureApp(tk.Tk):
    """Main window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("2D Gesture Recogniser")
        self.resizable(False, False)
        self.configure(bg=C["bg"])

        self._clf = KNNClassifier(k=K, dataset_path=DATASET_PATH)
        self._strokes: list[list[tuple[float, float]]] = []
        self._current_stroke: list[tuple[float, float]] = []
        self._mode = tk.StringVar(value="predict")
        self._label_var = tk.StringVar(value="")
        self._search_var = tk.StringVar(value="")

        # Skribbl game state
        self._game_target: str = ""
        self._game_time: int = ROUND_SECONDS
        self._game_score: int = 0
        self._game_streak: int = 0
        self._high_score: int = self._load_high_score()
        self._difficulty = tk.StringVar(value="Средно")
        self._self_learn = tk.BooleanVar(value=True)
        self._last_top3: list[str] = []
        self._game_active: bool = False
        self._timer_id: Optional[str] = None
        self._next_round_id: Optional[str] = None
        self._hint_visible: bool = False
        self._self_train_running = False
        self._self_train_queue: queue.Queue[tuple[Optional[int], Optional[Exception]]] = queue.Queue()
        self._self_train_poll_id: Optional[str] = None

        self._build_ui()
        self._bind_shortcuts()
        self._refresh_class_list()
        self._set_mode("predict")

    def _bind_shortcuts(self) -> None:
        self.bind("<Control-z>", lambda _: self._undo())
        self.bind("<Control-Z>", lambda _: self._undo())
        self.bind("<Delete>",    lambda _: self._clear())
        self.bind("<Escape>",    lambda _: self._clear())

    def destroy(self) -> None:
        self._cancel_timers()
        if self._self_train_poll_id is not None:
            self.after_cancel(self._self_train_poll_id)
            self._self_train_poll_id = None
        super().destroy()

    # ── UI construction ────────────────────────────────────────────────────

    def _build_ui(self) -> None:
        # Skribbl HUD bar (shown only in skribbl mode)
        self._skribbl_bar = tk.Frame(
            self, bg=C["surface"], height=52,
            highlightthickness=1, highlightbackground=C["primary"],
        )
        self._skribbl_bar.pack_propagate(False)
        self._build_skribbl_bar(self._skribbl_bar)

        self._row0 = tk.Frame(self, bg=C["bg"])
        self._row0.pack(padx=8, pady=(8, 0))
        row0 = self._row0

        # Drawing canvas card
        draw_outer, draw_inner = self._make_card(
            row0, "  Нарисувай форма (може на няколко щриха · десен клик = изчисти)  "
        )
        draw_outer.pack(side="left", padx=(0, 6))
        self._canvas = tk.Canvas(
            draw_inner, width=DRAW_W, height=DRAW_H,
            bg=C["surface2"], highlightthickness=0, cursor="crosshair",
        )
        self._canvas.pack()
        self._canvas.bind("<ButtonPress-1>",   self._on_press)
        self._canvas.bind("<B1-Motion>",       self._on_drag)
        self._canvas.bind("<ButtonRelease-1>", self._on_release)
        self._canvas.bind("<Button-3>",        lambda _: self._clear())
        self._draw_grid()

        # Normalised preview card
        prev_outer, prev_inner = self._make_card(
            row0, "  Нормализирани 32 точки  "
        )
        prev_outer.pack(side="left", padx=(0, 6))
        self._prev = tk.Canvas(
            prev_inner, width=PREV_W, height=PREV_H,
            bg=C["surface"], highlightthickness=0,
        )
        self._prev.pack()

        # Side panel
        side = tk.Frame(
            row0, bg=C["surface"], width=SIDE_W,
            highlightthickness=1, highlightbackground=C["border"],
        )
        side.pack(side="left", fill="y")
        side.pack_propagate(False)
        self._build_side(side)

        # Toolbar
        toolbar = tk.Frame(
            self, bg=C["surface"], height=TOOLBAR_H,
            highlightthickness=1, highlightbackground=C["border"],
        )
        toolbar.pack(fill="x", padx=8, pady=8)
        toolbar.pack_propagate(False)
        self._build_toolbar(toolbar)

    def _make_card(self, parent: tk.Widget,
                   title: str) -> tuple[tk.Frame, tk.Frame]:
        """Return (outer, inner) card frames."""
        outer = tk.Frame(parent, bg=C["border"], bd=1)
        hdr = tk.Frame(outer, bg=C["surface"], height=24)
        hdr.pack(fill="x")
        tk.Label(hdr, text=title, bg=C["surface"], fg=C["text2"],
                 font=("Segoe UI", 8)).pack(side="left", padx=8)
        inner = tk.Frame(outer, bg=C["surface2"])
        inner.pack(fill="both", expand=True)
        return outer, inner

    def _build_side(self, side: tk.Frame) -> None:
        # ── Results section ───────────────────────────────────────────
        tk.Label(side, text="  Резултат (Топ 3)", bg=C["surface"],
                 fg=C["text2"], font=("Segoe UI", 8),
                 anchor="w").pack(fill="x", pady=(10, 4), padx=8)

        self._result_frame = tk.Frame(side, bg=C["surface"])
        self._result_frame.pack(fill="x", padx=8)

        self._result_rows: list[dict] = []
        rank_colors = [C["success"], C["warn"], C["text2"]]

        for i in range(3):
            row = tk.Frame(self._result_frame, bg=C["surface"])
            row.pack(fill="x", pady=3)

            tk.Label(row, text=f"#{i + 1}", width=3, bg=C["surface"],
                     fg=rank_colors[i],
                     font=("Segoe UI", 8, "bold")).pack(side="left")

            icon_cv = tk.Canvas(
                row, width=28, height=28, bg=C["surface2"],
                highlightthickness=1, highlightbackground=C["border"],
            )
            icon_cv.pack(side="left", padx=(2, 6))

            name_lbl = tk.Label(
                row, text="—", anchor="w", bg=C["surface"], fg=C["text"],
                font=("Segoe UI", 10, "bold" if i == 0 else "normal"),
            )
            name_lbl.pack(side="left")

            conf_lbl = tk.Label(row, text="", width=5, bg=C["surface"],
                                fg=rank_colors[i], font=("Segoe UI", 8))
            conf_lbl.pack(side="right")

            bar_bg = tk.Frame(row, bg=C["border"], height=5)
            bar_bg.pack(side="right", fill="x", expand=True, padx=(6, 0))
            bar_fg = tk.Frame(bar_bg, bg=rank_colors[i], height=5, width=0)
            bar_fg.place(x=0, y=0, relheight=1)

            self._result_rows.append({
                "row": row, "icon": icon_cv, "name": name_lbl,
                "conf": conf_lbl, "bar_bg": bar_bg, "bar_fg": bar_fg,
                "color": rank_colors[i],
            })
            for widget in (row, icon_cv, name_lbl):
                widget.config(cursor="hand2")
                widget.bind("<Button-1>", lambda _, idx=i: self._teach_from_result(idx))

        tk.Frame(side, bg=C["border"], height=1).pack(
            fill="x", padx=8, pady=(10, 8))

        # ── Class browser (scrollable with vector graphics) ───────────
        hdr2 = tk.Frame(side, bg=C["surface"])
        hdr2.pack(fill="x", padx=8)
        tk.Label(hdr2, text="  Класове в базата", bg=C["surface"],
                 fg=C["text2"], font=("Segoe UI", 8),
                 anchor="w").pack(side="left")
        self._count_lbl = tk.Label(hdr2, text="", bg=C["surface"],
                                   fg=C["primary"], font=("Segoe UI", 8, "bold"))
        self._count_lbl.pack(side="right")

        # Search filter
        search_f = tk.Frame(side, bg=C["surface"])
        search_f.pack(fill="x", padx=8, pady=(4, 0))
        tk.Label(search_f, text="Търси:", bg=C["surface"],
                 fg=C["text2"], font=("Segoe UI", 8)).pack(side="left", padx=(0, 4))
        search_entry = tk.Entry(
            search_f, textvariable=self._search_var,
            bg=C["surface2"], fg=C["text"], insertbackground=C["text"],
            font=("Segoe UI", 8), bd=0, highlightthickness=1,
            highlightcolor=C["primary"], highlightbackground=C["border"],
        )
        search_entry.pack(side="left", fill="x", expand=True)
        self._search_var.trace_add("write", lambda *_: self._refresh_class_list())

        list_outer = tk.Frame(side, bg=C["surface2"],
                              highlightthickness=1,
                              highlightbackground=C["border"])
        list_outer.pack(fill="both", expand=True, padx=8, pady=(6, 6))

        self._class_canvas = tk.Canvas(
            list_outer, bg=C["surface2"], highlightthickness=0,
        )
        sb = tk.Scrollbar(list_outer, orient="vertical",
                          command=self._class_canvas.yview, width=10)
        self._class_inner = tk.Frame(self._class_canvas, bg=C["surface2"])
        self._class_inner.bind(
            "<Configure>",
            lambda _: self._class_canvas.configure(
                scrollregion=self._class_canvas.bbox("all")
            ),
        )
        self._class_window = self._class_canvas.create_window(
            (0, 0), window=self._class_inner, anchor="nw",
        )
        self._class_canvas.bind(
            "<Configure>",
            lambda e: self._class_canvas.itemconfig(
                self._class_window, width=e.width
            ),
        )
        self._class_canvas.configure(yscrollcommand=sb.set)
        self._class_canvas.pack(side="left", fill="both", expand=True)
        sb.pack(side="right", fill="y")

        # Mouse wheel scrolling anywhere inside the class browser
        def _on_wheel(e: tk.Event) -> None:  # type: ignore[type-arg]
            delta = -1 if e.delta > 0 else 1
            self._class_canvas.yview_scroll(delta, "units")

        list_outer.bind(
            "<Enter>",
            lambda _: self._class_canvas.bind_all("<MouseWheel>", _on_wheel),
        )
        list_outer.bind(
            "<Leave>",
            lambda _: self._class_canvas.unbind_all("<MouseWheel>"),
        )

        tk.Label(side, text="Клик → избери име  |  ✕ → изтрий клас",
                 bg=C["surface"], fg=C["text2"],
                 font=("Segoe UI", 7)).pack(pady=(0, 6))

    def _build_skribbl_bar(self, bar: tk.Frame) -> None:
        tk.Label(bar, text="SKRIBBL:", bg=C["surface"], fg=C["primary"],
                 font=("Segoe UI", 10, "bold")).pack(side="left", padx=(12, 8))

        tk.Label(bar, text="Нарисувай:", bg=C["surface"], fg=C["text2"],
                 font=("Segoe UI", 10)).pack(side="left")

        self._target_lbl = tk.Label(
            bar, text="—", bg=C["surface"], fg=C["text"],
            font=("Segoe UI", 13, "bold"),
        )
        self._target_lbl.pack(side="left", padx=(6, 8))

        self._hint_icon = tk.Canvas(
            bar, width=32, height=32, bg=C["surface2"],
            highlightthickness=1, highlightbackground=C["border"],
        )
        self._hint_icon.pack(side="left", padx=(0, 12))

        self._skribbl_msg = tk.Label(
            bar, text="", bg=C["surface"], fg=C["success"],
            font=("Segoe UI", 10, "bold"),
        )
        self._skribbl_msg.pack(side="left", padx=8)

        # Right side controls + stats
        _btn(bar, "Нова игра", C["surface2"], C["border"],
             command=self._reset_skribbl).pack(side="right", padx=(4, 10), pady=8)
        _btn(bar, "Пропусни", C["surface2"], C["warn"],
             command=self._skip_skribbl_round).pack(side="right", padx=4, pady=8)
        self._hint_btn = _btn(bar, "Подсказка", C["surface2"], C["primary"],
                              command=self._toggle_hint)
        self._hint_btn.pack(side="right", padx=4, pady=8)

        diff_menu = tk.OptionMenu(
            bar, self._difficulty, *DIFFICULTIES.keys(),
            command=lambda _: self._reset_skribbl(),
        )
        diff_menu.config(
            bg=C["surface2"], fg=C["text"], activebackground=C["primary"],
            activeforeground=C["text"], highlightthickness=0, bd=0,
            font=("Segoe UI", 8, "bold"), cursor="hand2",
        )
        diff_menu["menu"].config(
            bg=C["surface2"], fg=C["text"],
            activebackground=C["primary"], activeforeground=C["text"],
        )
        diff_menu.pack(side="right", padx=4, pady=8)

        self._score_lbl = tk.Label(
            bar, text="Точки: 0  |  Серия: 0  |  Рекорд: 0", bg=C["surface"],
            fg=C["text"], font=("Segoe UI", 9, "bold"),
        )
        self._score_lbl.pack(side="right", padx=10)

        self._timer_lbl = tk.Label(
            bar, text=f"{ROUND_SECONDS}s", bg=C["surface2"],
            fg=C["success"], font=("Segoe UI", 11, "bold"),
            width=5, pady=3,
        )
        self._timer_lbl.pack(side="right", padx=(8, 4))

    def _build_toolbar(self, tb: tk.Frame) -> None:
        mode_f = tk.Frame(tb, bg=C["surface"])
        mode_f.pack(side="left", padx=(10, 6), pady=12)
        self._mode_btns: dict[str, tk.Button] = {}
        for text, value in [
            ("Разпознаване", "predict"),
            ("Skribbl Игра", "skribbl"),
            ("Запис", "record"),
        ]:
            b = tk.Button(
                mode_f, text=text, bg=C["surface"], fg=C["text2"],
                activebackground=C["primary"], activeforeground=C["text"],
                relief="flat", bd=0, cursor="hand2",
                font=("Segoe UI", 9, "bold"), padx=12, pady=5,
                command=lambda v=value: self._set_mode(v),
            )
            b.pack(side="left", padx=2)
            self._mode_btns[value] = b

        tk.Frame(tb, bg=C["border"], width=1).pack(
            side="left", fill="y", pady=10, padx=4)

        self._rec_frame = tk.Frame(tb, bg=C["surface"])
        self._rec_frame.pack(side="left", padx=6)
        tk.Label(self._rec_frame, text="Форма:", bg=C["surface"],
                 fg=C["text2"], font=("Segoe UI", 9)).pack(side="left", padx=(0, 4))

        self._entry = tk.Entry(
            self._rec_frame, textvariable=self._label_var,
            bg=C["surface2"], fg=C["text"], insertbackground=C["text"],
            font=("Segoe UI", 10), width=13, bd=0,
            highlightthickness=1, highlightcolor=C["primary"],
            highlightbackground=C["border"],
        )
        self._entry.pack(side="left")
        self._entry.bind("<Return>", lambda _: self._save_template())

        self._save_btn = _btn(
            self._rec_frame, "Запази (Enter)",
            C["primary"], C["primary_dk"], command=self._save_template,
        )
        self._save_btn.pack(side="left", padx=(8, 0))

        tk.Frame(tb, bg=C["border"], width=1).pack(
            side="left", fill="y", pady=10, padx=4)

        _btn(tb, "Undo щрих", C["surface2"], C["border"],
             command=self._undo).pack(side="left", padx=3, pady=12)
        _btn(tb, "Изчисти", C["surface2"], C["border"],
             command=self._clear).pack(side="left", padx=3, pady=12)

        tk.Frame(tb, bg=C["border"], width=1).pack(
            side="left", fill="y", pady=10, padx=4)

        self._regen_btn = _btn(tb, "Регенерирай", C["surface2"], C["warn"],
                               command=self._regen_dataset)
        self._regen_btn.pack(side="left", padx=3, pady=12)
        self._self_train_btn = _btn(tb, "Самообучи ИИ", C["surface2"], C["success"],
                                    command=self._run_self_train)
        self._self_train_btn.pack(side="left", padx=3, pady=12)

        tk.Checkbutton(
            tb, text="Авто-обучение", variable=self._self_learn,
            bg=C["surface"], fg=C["text2"], selectcolor=C["surface2"],
            activebackground=C["surface"], activeforeground=C["text"],
            font=("Segoe UI", 8), cursor="hand2",
        ).pack(side="left", padx=(4, 2))

        self._status = tk.Label(
            tb, text="", bg=C["surface"], fg=C["text2"],
            font=("Segoe UI", 8), anchor="e",
        )
        self._status.pack(side="right", padx=12)

    # ── Drawing (multi-stroke) ─────────────────────────────────────────────

    def _draw_grid(self) -> None:
        step = 30
        for x in range(0, DRAW_W + 1, step):
            for y in range(0, DRAW_H + 1, step):
                self._canvas.create_oval(
                    x - 1, y - 1, x + 1, y + 1,
                    fill=C["border"], outline="", tags="grid",
                )

    def _on_press(self, e: tk.Event) -> None:  # type: ignore[type-arg]
        """Start a new stroke WITHOUT erasing previous strokes."""
        self._current_stroke = [(float(e.x), float(e.y))]

    def _on_drag(self, e: tk.Event) -> None:  # type: ignore[type-arg]
        x, y = float(e.x), float(e.y)
        if self._current_stroke:
            px, py = self._current_stroke[-1]
            self._canvas.create_line(
                px, py, x, y,
                fill=C["stroke"], width=3,
                capstyle=tk.ROUND, joinstyle=tk.ROUND,
                tags="stroke",
            )
        self._current_stroke.append((x, y))

    def _on_release(self, e: tk.Event) -> None:  # type: ignore[type-arg]
        """Finish current stroke, keep it on canvas, and predict live."""
        if len(self._current_stroke) < 2:
            self._current_stroke = []
            return
        self._strokes.append(self._current_stroke)
        self._current_stroke = []
        self._canvas.create_oval(
            e.x - 3, e.y - 3, e.x + 3, e.y + 3,
            fill=C["primary"], outline="", tags="stroke",
        )
        self._draw_preview()
        if self._mode.get() in ("predict", "skribbl"):
            self._predict()

    def _redraw_all_strokes(self) -> None:
        self._canvas.delete("stroke")
        for stroke in self._strokes:
            for i in range(1, len(stroke)):
                px, py = stroke[i - 1]
                x, y = stroke[i]
                self._canvas.create_line(
                    px, py, x, y,
                    fill=C["stroke"], width=3,
                    capstyle=tk.ROUND, joinstyle=tk.ROUND,
                    tags="stroke",
                )
            if stroke:
                ex, ey = stroke[-1]
                self._canvas.create_oval(
                    ex - 3, ey - 3, ex + 3, ey + 3,
                    fill=C["primary"], outline="", tags="stroke",
                )

    def _undo(self) -> None:
        if not self._strokes:
            self._status.config(text="Няма щрих за връщане.")
            return
        self._strokes.pop()
        self._redraw_all_strokes()
        self._prev.delete("all")
        self._clear_results()
        if self._strokes:
            self._draw_preview()
            if self._mode.get() in ("predict", "skribbl"):
                self._predict()
        else:
            self._status.config(text="Платното е изчистено.")

    def _clear(self) -> None:
        self._strokes.clear()
        self._current_stroke = []
        self._canvas.delete("stroke")
        self._prev.delete("all")
        self._clear_results()
        self._status.config(text="Платното е изчистено.")

    # ── Mode ───────────────────────────────────────────────────────────────

    def _set_mode(self, mode: str) -> None:
        self._mode.set(mode)
        is_record = (mode == "record")
        is_skribbl = (mode == "skribbl")

        for m, btn in self._mode_btns.items():
            btn.config(
                bg=C["primary"] if m == mode else C["surface"],
                fg=C["text"]    if m == mode else C["text2"],
            )
        state = "normal" if is_record else "disabled"
        self._entry.config(state=state)
        self._save_btn.config(state=state)
        self._clear_results()

        if is_skribbl:
            self._skribbl_bar.pack(fill="x", padx=8, pady=(8, 0), before=self._row0)
            self._reset_skribbl()
        else:
            self._cancel_timers()
            self._game_active = False
            self._skribbl_bar.pack_forget()
            if self._strokes and not is_record:
                self._predict()
            self._status.config(
                text=("Запис — нарисувай (може на няколко щриха) и запази."
                      if is_record else
                      "Разпознаване — нарисувай форма (може на няколко щриха).")
            )

    # ── Skribbl game logic ─────────────────────────────────────────────────

    def _cancel_timers(self) -> None:
        if self._timer_id is not None:
            self.after_cancel(self._timer_id)
            self._timer_id = None
        if self._next_round_id is not None:
            self.after_cancel(self._next_round_id)
            self._next_round_id = None

    def _load_high_score(self) -> int:
        if not SCORES_PATH.exists():
            return 0
        try:
            data = json.loads(SCORES_PATH.read_text(encoding="utf-8"))
            return int(data.get("high_score", 0))
        except (OSError, json.JSONDecodeError, ValueError):
            return 0

    def _save_high_score(self) -> None:
        SCORES_PATH.parent.mkdir(parents=True, exist_ok=True)
        SCORES_PATH.write_text(
            json.dumps({"high_score": self._high_score}, indent=2),
            encoding="utf-8",
        )

    def _reset_skribbl(self) -> None:
        self._game_score = 0
        self._game_streak = 0
        self._start_skribbl_round()

    def _start_skribbl_round(self) -> None:
        self._cancel_timers()
        labels = self._clf.labels
        if not labels:
            self._game_active = False
            self._target_lbl.config(text="Няма форми")
            return

        diff_name = self._difficulty.get()
        round_sec, allow_hint = DIFFICULTIES.get(diff_name, (ROUND_SECONDS, True))
        if diff_name == "Лесно":
            pool = [l for l in labels if l in EASY_SHAPES] or labels
        else:
            pool = labels

        choices = [l for l in pool if l != self._game_target] or pool
        self._game_target = random.choice(choices)
        self._game_time = round_sec
        self._game_active = True
        self._hint_visible = False
        self._hint_btn.config(state="normal" if allow_hint else "disabled")

        self._clear()
        self._target_lbl.config(text=_display_name(self._game_target), fg=C["text"])
        self._hint_icon.delete("all")
        self._skribbl_msg.config(text="Рисувай!", fg=C["text2"])
        self._update_skribbl_hud()
        self._tick_timer()

    def _skip_skribbl_round(self) -> None:
        if self._mode.get() != "skribbl":
            return
        self._game_streak = 0
        self._start_skribbl_round()

    def _toggle_hint(self) -> None:
        if not self._game_target:
            return
        self._hint_visible = not self._hint_visible
        if self._hint_visible:
            _draw_shape_icon(
                self._hint_icon,
                self._clf.get_prototype(self._game_target),
                size=32, color=C["warn"],
            )
        else:
            self._hint_icon.delete("all")

    def _tick_timer(self) -> None:
        if not self._game_active or self._mode.get() != "skribbl":
            return
        self._update_skribbl_hud()
        if self._game_time <= 0:
            self._game_active = False
            self._game_streak = 0
            self._skribbl_msg.config(
                text=f"Времето изтече! Беше: {_display_name(self._game_target)}",
                fg=C["danger"],
            )
            _draw_shape_icon(
                self._hint_icon,
                self._clf.get_prototype(self._game_target),
                size=32, color=C["danger"],
            )
            self._update_skribbl_hud()
            self._next_round_id = self.after(1800, self._start_skribbl_round)
            return
        self._game_time -= 1
        self._timer_id = self.after(1000, self._tick_timer)

    def _update_skribbl_hud(self) -> None:
        t_col = (C["success"] if self._game_time > 10
                 else C["warn"] if self._game_time > 5
                 else C["danger"])
        self._timer_lbl.config(text=f"{self._game_time}s", fg=t_col)
        self._score_lbl.config(
            text=f"Точки: {self._game_score}  |  Серия: {self._game_streak}  |  Рекорд: {self._high_score}"
        )

    def _on_skribbl_win(self) -> None:
        self._cancel_timers()
        self._game_active = False
        bonus = 100 + self._game_time * 5
        self._game_score += bonus
        self._game_streak += 1
        if self._game_score > self._high_score:
            self._high_score = self._game_score
            self._save_high_score()

        learned_note = ""
        if self._self_learn.get() and self._strokes:
            vec = normalize_strokes(self._strokes, N_POINTS)
            self._clf.add_sample(self._game_target, vec)
            self._refresh_class_list()
            learned_note = " · +1 самообучено"

        self._skribbl_msg.config(
            text=f"Браво! Познах «{_display_name(self._game_target)}» (+{bonus} т.{learned_note})",
            fg=C["success"],
        )
        _draw_shape_icon(
            self._hint_icon,
            self._clf.get_prototype(self._game_target),
            size=32, color=C["success"],
        )
        self._update_skribbl_hud()
        self._next_round_id = self.after(1500, self._start_skribbl_round)

    # ── Core actions ───────────────────────────────────────────────────────

    def _save_template(self) -> None:
        if self._self_train_running:
            return
        label = self._label_var.get().strip().lower().replace(" ", "_")
        if not label:
            messagebox.showwarning("Липсва етикет",
                                   "Въведи името на формата преди запис.")
            return
        if not self._strokes:
            messagebox.showwarning("Липсва жест",
                                   "Нарисувай форма преди запис.")
            return
        vec = normalize_strokes(self._strokes, N_POINTS)
        self._clf.add_sample(label, vec)
        self._refresh_class_list()
        n = self._clf.class_counts().get(label, 0)
        self._status.config(
            text=f"«{label}» записан.  Примери за клас: {n}"
        )
        self._clear()

    def _predict(self) -> None:
        if not self._strokes:
            return
        vec = normalize_strokes(self._strokes, N_POINTS)
        top3 = self._clf.predict_top_k(vec, top_n=3)
        self._last_top3 = [lbl for lbl, _ in top3]
        if not top3:
            self._status.config(text="Базата е празна — регенерирай датасет.")
            return

        for i, row in enumerate(self._result_rows):
            if i < len(top3):
                lbl, conf = top3[i]
                pct = int(conf * 100)
                proto = self._clf.get_prototype(lbl)
                _draw_shape_icon(row["icon"], proto, size=28, color=row["color"])
                row["name"].config(text=_display_name(lbl), fg=C["text"])
                row["conf"].config(text=f"{pct}%")
                row["bar_bg"].update_idletasks()
                bar_w = max(row["bar_bg"].winfo_width(), 1)
                row["bar_fg"].place(x=0, y=0,
                                    width=max(2, int(bar_w * conf)),
                                    relheight=1)
            else:
                row["icon"].delete("all")
                row["name"].config(text="—", fg=C["text"])
                row["conf"].config(text="")
                row["bar_fg"].place(width=0)

        best, best_conf = top3[0]
        conf_pct = int(best_conf * 100)
        color = (C["success"] if conf_pct >= 60
                 else C["warn"]    if conf_pct >= 40
                 else C["danger"])
        self._result_rows[0]["name"].config(fg=color)
        self._status.config(
            text=f"→ {_display_name(best)} ({conf_pct}% | щрихи: {len(self._strokes)} | "
                 f"база: {self._clf.n_samples})"
        )
        if self._mode.get() == "skribbl" and self._game_active:
            if best == self._game_target:
                self._on_skribbl_win()
            else:
                self._skribbl_msg.config(
                    text=f"AI мисли: «{_display_name(best)}» ({conf_pct}%)...",
                    fg=C["warn"],
                )

    def _clear_results(self) -> None:
        self._last_top3 = []
        for row in self._result_rows:
            row["icon"].delete("all")
            row["name"].config(text="—", fg=C["text"])
            row["conf"].config(text="")
            row["bar_fg"].place(width=0)

    def _teach_from_result(self, idx: int) -> None:
        """Teach the classifier by clicking one of the Top-3 result rows."""
        if self._self_train_running or self._mode.get() != "predict" or not self._strokes:
            return
        if idx >= len(self._last_top3):
            return
        label = self._last_top3[idx]
        vec = normalize_strokes(self._strokes, N_POINTS)
        self._clf.add_sample(label, vec)
        self._refresh_class_list()
        self._predict()
        self._status.config(
            text=f"Самообучено като «{_display_name(label)}» (общо база: {self._clf.n_samples})"
        )

    def _run_self_train(self) -> None:
        """Run one bounded self-training pass without blocking the GUI."""
        if self._self_train_running:
            return

        import generate_dataset as gd
        self._self_train_running = True
        self._self_train_btn.config(state="disabled")
        self._regen_btn.config(state="disabled")
        self._save_btn.config(state="disabled")
        self._status.config(text="Самообучение в ход...")
        self.update_idletasks()

        def train() -> None:
            try:
                added = gd.self_train(dataset_path=DATASET_PATH, rounds=1, probes_per_class=4)
                self._self_train_queue.put((added, None))
            except Exception as exc:
                self._self_train_queue.put((None, exc))

        threading.Thread(target=train, name="gesture-self-training", daemon=True).start()
        self._self_train_poll_id = self.after(100, self._finish_self_train)

    def _finish_self_train(self) -> None:
        self._self_train_poll_id = None
        try:
            added, error = self._self_train_queue.get_nowait()
        except queue.Empty:
            self._self_train_poll_id = self.after(100, self._finish_self_train)
            return

        self._self_train_running = False
        self._self_train_btn.config(state="normal")
        self._regen_btn.config(state="normal")
        self._save_btn.config(state="normal" if self._mode.get() == "record" else "disabled")
        if error is not None:
            self._status.config(text="Самообучението не завърши.")
            messagebox.showerror("Грешка при самообучение", str(error))
            return

        self._clf = KNNClassifier(k=K, dataset_path=DATASET_PATH)
        self._refresh_class_list()
        self._status.config(
            text=f"Самообучението завърши: +{added} примера (база: {self._clf.n_samples})"
        )

    def _regen_dataset(self) -> None:
        if self._self_train_running:
            return
        if not messagebox.askyesno(
            "Регенерирай датасет",
            "Ще се генерират синтетични примери за 25 форми.\n"
            "Ръчно записаните нови класове остават.\nПродължи?",
        ):
            return
        import generate_dataset as gd
        custom = [
            (lbl, self._clf.get_prototype(lbl))
            for lbl in self._clf.labels
            if lbl not in gd.SHAPES
        ]
        # Preserve all samples of custom classes
        custom_samples = [
            (s.label, s.vector)
            for s in getattr(self._clf, "_samples", [])
            if s.label not in gd.SHAPES
        ]
        gd.generate()
        self._clf = KNNClassifier(k=K, dataset_path=DATASET_PATH)
        for lbl, vec in custom_samples:
            self._clf.add_sample(lbl, vec)
        self._refresh_class_list()
        self._status.config(
            text=f"Датасетът е регенериран. Примери: {self._clf.n_samples}"
        )

    def _delete_class(self, label: str) -> None:
        if self._self_train_running:
            return
        if not messagebox.askyesno("Изтрий клас",
                                   f"Изтрий всички примери за «{label}»?"):
            return
        self._clf.remove_label(label)
        self._refresh_class_list()
        self._status.config(text=f"Клас «{label}» е изтрит.")

    def _select_class(self, label: str) -> None:
        self._label_var.set(label)
        self._status.config(text=f"Избран етикет: «{label}»")

    def _refresh_class_list(self) -> None:
        for child in self._class_inner.winfo_children():
            child.destroy()

        counts = self._clf.class_counts()
        query = self._search_var.get().strip().lower()
        for lbl, count in counts.items():
            disp = _display_name(lbl)
            if query and query not in lbl.lower() and query not in disp.lower():
                continue
            row = tk.Frame(self._class_inner, bg=C["surface2"], pady=2)
            row.pack(fill="x", padx=4, pady=1)

            icon = tk.Canvas(
                row, width=24, height=24, bg=C["surface"],
                highlightthickness=1, highlightbackground=C["border"],
            )
            icon.pack(side="left", padx=(2, 6))
            _draw_shape_icon(icon, self._clf.get_prototype(lbl),
                             size=24, color=C["stroke"])

            name_lbl = tk.Label(
                row, text=f"{disp} ({count})", anchor="w",
                bg=C["surface2"], fg=C["text"],
                font=("Segoe UI", 9), cursor="hand2",
            )
            name_lbl.pack(side="left", fill="x", expand=True)
            name_lbl.bind("<Button-1>", lambda _, l=lbl: self._select_class(l))
            icon.bind("<Button-1>", lambda _, l=lbl: self._select_class(l))

            del_btn = tk.Label(
                row, text="✕", bg=C["surface2"], fg=C["text2"],
                font=("Segoe UI", 8, "bold"), cursor="hand2", padx=4,
            )
            del_btn.pack(side="right")
            del_btn.bind("<Button-1>", lambda _, l=lbl: self._delete_class(l))
            del_btn.bind("<Enter>", lambda _, b=del_btn: b.config(fg=C["danger"]))
            del_btn.bind("<Leave>", lambda _, b=del_btn: b.config(fg=C["text2"]))

        self._count_lbl.config(
            text=f"{len(counts)} кл · {self._clf.n_samples} прим"
        )

    # ── Preview ────────────────────────────────────────────────────────────

    def _draw_preview(self) -> None:
        if not self._strokes:
            return
        pts_norm = normalize_strokes_to_array(self._strokes, N_POINTS)
        self._prev.delete("all")

        m = 28
        cx, cy = PREV_W / 2, PREV_H / 2
        sc = min((PREV_W - 2 * m) / 2, (PREV_H - 2 * m) / 2)

        def to_px(nx: float, ny: float) -> tuple[float, float]:
            return cx + nx * sc, cy + ny * sc

        self._prev.create_line(cx, m, cx, PREV_H - m,
                               fill=C["axis"], dash=(2, 6))
        self._prev.create_line(m, cy, PREV_W - m, cy,
                               fill=C["axis"], dash=(2, 6))

        bx1, by1 = to_px(-1, -1)
        bx2, by2 = to_px(1, 1)
        self._prev.create_rectangle(bx1, by1, bx2, by2,
                                    outline=C["border"], dash=(4, 4), width=1)

        pixel_pts = [to_px(x, y) for x, y in pts_norm]
        n = len(pixel_pts)
        for i, (ppx, ppy) in enumerate(pixel_pts):
            t   = i / max(n - 1, 1)
            r_c = int(0x81 + t * (0xF4 - 0x81))
            g_c = int(0x8C - t * (0x8C - 0x3F))
            b_c = int(0xF8 - t * (0xF8 - 0x5E))
            color = f"#{r_c:02X}{g_c:02X}{b_c:02X}"
            rr = 4 if i in (0, n - 1) else 2.5
            self._prev.create_oval(
                ppx - rr, ppy - rr, ppx + rr, ppy + rr,
                fill=color, outline="",
            )
            if i % 8 == 0:
                self._prev.create_text(
                    ppx + 8, ppy - 8, text=str(i),
                    fill=C["text2"], font=("Segoe UI", 7),
                )


# ── Entry point ────────────────────────────────────────────────────────────

def main() -> None:
    app = GestureApp()
    app.mainloop()


if __name__ == "__main__":
    main()
