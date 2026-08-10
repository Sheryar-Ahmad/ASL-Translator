import logging
import platform
import queue
import threading
import time
from pathlib import Path
from typing import Optional
import tkinter as tk
from tkinter import ttk, font as tkfont
import cv2
from PIL import Image, ImageDraw, ImageTk

from app.asl_engine import ASLEngine
from app.config import load_config
from app.text_buffer import AutoCommitController, TextBuffer
from app.tts_engine import TTSEngine

logger = logging.getLogger(__name__)

# ── Palette ────────────────────────────────────────────────────────────────
BG = "#0d1117"
CARD = "#161b22"
CARD2 = "#1c2333"
BORDER = "#30363d"
TEXT = "#e6edf3"
MUTED = "#8b949e"
GREEN = "#3fb950"
BLUE = "#58a6ff"
PURPLE = "#bc8cff"
GOLD = "#d29922"
RED = "#f85149"
TEAL = "#39d353"

BTN_START = "#1a7f37"
BTN_STOP_C = "#6e2020"
BTN_SPEAK = "#1a3a6e"
BTN_SPKSEN = "#3b1f6e"
BTN_CLR = "#5a3e00"
BTN_SPACE = "#1a4a6e"


class ASLGuiApp:
    def __init__(self, root: tk.Tk, cfg: dict):
        self.root = root
        self.cfg = cfg
        self.running = False

        self.asl = ASLEngine(cfg.get("asl", {}))

        tts_cfg = cfg.get("tts", {})
        self.tts = TTSEngine(tts_cfg)
        self.tts.start()
        self.tts_model_path = str(tts_cfg.get("model_path", ""))
        self.tts_available = bool(
            self.tts_model_path and Path(self.tts_model_path).exists()
        )

        labels_cfg = cfg.get("asl", {}).get("labels", {})
        self.text_buffer = TextBuffer(
            space_label=labels_cfg.get("space_label", "space"),
            delete_label=labels_cfg.get("delete_label", "delete"),
            nothing_label=labels_cfg.get("nothing_label", "nothing"),
        )

        sentence_cfg = cfg.get("sentence", {})
        self.auto_speak_var = tk.BooleanVar(
            value=bool(sentence_cfg.get("auto_speak", True))
        )
        self.speak_on_space_var = tk.BooleanVar(
            value=bool(sentence_cfg.get("speak_on_space", True))
        )
        self.tts_enabled_var = tk.BooleanVar(value=True)

        self.confidence_threshold = float(
            cfg.get("asl", {}).get("confidence_threshold", 0.65)
        )
        self.stable_frames = int(sentence_cfg.get("stable_frames", 15))
        self.cooldown_frames = int(sentence_cfg.get("cooldown_frames", 45))
        self.min_chars = int(sentence_cfg.get("min_chars", 3))
        self.complete_pause_ms = int(sentence_cfg.get("complete_pause_ms", 2500))
        self.max_sentence_length = int(sentence_cfg.get("max_sentence_length", 200))

        self.auto_commit = self._build_auto_commit()

        self.capture: Optional[cv2.VideoCapture] = None
        self.camera_index = int(cfg.get("camera", {}).get("index", 0))
        self._open_camera(self.camera_index)

        self.latest_frame = None
        self.latest_prediction = {
            "label": "nothing",
            "confidence": 0.0,
            "hand_present": False,
        }
        self._commits: queue.Queue = queue.Queue()
        self.fps = 0.0
        self._frame_count = 0
        self._fps_time = time.time()
        self.recognition_active = False
        self.last_commit_ms = 0
        self.last_spoken_text = None
        self.last_committed_label: Optional[str] = None
        self._was_speaking = False
        self.speech_status = "Idle"
        self._speech_end_time = 0.0

        self._build_ui()

        self.running = True
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        self._update_ui()

    # ── helpers ────────────────────────────────────────────────────────────

    def _build_auto_commit(self) -> AutoCommitController:
        ctrl_cfg = dict(self.cfg.get("asl", {}))
        ctrl_cfg["confidence_threshold"] = self.confidence_threshold
        auto = dict(ctrl_cfg.get("auto", {}))
        fps = float(self.cfg.get("camera", {}).get("fps", 30))
        auto["stable_ms"] = max(1, int(self.stable_frames * (1000.0 / fps)))
        auto["reset_ms"] = max(1, int(self.cooldown_frames * (1000.0 / fps)))
        ctrl_cfg["auto"] = auto
        return AutoCommitController(ctrl_cfg)

    def _open_camera(self, index: int):
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        if platform.system() == "Windows":
            self.capture = cv2.VideoCapture(int(index), cv2.CAP_DSHOW)
        else:
            self.capture = cv2.VideoCapture(int(index))
        cam_cfg = self.cfg.get("camera", {})
        if self.capture.isOpened():
            self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, int(cam_cfg.get("width", 640)))
            self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, int(cam_cfg.get("height", 480)))
            self.capture.set(cv2.CAP_PROP_FPS, int(cam_cfg.get("fps", 30)))

    @staticmethod
    def list_cameras(max_index: int = 5):
        cams = []
        for i in range(max_index):
            cap = cv2.VideoCapture(
                i,
                cv2.CAP_DSHOW
                if platform.system() == "Windows"
                else cv2.CAP_ANY,
            )
            if cap.isOpened():
                cams.append(str(i))
            cap.release()
        return cams or ["0"]

    # ── UI construction ────────────────────────────────────────────────────

    def _card(self, parent, **kwargs) -> tk.Frame:
        return tk.Frame(
            parent,
            bg=CARD,
            bd=0,
            highlightthickness=1,
            highlightbackground=BORDER,
            **kwargs,
        )

    def _label(
        self, parent, text="", color=TEXT, size=10, bold=False, **kw
    ) -> tk.Label:
        weight = "bold" if bold else "normal"
        return tk.Label(
            parent,
            text=text,
            bg=parent["bg"],
            fg=color,
            font=("Segoe UI", size, weight),
            **kw,
        )

    def _icon_btn(self, parent, text, bg, command, fg=TEXT, width=16):
        b = tk.Button(
            parent,
            text=text,
            bg=bg,
            fg=fg,
            activebackground=bg,
            activeforeground=fg,
            font=("Segoe UI", 9, "bold"),
            relief="flat",
            bd=0,
            cursor="hand2",
            width=width,
            command=command,
            pady=7,
        )
        b.bind("<Enter>", lambda e: b.config(bg=self._lighten(bg)))
        b.bind("<Leave>", lambda e: b.config(bg=bg))
        return b

    @staticmethod
    def _lighten(hex_color: str) -> str:
        h = hex_color.lstrip("#")
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = min(255, r + 30)
        g = min(255, g + 30)
        b = min(255, b + 30)
        return f"#{r:02x}{g:02x}{b:02x}"

    def _toggle_switch(self, parent, variable, command=None):
        """Simple toggle drawn on a Canvas."""
        c = tk.Canvas(
            parent,
            width=44,
            height=22,
            bg=parent["bg"],
            highlightthickness=0,
            cursor="hand2",
        )

        def _draw():
            c.delete("all")
            on = variable.get()
            track_color = BLUE if on else BORDER
            c.create_rounded_rect = _round_rect
            _round_rect(c, 0, 2, 44, 20, 10, fill=track_color, outline="")
            cx = 32 if on else 12
            c.create_oval(cx - 8, 3, cx + 8, 19, fill=TEXT, outline="")

        def _round_rect(canvas, x1, y1, x2, y2, r, **kw):
            pts = [
                x1 + r,
                y1,
                x2 - r,
                y1,
                x2,
                y1,
                x2,
                y1 + r,
                x2,
                y2 - r,
                x2,
                y2,
                x2 - r,
                y2,
                x1 + r,
                y2,
                x1,
                y2,
                x1,
                y2 - r,
                x1,
                y1 + r,
                x1,
                y1,
            ]
            return canvas.create_polygon(pts, smooth=True, **kw)

        def _click(e):
            variable.set(not variable.get())
            _draw()
            if command:
                command()

        c.bind("<Button-1>", _click)
        _draw()
        return c

    def _build_ui(self):
        self.root.title("ASL Translator")
        self.root.geometry("1380x860")
        self.root.minsize(1200, 780)
        self.root.configure(bg=BG)
        self.root.option_add("*Font", '"Segoe UI" 10')

        # ── Top header bar ────────────────────────────────────────────────
        hdr = tk.Frame(
            self.root,
            bg=CARD,
            height=56,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        hdr.pack(fill="x", side="top")
        hdr.pack_propagate(False)

        # Logo / title
        logo_frame = tk.Frame(hdr, bg=CARD)
        logo_frame.pack(side="left", padx=20)
        tk.Label(logo_frame, text="✋", bg=CARD, fg=BLUE, font=("Segoe UI", 18)).pack(
            side="left", pady=12
        )
        title_f = tk.Frame(logo_frame, bg=CARD)
        title_f.pack(side="left", padx=(8, 0))
        tk.Label(
            title_f, text="ASL", bg=CARD, fg=TEXT, font=("Segoe UI", 14, "bold")
        ).pack(anchor="w")
        tk.Label(
            title_f, text="TRANSLATOR", bg=CARD, fg=MUTED, font=("Segoe UI", 7, "bold")
        ).pack(anchor="w")

        # Right: time
        self._clock_var = tk.StringVar()
        tk.Label(
            hdr, textvariable=self._clock_var, bg=CARD, fg=MUTED, font=("Segoe UI", 10)
        ).pack(side="right", padx=20)
        self._tick_clock()

        # ── Body ──────────────────────────────────────────────────────────
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=14, pady=(10, 6))

        # Column weights
        body.columnconfigure(0, weight=7)  # Left column gets more space
        body.columnconfigure(1, weight=3)  # Right column
        body.rowconfigure(0, weight=1)  # Single row that fills all space

        # ── LEFT column ───────────────────────────────────────────────────
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        # CRITICAL: Give camera almost all the space
        left.rowconfigure(0, weight=20)  # Camera - gets 20/22 of the space
        left.rowconfigure(1, weight=1)  # Sentence - minimal
        left.rowconfigure(2, weight=1)  # Settings - minimal
        left.columnconfigure(0, weight=1)

        # Camera card - HUGE
        cam_card = self._card(left)
        cam_card.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        cam_card.rowconfigure(
            1, weight=1
        )  # Camera preview takes all available space in card
        cam_card.columnconfigure(0, weight=1)

        cam_hdr = tk.Frame(cam_card, bg=CARD)
        cam_hdr.grid(row=0, column=0, sticky="ew", padx=12, pady=(6, 0))
        self._label(cam_hdr, "Live Camera", color=BLUE, size=10, bold=True).pack(
            side="left"
        )
        self._fps_badge = self._label(cam_hdr, "FPS: —", color=MUTED, size=9)
        self._fps_badge.pack(side="right")
        self._hand_badge = tk.Label(
            cam_hdr, text="● Hand Detected", bg=CARD, fg=GREEN, font=("Segoe UI", 9)
        )
        self._hand_badge.pack(side="right", padx=(0, 12))

        self.cam_label = tk.Label(cam_card, bg="black")
        self.cam_label.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 0))

        cam_foot = tk.Frame(cam_card, bg=CARD)
        cam_foot.grid(row=2, column=0, sticky="ew", padx=12, pady=4)
        self._cam_index_lbl = self._label(
            cam_foot, f"📷  Camera: {self.camera_index}", color=MUTED, size=9
        )
        self._cam_index_lbl.pack(side="left")
        self._res_lbl = self._label(cam_foot, "640 × 480", color=MUTED, size=9)
        self._res_lbl.pack(side="right")

        # Sentence card - SMALL (just enough to see text)
        sent_card = self._card(left)
        sent_card.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        sent_card.rowconfigure(1, weight=1)
        sent_card.columnconfigure(0, weight=1)

        sent_hdr = tk.Frame(sent_card, bg=CARD)
        sent_hdr.grid(row=0, column=0, sticky="ew", padx=10, pady=(4, 2))
        self._label(sent_hdr, "💬 Sentence", color=TEAL, size=9, bold=True).pack(
            side="left"
        )
        btn_frame = tk.Frame(sent_hdr, bg=CARD)
        btn_frame.pack(side="right")
        tk.Button(
            btn_frame,
            text="🗑",
            bg=CARD2,
            fg=RED,
            font=("Segoe UI", 8),
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self.clear_sentence,
            padx=6,
            pady=2,
        ).pack(side="left", padx=2)
        tk.Button(
            btn_frame,
            text="📋",
            bg=CARD2,
            fg=MUTED,
            font=("Segoe UI", 8),
            relief="flat",
            bd=0,
            cursor="hand2",
            command=self._copy_sentence,
            padx=6,
            pady=2,
        ).pack(side="left")

        self.sent_text = tk.Text(
            sent_card,
            wrap="word",
            font=("Segoe UI", 10),
            bg=CARD2,
            fg=TEXT,
            relief="flat",
            bd=0,
            insertbackground=TEXT,
            padx=10,
            pady=4,
            highlightthickness=0,
            height=2,
        )
        self.sent_text.grid(row=1, column=0, sticky="nsew", padx=8, pady=(2, 4))
        self.sent_text.config(state="disabled")

        # Settings - COMPACT at bottom
        settings_card = self._card(left)
        settings_card.grid(row=2, column=0, sticky="ew")
        settings_inner = tk.Frame(settings_card, bg=CARD)
        settings_inner.pack(fill="x", padx=8, pady=3)

        # Settings in a single row
        self._label(settings_inner, "⚙", color=MUTED, size=8).pack(
            side="left", padx=(0, 4)
        )

        # Confidence slider
        conf_frame = tk.Frame(settings_inner, bg=CARD)
        conf_frame.pack(side="left", padx=4)
        self._label(conf_frame, "Conf", color=MUTED, size=7).pack(side="left")
        self._conf_lbl = self._label(
            conf_frame,
            f"{self.confidence_threshold:.2f}",
            color=TEXT,
            size=7,
            bold=True,
        )
        self._conf_lbl.pack(side="left", padx=(2, 0))
        s1 = ttk.Scale(
            conf_frame,
            from_=0.0,
            to=1.0,
            value=self.confidence_threshold,
            command=self._on_conf_change,
            length=50,
        )
        s1.pack(side="left", padx=2)

        # Stable frames
        stable_frame = tk.Frame(settings_inner, bg=CARD)
        stable_frame.pack(side="left", padx=4)
        self._label(stable_frame, "Stable", color=MUTED, size=7).pack(side="left")
        self._stable_lbl = self._label(
            stable_frame, str(self.stable_frames), color=TEXT, size=7, bold=True
        )
        self._stable_lbl.pack(side="left", padx=(2, 0))
        s2 = ttk.Scale(
            stable_frame,
            from_=1,
            to=60,
            value=self.stable_frames,
            command=self._on_stable_change,
            length=50,
        )
        s2.pack(side="left", padx=2)

        # Toggles
        self._add_toggle(settings_inner, "Auto", self.auto_speak_var)
        self._add_toggle(settings_inner, "Space", self.speak_on_space_var)

        # Camera selector
        cams = self.list_cameras()
        self.camera_var = tk.StringVar(value=str(self.camera_index))
        cam_sel_frame = tk.Frame(settings_inner, bg=CARD)
        cam_sel_frame.pack(side="left", padx=4)
        self._label(cam_sel_frame, "📷", color=MUTED, size=8).pack(side="left")
        om = ttk.OptionMenu(
            cam_sel_frame,
            self.camera_var,
            self.camera_var.get(),
            *cams,
            command=self._on_camera_change,
        )
        om.config(width=3)
        om.pack(side="left", padx=(2, 0))

        # ── RIGHT column ──────────────────────────────────────────────────
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=2)
        right.rowconfigure(1, weight=2)
        right.rowconfigure(2, weight=3)
        right.columnconfigure(0, weight=1)

        # Recognition card
        recog_card = self._card(right)
        recog_card.grid(row=0, column=0, sticky="nsew", pady=(0, 4))
        self._label(recog_card, "Recognition", color=TEXT, size=11, bold=True).pack(
            anchor="w", padx=14, pady=(8, 4)
        )

        r_inner = tk.Frame(recog_card, bg=CARD)
        r_inner.pack(fill="x", padx=14)

        self._label(r_inner, "Current Sign", color=MUTED, size=9).grid(
            row=0, column=0, sticky="w"
        )
        self._label(r_inner, "Confidence", color=MUTED, size=9).grid(
            row=0, column=1, sticky="w", padx=(20, 0)
        )

        self.char_var = tk.StringVar(value="Nothing")
        tk.Label(
            r_inner,
            textvariable=self.char_var,
            bg=CARD,
            fg=GREEN,
            font=("Segoe UI", 24, "bold"),
        ).grid(row=1, column=0, sticky="w")

        self.conf_var = tk.StringVar(value="0.00%")
        tk.Label(
            r_inner,
            textvariable=self.conf_var,
            bg=CARD,
            fg=TEXT,
            font=("Segoe UI", 18, "bold"),
        ).grid(row=1, column=1, sticky="w", padx=(20, 0))

        # Confidence bar
        self._conf_bar_canvas = tk.Canvas(
            recog_card, bg=CARD2, height=4, highlightthickness=0
        )
        self._conf_bar_canvas.pack(fill="x", padx=14, pady=(4, 0))
        self._conf_fill = self._conf_bar_canvas.create_rectangle(
            0, 0, 0, 4, fill=BLUE, outline=""
        )

        # Status
        stat_row = tk.Frame(recog_card, bg=CARD)
        stat_row.pack(fill="x", padx=14, pady=(6, 6))
        self._label(stat_row, "Status", color=MUTED, size=9).pack(side="left")
        self._recog_dot = tk.Label(
            stat_row, text="●", bg=CARD, fg=GREEN, font=("Segoe UI", 10)
        )
        self._recog_dot.pack(side="left", padx=(10, 4))
        self.status_var = tk.StringVar(value="Idle")
        tk.Label(
            stat_row,
            textvariable=self.status_var,
            bg=CARD,
            fg=GREEN,
            font=("Segoe UI", 10, "bold"),
        ).pack(side="left")

        # Speech engine card
        speech_card = self._card(right)
        speech_card.grid(row=1, column=0, sticky="nsew", pady=(0, 4))
        self._label(speech_card, "Speech Engine", color=TEXT, size=11, bold=True).pack(
            anchor="w", padx=14, pady=(8, 4)
        )

        sp_inner = tk.Frame(speech_card, bg=CARD)
        sp_inner.pack(fill="x", padx=14, pady=(0, 4))
        sp_inner.columnconfigure(1, weight=1)

        rows = [
            ("Engine", "Piper", BLUE),
            (
                "Model",
                Path(self.tts_model_path).name if self.tts_model_path else "—",
                MUTED,
            ),
        ]
        for i, (lbl, val, col) in enumerate(rows):
            self._label(sp_inner, lbl + ":", color=MUTED, size=9).grid(
                row=i, column=0, sticky="w", pady=2
            )
            self._label(sp_inner, val, color=col, size=9, bold=(col != MUTED)).grid(
                row=i, column=1, sticky="w", padx=(10, 0), pady=2
            )

        # Status
        sp_stat = tk.Frame(speech_card, bg=CARD)
        sp_stat.pack(fill="x", padx=14, pady=(2, 0))
        self._label(sp_stat, "Status:", color=MUTED, size=9).pack(side="left")
        self.speech_var = tk.StringVar(value="Idle")
        tk.Label(
            sp_stat,
            textvariable=self.speech_var,
            bg=CARD,
            fg=BLUE,
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left", padx=(8, 0))

        # Quick Actions card
        qa_card = self._card(right)
        qa_card.grid(row=2, column=0, sticky="nsew")
        self._label(qa_card, "Quick Actions", color=TEXT, size=11, bold=True).pack(
            anchor="w", padx=14, pady=(8, 6)
        )

        qa_grid = tk.Frame(qa_card, bg=CARD)
        qa_grid.pack(fill="both", expand=True, padx=14, pady=(0, 10))
        qa_grid.columnconfigure((0, 1), weight=1)

        # Only 6 buttons now (removed duplicate clear)
        actions = [
            ("▶ Start", BTN_START, self.start_recognition, GREEN),
            ("⏹ Stop", BTN_STOP_C, self.stop_recognition, RED),
            ("␣ Space", BTN_SPACE, self.insert_space, TEAL),
            ("🔊 Word", BTN_SPEAK, self.speak_word, BLUE),
            ("🗣 Sent", BTN_SPKSEN, self.speak_sentence, PURPLE),
            ("🗑 Clear", BTN_CLR, self.clear_sentence, GOLD),
        ]
        for idx, (label, bg, cmd, fg) in enumerate(actions):
            r, c = divmod(idx, 2)
            b = self._icon_btn(qa_grid, label, bg, cmd, fg=fg, width=12)
            b.grid(
                row=r,
                column=c,
                sticky="ew",
                padx=(0 if c == 0 else 4, 4 if c == 0 else 0),
                pady=3,
                ipady=2,
            )
            if idx == 0:
                self.btn_start = b
            if idx == 1:
                self.btn_stop = b

        # ── Bottom status bar ─────────────────────────────────────────────
        bar = tk.Frame(
            self.root,
            bg=CARD,
            height=28,
            highlightthickness=1,
            highlightbackground=BORDER,
        )
        bar.pack(fill="x", side="bottom")
        bar.pack_propagate(False)

        self._status_items = {}
        items = [
            ("camera", "Camera"),
            ("mediapipe", "MP"),
            ("model", "Model"),
            ("piper", "TTS"),
            ("fps", "FPS"),
        ]
        for key, lbl in items:
            f = tk.Frame(bar, bg=CARD)
            f.pack(side="left", padx=(10, 0))
            tk.Label(f, text=lbl, bg=CARD, fg=MUTED, font=("Segoe UI", 8)).pack(
                side="left"
            )
            dot = tk.Label(f, text="●", bg=CARD, fg=GREEN, font=("Segoe UI", 8))
            dot.pack(side="left", padx=(4, 2))
            val = tk.Label(f, text="—", bg=CARD, fg=TEXT, font=("Segoe UI", 8))
            val.pack(side="left")
            self._status_items[key] = (dot, val)

        self._update_statusbar()

    # ── slider helper ──────────────────────────────────────────────────────

    def _add_toggle(self, parent, label, var, command=None):
        f = tk.Frame(parent, bg=CARD)
        f.pack(side="left", padx=(0, 6))
        self._label(f, label, color=MUTED, size=7).pack(side="left", padx=(0, 3))
        sw = self._toggle_switch(f, var, command)
        sw.pack(side="left")

    # ── clock ──────────────────────────────────────────────────────────────

    def _tick_clock(self):
        import datetime

        now = datetime.datetime.now()
        self._clock_var.set(now.strftime("%I:%M:%S %p"))
        self.root.after(1000, self._tick_clock)

    # ── camera loop ────────────────────────────────────────────────────────

    def _camera_loop(self):
        while self.running:
            if self.capture is None or not self.capture.isOpened():
                time.sleep(0.1)
                self.latest_frame = None
                continue

            ok, frame = self.capture.read()
            if not ok:
                time.sleep(0.01)
                continue

            if self.cfg.get("camera", {}).get("mirror", True):
                frame = cv2.flip(frame, 1)

            self._frame_count += 1
            now = time.time()
            if now - self._fps_time >= 1.0:
                self.fps = self._frame_count / (now - self._fps_time)
                self._frame_count = 0
                self._fps_time = now

            prediction = {"label": "nothing", "confidence": 0.0, "hand_present": False}
            if self.recognition_active:
                prediction = self.asl.predict(frame)

            self.latest_frame = frame
            self.latest_prediction = prediction

            if self.recognition_active:
                now_ms = int(time.time() * 1000)
                committed = self.auto_commit.update(
                    prediction.get("label"),
                    float(prediction.get("confidence", 0.0)),
                    now_ms,
                )
                if committed:
                    text_before = self.text_buffer.text
                    if self.text_buffer.commit_label(committed, 1.0):
                        self.last_commit_ms = now_ms
                        self.last_spoken_text = None
                        self._commits.put((committed, text_before))

    # ── UI update loop ─────────────────────────────────────────────────────

    def _update_ui(self):
        if not self.running:
            return

        # Camera frame - now with bigger default size
        if self.latest_frame is not None:
            try:
                widget_w = max(self.cam_label.winfo_width(), 400)
                widget_h = max(self.cam_label.winfo_height(), 300)
                pil_img = Image.fromarray(
                    cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB)
                )
                pil_img = pil_img.resize((widget_w, widget_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(pil_img)
                self.cam_label.config(image=photo)
                self.cam_label.image = photo
            except Exception:
                pass

        pred = self.latest_prediction
        label = str(pred.get("label", "")).strip()
        conf = float(pred.get("confidence", 0.0))
        hand = bool(pred.get("hand_present", False))

        display_label = (
            label.upper() if label and label.lower() != "nothing" else "Nothing"
        )
        self.char_var.set(display_label)
        self.conf_var.set(f"{conf * 100:.2f}%")

        # Confidence bar
        try:
            bw = self._conf_bar_canvas.winfo_width()
            self._conf_bar_canvas.coords(self._conf_fill, 0, 0, bw * conf, 4)
        except Exception:
            pass

        # Hand badge / status
        if hand:
            self._hand_badge.config(text="● Hand Detected", fg=GREEN)
            self.status_var.set("Hand Detected")
            self._recog_dot.config(fg=GREEN)
        else:
            self._hand_badge.config(text="● No Hand", fg=MUTED)
            self.status_var.set("Idle")
            self._recog_dot.config(fg=MUTED)

        # FPS badge
        self._fps_badge.config(text=f"FPS: {self.fps:.1f}")

        # Commits
        while True:
            try:
                lbl, text_before = self._commits.get_nowait()
                self._handle_committed(lbl, text_before)
            except queue.Empty:
                break

        # Update sentence display
        text = self.text_buffer.text
        if getattr(self, "_last_sent_text", None) != text:
            self._last_sent_text = text
            self.sent_text.config(state="normal")
            self.sent_text.delete("1.0", "end")
            self.sent_text.insert("1.0", text)
            self.sent_text.config(state="disabled")
            self.sent_text.see("end")

        self._update_speech_status()
        self._update_statusbar()
        self.root.after(30, self._update_ui)

    # ── committed label handler ────────────────────────────────────────────

    def _handle_committed(self, label: str, text_before: str):
        self.last_commit_ms = int(time.time() * 1000)
        self.last_spoken_text = None
        normalized = label.lower().strip()

        if normalized == self.text_buffer.space_label:
            if (
                self.speak_on_space_var.get()
                and self.tts_enabled_var.get()
                and self.tts_available
            ):
                word = text_before.split()[-1] if text_before else ""
                if word:
                    self.tts.speak(word)
        elif normalized == self.text_buffer.delete_label:
            pass
        else:
            logger.info("Character accepted: %s", normalized)

    # ── speech status ──────────────────────────────────────────────────────

    def _update_speech_status(self):
        speaking = False
        if self.tts_enabled_var.get() and self.tts_available:
            try:
                import pygame

                speaking = pygame.mixer.music.get_busy()
            except Exception:
                pass

        if speaking:
            self._was_speaking = True
            self.speech_status = "Speaking..."
        else:
            if self._was_speaking:
                self.speech_status = "Completed"
                self._was_speaking = False
                self._speech_end_time = time.time()
            elif self.speech_status == "Completed":
                if time.time() - self._speech_end_time > 2.0:
                    self.speech_status = "Idle"

        self.speech_var.set(self.speech_status)

        if (
            self.auto_speak_var.get()
            and self.tts_enabled_var.get()
            and self.tts_available
            and not speaking
        ):
            self._check_auto_speak()

    def _check_auto_speak(self):
        now_ms = int(time.time() * 1000)
        text = self.text_buffer.text
        if not text or len(text) < self.min_chars:
            return
        if self.last_spoken_text == text:
            return
        ends_sentence = text[-1] in ".!?"
        paused_long_enough = (
            self.last_commit_ms > 0
            and (now_ms - self.last_commit_ms) >= self.complete_pause_ms
        )
        if ends_sentence or paused_long_enough:
            self.tts.speak(text)
            self.last_spoken_text = text
            if self.cfg.get("sentence", {}).get("clear_after_speak", False):
                self.text_buffer.clear()
                self.last_spoken_text = None
                self.last_commit_ms = 0

    # ── status bar ─────────────────────────────────────────────────────────

    def _update_statusbar(self):
        cam_ok = self.capture is not None and self.capture.isOpened()
        self._set_status("camera", "Connected" if cam_ok else "Disconnected", cam_ok)

        mp_ok = hasattr(self.asl, "hands") and self.asl.hands is not None
        self._set_status("mediapipe", "Ready" if mp_ok else "Error", mp_ok)

        model_ok = not self.asl.dummy
        self._set_status("model", "Loaded" if model_ok else "Dummy", model_ok)

        piper_ok = self.tts_available
        self._set_status("piper", "Ready" if piper_ok else "Missing", piper_ok)

        self._status_items["fps"][0].config(fg=GREEN)
        self._status_items["fps"][1].config(text=f"{self.fps:.1f}")

    def _set_status(self, key, text, ok: bool):
        color = GREEN if ok else RED
        self._status_items[key][0].config(fg=color)
        self._status_items[key][1].config(text=text)

    # ── controls ───────────────────────────────────────────────────────────

    def start_recognition(self):
        self.recognition_active = True
        self.last_commit_ms = 0
        self.last_spoken_text = None

    def stop_recognition(self):
        self.recognition_active = False

    def insert_space(self):
        """Manually insert a space into the text buffer."""
        text_before = self.text_buffer.text
        if self.text_buffer.commit_label("space", 1.0):
            self.last_commit_ms = int(time.time() * 1000)
            self.last_spoken_text = None
            self._commits.put(("space", text_before))

    def clear_sentence(self):
        self.text_buffer.clear()
        self.last_spoken_text = None
        self.last_commit_ms = 0

    def _copy_sentence(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.text_buffer.text)

    def speak_word(self):
        if not self.tts_enabled_var.get() or not self.tts_available:
            return
        text = self.text_buffer.text
        last_space = text.rfind(" ")
        word = text[last_space + 1 :] if last_space != -1 else text
        if word:
            self.tts.speak(word)

    def speak_sentence(self):
        if not self.tts_enabled_var.get() or not self.tts_available:
            return
        text = self.text_buffer.text
        if text:
            self.tts.speak(text)
            self.last_spoken_text = text

    def stop_speech(self):
        try:
            self.tts.stop(timeout=2.0)
        except Exception:
            pass
        self._was_speaking = False
        self.speech_status = "Idle"
        self._speech_end_time = 0.0

    # ── settings callbacks ─────────────────────────────────────────────────

    def _on_conf_change(self, value):
        self.confidence_threshold = float(value)
        self._conf_lbl.config(text=f"{self.confidence_threshold:.2f}")
        self.auto_commit = self._build_auto_commit()
        self._save_config()

    def _on_stable_change(self, value):
        self.stable_frames = int(float(value))
        self._stable_lbl.config(text=str(self.stable_frames))
        self.auto_commit = self._build_auto_commit()
        self._save_config()

    def _on_cooldown_change(self, value):
        self.cooldown_frames = int(float(value))
        self._cool_lbl.config(text=str(self.cooldown_frames))
        self.auto_commit = self._build_auto_commit()
        self._save_config()

    def _on_camera_change(self, value):
        self.camera_index = int(value)
        self.cfg.setdefault("camera", {})["index"] = self.camera_index
        self._cam_index_lbl.config(text=f"📷  Camera: {self.camera_index}")
        self._open_camera(self.camera_index)
        self._save_config()

    def _save_config(self):
        try:
            import yaml

            path = Path("config.yaml")
            cfg = (
                yaml.safe_load(path.read_text(encoding="utf-8"))
                if path.exists()
                else {}
            )
            cfg = cfg or {}
            cfg.setdefault("sentence", {})
            cfg["sentence"]["auto_speak"] = self.auto_speak_var.get()
            cfg["sentence"]["speak_on_space"] = self.speak_on_space_var.get()
            cfg["sentence"]["stable_frames"] = self.stable_frames
            cfg["sentence"]["cooldown_frames"] = self.cooldown_frames
            cfg.setdefault("asl", {})
            cfg["asl"]["confidence_threshold"] = self.confidence_threshold
            cfg.setdefault("camera", {})
            cfg["camera"]["index"] = self.camera_index
            path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not save config: %s", exc)

    # ── teardown ───────────────────────────────────────────────────────────

    def on_close(self):
        self.running = False
        if hasattr(self, "camera_thread"):
            self.camera_thread.join(timeout=1.0)
        try:
            self.asl.close()
        except Exception:
            pass
        try:
            self.tts.stop(timeout=2.0)
        except Exception:
            pass
        if self.capture is not None:
            try:
                self.capture.release()
            except Exception:
                pass
        self.root.destroy()


def run_gui(cfg: dict):
    root = tk.Tk()
    app = ASLGuiApp(root, cfg)
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
