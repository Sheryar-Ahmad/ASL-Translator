"""Desktop UI. Tk and text state belong to the main thread; capture to its worker."""
import logging
import platform
import queue
import threading
import time
from pathlib import Path
import tkinter as tk
from tkinter import ttk

import cv2
from PIL import Image, ImageOps, ImageTk
import yaml

from app.asl_engine import ASLEngine
from app.config import load_config
from app.text_buffer import AutoCommitController, TextBuffer
from app.tts_engine import TTSEngine

logger = logging.getLogger(__name__)
BG, CARD, BORDER = "#0d1117", "#161b22", "#30363d"
TEXT, MUTED, BLUE, GREEN, RED = "#e6edf3", "#a5afbd", "#58a6ff", "#3fb950", "#ff7b72"


class ASLGuiApp:
    def __init__(self, root: tk.Tk, cfg: dict, config_path="config.yaml"):
        self.root, self.cfg = root, cfg
        self.config_path = Path(config_path)
        self.running = True
        self.recognition_active = False
        self._epoch = 0
        self._packets = queue.Queue(maxsize=1)
        self._stop = threading.Event()
        self._save_job = None
        self._ui_job = None
        self.latest_frame = None
        self.latest_prediction = {}
        self._frame_version = 0
        self._render_key = None
        self._photo = None
        self.capture = None
        self.camera_index = int(cfg.get("camera", {}).get("index", 0))
        self._camera_request = (self.camera_index, 0)
        self.camera_ok = False
        self.camera_error = "Opening camera…"
        self.fps = 0.0
        self.asl = ASLEngine(cfg.get("asl", {}))
        self.tts = TTSEngine(cfg.get("tts", {}))
        self.tts.start()
        self.tts_available = self.tts.available
        labels = cfg.get("asl", {}).get("labels", {})
        self.text_buffer = TextBuffer(**{k: labels.get(k, k.removesuffix("_label"))
                                       for k in ("space_label", "delete_label", "nothing_label")})
        sentence = cfg.get("sentence", {})
        self.auto_speak_var = tk.BooleanVar(value=sentence.get("auto_speak", True))
        self.speak_on_space_var = tk.BooleanVar(value=sentence.get("speak_on_space", True))
        self.confidence_threshold = float(cfg.get("asl", {}).get("confidence_threshold", .65))
        self.stable_frames = max(1, int(sentence.get("stable_frames", 15)))
        self.cooldown_frames = max(1, int(sentence.get("cooldown_frames", 45)))
        self.min_chars = int(sentence.get("min_chars", 3))
        self.complete_pause_ms = int(sentence.get("complete_pause_ms", 2500))
        self.max_sentence_length = max(1, int(sentence.get("max_sentence_length", 200)))
        self.last_commit_ms = 0
        self.last_spoken_text = None
        self.auto_commit = self._build_auto_commit()
        self._build_ui()
        self.camera_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self.camera_thread.start()
        self._update_ui()

    def _build_auto_commit(self):
        cfg = dict(self.cfg.get("asl", {}))
        cfg["confidence_threshold"] = self.confidence_threshold
        fps = max(1.0, float(self.cfg.get("camera", {}).get("fps", 30)))
        cfg["auto"] = dict(cfg.get("auto", {}),
                           stable_ms=max(1, int(self.stable_frames * 1000 / fps)),
                           reset_ms=max(1, int(self.cooldown_frames * 1000 / fps)))
        return AutoCommitController(cfg)

    def _card(self, parent, title):
        frame = tk.Frame(parent, bg=CARD, highlightthickness=1, highlightbackground=BORDER)
        tk.Label(frame, text=title, bg=CARD, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", padx=12, pady=(10, 6))
        return frame

    def _button(self, parent, text, command):
        return tk.Button(parent, text=text, command=command, bg="#253348", fg=TEXT,
                         activebackground="#344862", activeforeground=TEXT,
                         disabledforeground=MUTED, relief="flat", cursor="hand2",
                         padx=10, pady=7, takefocus=True)

    def _build_ui(self):
        self.root.title(self.cfg.get("ui", {}).get("window_name", "ASL Translator"))
        sw, sh = self.root.winfo_screenwidth(), self.root.winfo_screenheight()
        self.root.geometry(f"{min(1120, sw - 60)}x{min(760, sh - 100)}")
        self.root.minsize(min(860, sw - 60), min(560, sh - 100))
        self.root.configure(bg=BG)
        self.root.option_add("*Font", '"Segoe UI" 10')
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("TNotebook", background=BG, borderwidth=0)
        style.configure("TNotebook.Tab", background=CARD, foreground=TEXT, padding=(12, 6))
        style.map("TNotebook.Tab", background=[("selected", "#253348")])
        style.configure("TCheckbutton", background=CARD, foreground=TEXT)
        style.map("TCheckbutton", background=[("active", CARD)])
        header = tk.Frame(self.root, bg=BG)
        header.pack(fill="x", padx=16, pady=(12, 8))
        tk.Label(header, text="ASL Translator", bg=BG, fg=TEXT,
                 font=("Segoe UI", 18, "bold")).pack(side="left")
        tk.Label(header, text="Offline • Camera to text and speech", bg=BG,
                 fg=MUTED).pack(side="right")
        self.notice_var = tk.StringVar(value="Start recognition to build a sentence.")
        tk.Label(self.root, textvariable=self.notice_var, bg=BG, fg=BLUE,
                 anchor="w").pack(fill="x", padx=16, pady=(0, 8))
        body = tk.Frame(self.root, bg=BG)
        body.pack(fill="both", expand=True, padx=16, pady=(0, 10))
        body.columnconfigure(0, weight=1)
        body.columnconfigure(1, weight=0, minsize=310)
        body.rowconfigure(0, weight=1)
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        left.columnconfigure(0, weight=1)
        left.rowconfigure(0, weight=4)
        left.rowconfigure(1, weight=1)
        camera = self._card(left, "Live camera")
        camera.grid(row=0, column=0, sticky="nsew", pady=(0, 10))
        self.cam_canvas = tk.Canvas(camera, bg="#070b10", width=1, height=1,
                                    highlightthickness=0)
        self.cam_canvas.pack(fill="both", expand=True, padx=10, pady=(0, 8))
        self._image_item = self.cam_canvas.create_image(0, 0, anchor="center")
        self._placeholder = self.cam_canvas.create_text(0, 0, text="Opening camera…", fill=MUTED)
        self.camera_var = tk.StringVar(value="Opening camera…")
        tk.Label(camera, textvariable=self.camera_var, bg=CARD, fg=MUTED,
                 anchor="w").pack(fill="x", padx=12, pady=(0, 8))
        sentence = self._card(left, "Your sentence")
        sentence.grid(row=1, column=0, sticky="nsew")
        text_area = tk.Frame(sentence, bg=CARD)
        text_area.pack(fill="both", expand=True, padx=10)
        self.sent_text = tk.Text(text_area, wrap="word", width=1, height=3,
                                 font=("Segoe UI", 14), bg="#1c2333", fg=TEXT,
                                 relief="flat", padx=10, pady=8, state="disabled")
        scroll = ttk.Scrollbar(text_area, command=self.sent_text.yview)
        self.sent_text.configure(yscrollcommand=scroll.set)
        scroll.pack(side="right", fill="y")
        self.sent_text.pack(fill="both", expand=True)
        toolbar = tk.Frame(sentence, bg=CARD)
        toolbar.pack(fill="x", padx=10, pady=8)
        for text, cmd in [("Space", self.insert_space), ("Delete", self.delete_character),
                          ("Copy", self._copy_sentence), ("Clear", self.clear_sentence)]:
            self._button(toolbar, text, cmd).pack(side="left", padx=(0, 5))
        self.count_var = tk.StringVar(value="0 characters")
        tk.Label(toolbar, textvariable=self.count_var, bg=CARD, fg=MUTED).pack(side="right")
        tabs = ttk.Notebook(body)
        tabs.grid(row=0, column=1, sticky="nsew")
        translate, self.translate_scroll = self._scroll_page(tabs, "Translate")
        settings, self.settings_scroll = self._scroll_page(tabs, "Settings")
        self.char_var = tk.StringVar(value="—")
        self.conf_var = tk.StringVar(value="Confidence: —")
        tk.Label(translate, text="Current sign", bg=CARD, fg=MUTED).pack(anchor="w")
        tk.Label(translate, textvariable=self.char_var, bg=CARD, fg=GREEN,
                 font=("Segoe UI", 30, "bold")).pack(anchor="w", pady=(4, 0))
        tk.Label(translate, textvariable=self.conf_var, bg=CARD, fg=TEXT).pack(anchor="w")
        self.conf_bar = ttk.Progressbar(translate, maximum=1)
        self.conf_bar.pack(fill="x", pady=(8, 12))
        self.status_var = tk.StringVar(value="Recognition paused")
        tk.Label(translate, textvariable=self.status_var, bg=CARD, fg=BLUE,
                 wraplength=270, justify="left").pack(anchor="w")
        controls = tk.Frame(translate, bg=CARD)
        controls.pack(fill="x", pady=12)
        self.btn_start = self._button(controls, "Start (F5)", self.start_recognition)
        self.btn_start.pack(side="left", fill="x", expand=True, padx=(0, 6))
        self.btn_stop = self._button(controls, "Pause (Esc)", self.stop_recognition)
        self.btn_stop.pack(side="left", fill="x", expand=True)
        ttk.Separator(translate).pack(fill="x", pady=8)
        tk.Label(translate, text="Speech", bg=CARD, fg=TEXT,
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(4, 8))
        self.speech_var = tk.StringVar()
        tk.Label(translate, textvariable=self.speech_var, bg=CARD, fg=MUTED,
                 wraplength=270, justify="left").pack(anchor="w", pady=(0, 8))
        self.speech_buttons = []
        for text, cmd in [("Speak word", self.speak_word),
                          ("Speak sentence (Ctrl+Enter)", self.speak_sentence),
                          ("Cancel speech", self.stop_speech)]:
            b = self._button(translate, text, cmd)
            b.pack(fill="x", pady=3)
            self.speech_buttons.append(b)
        tk.Label(translate, text="Hold a sign steady to accept it.\nUse Delete to correct the sentence.\nCtrl+Space inserts a word break.",
                 bg=CARD, fg=MUTED, justify="left", wraplength=270).pack(anchor="w", pady=16)
        self._settings_scale(settings, "Confidence threshold", self.confidence_threshold,
                             0.05, 1, self._on_conf_change, "_conf_lbl")
        self._settings_scale(settings, "Stable frames", self.stable_frames,
                             1, 60, self._on_stable_change, "_stable_lbl")
        self._settings_scale(settings, "Cooldown frames", self.cooldown_frames,
                             1, 90, self._on_cooldown_change, "_cool_lbl")
        ttk.Checkbutton(settings, text="Speak sentence after a pause", variable=self.auto_speak_var,
                        command=self._schedule_save).pack(anchor="w", pady=8)
        ttk.Checkbutton(settings, text="Speak word on space", variable=self.speak_on_space_var,
                        command=self._schedule_save).pack(anchor="w", pady=8)
        tk.Label(settings, text="Camera index", bg=CARD, fg=TEXT).pack(anchor="w", pady=(16, 6))
        self.camera_index_var = tk.StringVar(value=str(self.camera_index))
        camera_row = tk.Frame(settings, bg=CARD)
        camera_row.pack(fill="x")
        ttk.Spinbox(camera_row, from_=0, to=20, width=5,
                    textvariable=self.camera_index_var).pack(side="left", padx=(0, 8))
        self._button(camera_row, "Apply / Retry", self._on_camera_change).pack(side="left")
        tk.Label(settings, text="Settings save automatically.\nIf the camera fails, check permissions\nor try another camera index.",
                 bg=CARD, fg=MUTED, justify="left", wraplength=270).pack(anchor="w", pady=16)
        self.root.bind("<F5>", lambda e: self.start_recognition())
        self.root.bind("<Escape>", lambda e: self.stop_recognition())
        self.root.bind("<Control-Return>", lambda e: self.speak_sentence())
        self.root.bind("<Control-space>", lambda e: self.insert_space())
        self.root.bind("<Control-BackSpace>", lambda e: self.delete_character())

    def _scroll_page(self, tabs, title):
        page = tk.Frame(tabs, bg=CARD)
        tabs.add(page, text=title)
        canvas = tk.Canvas(page, bg=CARD, width=1, height=1, highlightthickness=0)
        scrollbar = ttk.Scrollbar(page, orient="vertical", command=canvas.yview)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)
        content = tk.Frame(canvas, bg=CARD, padx=12, pady=12)
        item = canvas.create_window(0, 0, window=content, anchor="nw")
        content.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.bind("<Configure>", lambda e: canvas.itemconfigure(item, width=e.width))
        def focus_visible(event):
            widget = event.widget
            if widget is content or not str(widget).startswith(str(content) + "."):
                return
            total = max(1, content.winfo_height())
            top = widget.winfo_rooty() - content.winfo_rooty()
            bottom = top + widget.winfo_height()
            view_top = canvas.canvasy(0)
            if top < view_top:
                canvas.yview_moveto(top / total)
            elif bottom > view_top + canvas.winfo_height():
                canvas.yview_moveto((bottom - canvas.winfo_height()) / total)
        self.root.bind("<FocusIn>", focus_visible, add="+")
        # Scope wheel handling to this pane, so settings controls keep their bindings.
        def wheel(event):
            if str(event.widget).startswith(str(page)):
                canvas.yview_scroll(-1 if event.delta > 0 else 1, "units")
        self.root.bind("<MouseWheel>", wheel, add="+")
        return content, canvas

    def _settings_scale(self, parent, title, value, low, high, callback, attr):
        row = tk.Frame(parent, bg=CARD)
        row.pack(fill="x", pady=(6, 4))
        tk.Label(row, text=title, bg=CARD, fg=TEXT).pack(side="left")
        label = tk.Label(row, text=f"{value:.2f}" if high == 1 else str(value), bg=CARD, fg=BLUE)
        label.pack(side="right")
        setattr(self, attr, label)
        scale = ttk.Scale(parent, from_=low, to=high, value=value, command=callback)
        scale.pack(fill="x", pady=(0, 12))

    def _publish(self, packet):
        try:
            self._packets.get_nowait()
        except queue.Empty:
            pass
        self._packets.put_nowait(packet)

    def _camera_loop(self):
        request = None
        capture = None
        last_frame = time.monotonic()
        try:
            while not self._stop.is_set():
                if request != self._camera_request:
                    request = self._camera_request
                    if capture is not None:
                        capture.release()
                    capture = cv2.VideoCapture(request[0], cv2.CAP_DSHOW if platform.system() == "Windows" else cv2.CAP_ANY)
                    self.capture = capture
                    cam = self.cfg.get("camera", {})
                    if capture.isOpened():
                        for prop, key, default in [(cv2.CAP_PROP_FRAME_WIDTH, "width", 640),
                                                   (cv2.CAP_PROP_FRAME_HEIGHT, "height", 480),
                                                   (cv2.CAP_PROP_FPS, "fps", 30)]:
                            capture.set(prop, int(cam.get(key, default)))
                if not capture.isOpened():
                    self._publish((None, {}, self._epoch, 0, "Camera unavailable. Check permissions or select another index."))
                    self._stop.wait(.2)
                    continue
                epoch = self._epoch
                active = self.recognition_active
                ok, frame = capture.read()
                if not ok:
                    self._publish((None, {}, self._epoch, 0, "Camera frame unavailable. Use Apply / Retry in Settings."))
                    self._stop.wait(.1)
                    continue
                if self.cfg.get("camera", {}).get("mirror", True):
                    frame = cv2.flip(frame, 1)
                try:
                    pred = self.asl.predict(frame) if active else {}
                except Exception as exc:
                    logger.exception("Recognition failed")
                    self._publish((None, {}, epoch, 0, f"Recognition error: {exc}"))
                    self._stop.wait(.1)
                    continue
                now = time.monotonic()
                fps = 1 / max(now - last_frame, .001)
                last_frame = now
                self._publish((frame, pred, epoch, fps, ""))
        except Exception as exc:
            logger.exception("Camera worker failed")
            self._publish((None, {}, self._epoch, 0, f"Recognition error: {exc}"))
        finally:
            if capture is not None:
                capture.release()
            self.asl.close()

    def _update_ui(self):
        if not self.running:
            return
        try:
            frame, pred, epoch, fps, error = self._packets.get_nowait()
            if epoch == self._epoch:
                self.camera_ok = frame is not None
                self.camera_error = error
                self.latest_frame = frame
                self._frame_version += 1
                self.fps = fps
                self.latest_prediction = pred
                if error:
                    self.stop_recognition()
                    self.notice_var.set(error)
                if self.recognition_active:
                    label = self.auto_commit.update(pred.get("label"), float(pred.get("confidence", 0)),
                                                    int(time.monotonic() * 1000))
                    if label:
                        self._commit(label)
        except queue.Empty:
            pass
        self._render_frame()
        pred = self.latest_prediction
        label = pred.get("label", "")
        self.char_var.set(label.upper() if label and label != self.text_buffer.nothing_label else "—")
        confidence = float(pred.get("confidence", 0))
        self.conf_var.set(f"Confidence: {confidence:.0%}")
        self.conf_bar["value"] = confidence
        self.status_var.set("Model missing: train or install the ASL model." if self.asl.dummy else
                            "Recognition paused" if not self.recognition_active else
                            "Hand detected" if pred.get("hand_present") else "Show a hand to the camera")
        self.btn_start.configure(state="normal" if self.camera_ok and not self.asl.dummy and not self.recognition_active else "disabled")
        self.btn_stop.configure(state="normal" if self.recognition_active else "disabled")
        self.tts_available = self.tts.available
        if not self.tts_available:
            self.speech_var.set("Voice unavailable. Run scripts/download_models.py to install the model and voice configuration.")
        else:
            self.speech_var.set(self.tts.last_error or self.tts.status)
        has_text = bool(self.text_buffer.text)
        for i, button in enumerate(self.speech_buttons):
            enabled = self.tts.busy if i == 2 else self.tts_available and has_text
            button.configure(state="normal" if enabled else "disabled")
        text = self.text_buffer.text
        if getattr(self, "_displayed_text", None) != text:
            self._displayed_text = text
            self.sent_text.configure(state="normal")
            self.sent_text.delete("1.0", "end")
            self.sent_text.insert("1.0", text)
            self.sent_text.configure(state="disabled")
            self.sent_text.see("end")
            self.count_var.set(f"{len(text)}/{self.max_sentence_length}")
        self._check_auto_speak()
        self._ui_job = self.root.after(30, self._update_ui)

    def _render_frame(self):
        w, h = max(1, self.cam_canvas.winfo_width()), max(1, self.cam_canvas.winfo_height())
        key = (self._frame_version, w, h)
        if key == self._render_key:
            return
        self._render_key = key
        self.cam_canvas.coords(self._image_item, w / 2, h / 2)
        self.cam_canvas.coords(self._placeholder, w / 2, h / 2)
        self.cam_canvas.itemconfigure(self._placeholder, width=max(1, w - 30))
        if self.latest_frame is None:
            self.cam_canvas.itemconfigure(self._image_item, image="")
            self.cam_canvas.itemconfigure(self._placeholder, text=self.camera_error, state="normal")
            self.camera_var.set(self.camera_error)
            return
        image = Image.fromarray(cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2RGB))
        image = ImageOps.contain(image, (w, h), Image.Resampling.LANCZOS)
        self._photo = ImageTk.PhotoImage(image)
        self.cam_canvas.itemconfigure(self._image_item, image=self._photo)
        self.cam_canvas.itemconfigure(self._placeholder, state="hidden")
        height, width = self.latest_frame.shape[:2]
        self.camera_var.set(f"Camera {self.camera_index} • {width} × {height} • {self.fps:.0f} FPS")

    def _commit(self, label):
        before = self.text_buffer.text
        normalized = str(label).lower().strip()
        if normalized not in (self.text_buffer.delete_label, self.text_buffer.nothing_label, self.text_buffer.space_label):
            extra = len(normalized) + (1 if self.text_buffer.parts and self.text_buffer.parts[-1] == " " else 0)
            if len(before) + extra > self.max_sentence_length:
                self.notice_var.set("Sentence limit reached. Speak or clear the text to continue.")
                return False
        if not self.text_buffer.commit_label(label):
            return False
        self.last_commit_ms = int(time.monotonic() * 1000)
        self.last_spoken_text = None
        if normalized == self.text_buffer.space_label and self.speak_on_space_var.get() and self.tts.available:
            words = before.split()
            if words:
                self.tts.speak(words[-1])
        return True

    def _check_auto_speak(self):
        text = self.text_buffer.text
        if not self.auto_speak_var.get() or not self.tts.available or self.tts.busy:
            return
        if not text or len(text) < self.min_chars or text == self.last_spoken_text:
            return
        paused = self.last_commit_ms > 0 and time.monotonic() * 1000 - self.last_commit_ms >= self.complete_pause_ms
        if text[-1] in ".!?" or paused:
            self.speak_sentence()
            if self.cfg.get("sentence", {}).get("clear_after_speak", False):
                self.clear_sentence()

    def start_recognition(self):
        if not self.camera_ok or self.asl.dummy:
            return
        self._epoch += 1
        self.auto_commit.reset()
        self.recognition_active = True
        self.notice_var.set("Recognition active. Hold each sign steady; pause to speak the sentence.")

    def stop_recognition(self):
        self.recognition_active = False
        self._epoch += 1
        self.auto_commit.reset()
        self.latest_prediction = {}
        self.notice_var.set("Recognition paused. Your sentence is kept.")

    def insert_space(self):
        self._commit(self.text_buffer.space_label)

    def delete_character(self):
        self._commit(self.text_buffer.delete_label)

    def clear_sentence(self):
        self._epoch += 1
        self.auto_commit.reset()
        self.text_buffer.clear()
        self.last_commit_ms = 0
        self.last_spoken_text = None
        self.notice_var.set("Sentence cleared.")

    def _copy_sentence(self):
        if self.text_buffer.text:
            self.root.clipboard_clear()
            self.root.clipboard_append(self.text_buffer.text)
            self.notice_var.set("Sentence copied.")

    def speak_word(self):
        words = self.text_buffer.text.split()
        if self.tts.available and words:
            self.tts.speak(words[-1])

    def speak_sentence(self):
        text = self.text_buffer.text
        if self.tts.available and text:
            self.tts.speak(text)
            self.last_spoken_text = text

    def stop_speech(self):
        self.tts.cancel()

    def _on_conf_change(self, value):
        self.confidence_threshold = float(value)
        self._conf_lbl.configure(text=f"{self.confidence_threshold:.2f}")
        self.auto_commit = self._build_auto_commit()
        self._schedule_save()

    def _on_stable_change(self, value):
        self.stable_frames = max(1, round(float(value)))
        self._stable_lbl.configure(text=str(self.stable_frames))
        self.auto_commit = self._build_auto_commit()
        self._schedule_save()

    def _on_cooldown_change(self, value):
        self.cooldown_frames = max(1, round(float(value)))
        self._cool_lbl.configure(text=str(self.cooldown_frames))
        self.auto_commit = self._build_auto_commit()
        self._schedule_save()

    def _on_camera_change(self):
        try:
            index = int(self.camera_index_var.get())
            if index < 0:
                raise ValueError()
        except ValueError:
            self.notice_var.set("Camera index must be a non-negative whole number.")
            return
        self.stop_recognition()
        self.camera_index = index
        self._camera_request = (index, self._camera_request[1] + 1)
        self.camera_ok = False
        self.latest_frame = None
        self.camera_error = "Opening camera…"
        self._frame_version += 1
        self._schedule_save()

    def _schedule_save(self):
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
        self._save_job = self.root.after(350, self._save_config)

    def _save_config(self):
        self._save_job = None
        try:
            cfg = load_config(str(self.config_path))
            cfg["sentence"].update(auto_speak=self.auto_speak_var.get(),
                                   speak_on_space=self.speak_on_space_var.get(),
                                   stable_frames=self.stable_frames, cooldown_frames=self.cooldown_frames)
            cfg["asl"]["confidence_threshold"] = self.confidence_threshold
            cfg["camera"]["index"] = self.camera_index
            self.config_path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not save settings: %s", exc)
            self.notice_var.set(f"Could not save settings: {exc}")

    def on_close(self):
        self.running = False
        self.recognition_active = False
        self._stop.set()
        if self._ui_job is not None:
            self.root.after_cancel(self._ui_job)
        if self._save_job is not None:
            self.root.after_cancel(self._save_job)
            self._save_config()
        self.tts.stop(timeout=2)
        self.camera_thread.join(timeout=2)
        self.root.destroy()


def run_gui(cfg: dict, config_path="config.yaml"):
    root = tk.Tk()
    try:
        app = ASLGuiApp(root, cfg, config_path)
    except Exception:
        root.destroy()
        raise
    root.protocol("WM_DELETE_WINDOW", app.on_close)
    root.mainloop()
