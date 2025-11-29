import os
import sys
import time
import threading
import ctypes
import inspect
from pathlib import Path

import re
import queue
import subprocess

import keyboard
import mss
from PIL import Image, ImageFilter, ImageOps, ImageDraw
import pytesseract

import numpy as np
import simpleaudio as sa

import torch
import tkinter as tk

import num2words

from pystray import Icon, Menu, MenuItem


# ---------------------------------------------------------------------------
#  Console suppression (for PyInstaller onefile without attached console)
# ---------------------------------------------------------------------------

class _NullIO:
    def write(self, *_, **__):
        pass

    def flush(self):
        pass


if getattr(sys, "stderr", None) is None:
    sys.stderr = _NullIO()
if getattr(sys, "stdout", None) is None:
    sys.stdout = _NullIO()

# ---------------------------------------------------------------------------
#  Global config
# ---------------------------------------------------------------------------

SPEED_MIN_PERCENT = 10
SPEED_MAX_PERCENT = 400
SPEED_STEP_PERCENT = 25

BASE_DIR = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))

ffmpeg_exe = BASE_DIR / "ffmpeg.exe"
if ffmpeg_exe.exists():
    os.environ["PATH"] = str(BASE_DIR) + os.pathsep + os.environ.get("PATH", "")


def _msg_box(title: str, text: str):
    try:
        ctypes.windll.user32.MessageBoxW(0, text, title, 0x00000010)
    except Exception:
        pass


def ensure_ffmpeg_available():
    print("[FFMPEG] Checking ffmpeg availability...")
    try:
        res = subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )
        if res.returncode != 0:
            msg = (
                "ffmpeg.exe is required but returned non-zero exit code.\n\n"
                "Place ffmpeg.exe next to Lumika.exe or add it to PATH."
            )
            print("[FFMPEG]", msg)
            _msg_box("Lumika - ffmpeg error", msg)
            sys.exit(1)
        print("[FFMPEG] ffmpeg is available.")
    except FileNotFoundError:
        msg = (
            "ffmpeg.exe was not found.\n\n"
            "Place ffmpeg.exe next to Lumika.exe or add it to PATH."
        )
        print("[FFMPEG]", msg)
        _msg_box("Lumika - ffmpeg missing", msg)
        sys.exit(1)
    except Exception as e:
        msg = f"Unexpected error when checking ffmpeg: {e}"
        print("[FFMPEG]", msg)
        _msg_box("Lumika - ffmpeg error", msg)
        sys.exit(1)


def setup_tesseract():
    embedded = BASE_DIR / "Tesseract-OCR" / "tesseract.exe"
    if embedded.exists():
        tess_dir = embedded.parent
        pytesseract.pytesseract.tesseract_cmd = str(embedded)
        os.environ["TESSDATA_PREFIX"] = str(tess_dir / "tessdata")
        print(f"[OCR] Using embedded Tesseract at {embedded}")
        print(f"[OCR] TESSDATA_PREFIX = {os.environ.get('TESSDATA_PREFIX')}")
    else:
        print("[OCR] Using system Tesseract (tesseract.exe must be in PATH)")


# ---------------------------------------------------------------------------
#  Screen region capture
# ---------------------------------------------------------------------------

def capture_region():
    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    pt = POINT()
    try:
        ctypes.windll.user32.GetCursorPos(ctypes.byref(pt))
        cursor_x, cursor_y = pt.x, pt.y
    except Exception:
        cursor_x, cursor_y = 0, 0

    with mss.mss() as sct:
        monitors = sct.monitors

    target_mon = None
    for m in monitors[1:]:
        left, top = m["left"], m["top"]
        right = left + m["width"]
        bottom = top + m["height"]
        if left <= cursor_x < right and top <= cursor_y < bottom:
            target_mon = m
            break

    if target_mon is None:
        target_mon = monitors[1] if len(monitors) > 1 else monitors[0]

    mon_left = target_mon["left"]
    mon_top = target_mon["top"]
    mon_width = target_mon["width"]
    mon_height = target_mon["height"]

    root = tk.Tk()
    root.overrideredirect(True)
    root.attributes("-topmost", True)
    root.attributes("-alpha", 0.3)
    root.configure(bg="black")
    root.geometry(f"{mon_width}x{mon_height}+{mon_left}+{mon_top}")

    canvas = tk.Canvas(root, cursor="cross", bg="black", highlightthickness=0)
    canvas.pack(fill="both", expand=True)

    sel = {"x1": None, "y1": None, "x2": None, "y2": None, "lx1": None, "ly1": None}

    def clamp_local(x: int, y: int):
        return max(0, min(mon_width - 1, x)), max(0, min(mon_height - 1, y))

    def on_press(event):
        ex, ey = clamp_local(event.x, event.y)
        sel["lx1"], sel["ly1"] = ex, ey
        sel["x1"], sel["y1"] = mon_left + ex, mon_top + ey
        sel["x2"], sel["y2"] = sel["x1"], sel["y1"]
        canvas.delete("rect")

    def on_drag(event):
        ex, ey = clamp_local(event.x, event.y)
        sel["x2"], sel["y2"] = mon_left + ex, mon_top + ey
        canvas.delete("rect")
        if sel["lx1"] is not None and sel["ly1"] is not None:
            canvas.create_rectangle(
                sel["lx1"],
                sel["ly1"],
                ex,
                ey,
                outline="red",
                width=2,
                tag="rect",
            )

    def on_release(event):
        ex, ey = clamp_local(event.x, event.y)
        sel["x2"], sel["y2"] = mon_left + ex, mon_top + ey
        root.quit()

    def on_escape(_event):
        sel["x1"] = sel["y1"] = sel["x2"] = sel["y2"] = None
        root.quit()

    root.bind("<ButtonPress-1>", on_press)
    root.bind("<B1-Motion>", on_drag)
    root.bind("<ButtonRelease-1>", on_release)
    root.bind("<Escape>", on_escape)

    root.mainloop()
    root.destroy()

    if None in (sel["x1"], sel["y1"], sel["x2"], sel["y2"]):
        return None

    x1, y1, x2, y2 = sel["x1"], sel["y1"], sel["x2"], sel["y2"]
    left, top = min(x1, x2), min(y1, y2)
    width, height = abs(x2 - x1), abs(y2 - y2)

    # bug: should be y2 - y1; fix:
    height = abs(y2 - y1)

    if width < 5 or height < 5:
        return None

    with mss.mss() as sct:
        monitor = {"left": left, "top": top, "width": width, "height": height}
        shot = sct.grab(monitor)
        img = Image.frombytes("RGB", shot.size, shot.rgb)

    return img


# ---------------------------------------------------------------------------
#  OCR + text preprocessing
# ---------------------------------------------------------------------------

WORD_OR_SPACE_RE = re.compile(r"\S+|\s+")
WHITESPACE_RE = re.compile(r"\s+")
NON_SPEECH_RE = re.compile(r"[\u200b]+")
ALNUM_RE = re.compile(r"[0-9A-Za-zА-Яа-яЁё]")
NUMBER_RE = re.compile(r"\d+([.,]\d+)?")


def _alnum_ratio(text: str) -> float:
    """Heuristic: share of non-space chars that are digits/letters."""
    if not text:
        return 0.0
    total = sum(1 for ch in text if not ch.isspace())
    if total == 0:
        return 0.0
    valid = len(ALNUM_RE.findall(text))
    return valid / total


def ocr_image(img: Image.Image) -> str:
    """
    1) Normalize resolution (scale small regions up, huge ones down).
    2) Try several preprocess variants (gray / bw / inverted).
    3) Run Tesseract with BOTH lang orders: eng+rus and rus+eng.
    4) Blacklist junk characters like '$' that you never want.
    5) Choose the result with the best 'text-like' score.
    """
    w, h = img.size
    max_side = max(w, h)

    TARGET_MIN = 900   # scale UP if smaller than this
    TARGET_MAX = 2200  # scale DOWN if larger than this

    scale = 1.0
    if max_side < TARGET_MIN:
        scale = TARGET_MIN / max_side
    elif max_side > TARGET_MAX:
        scale = TARGET_MAX / max_side

    if abs(scale - 1.0) > 1e-2:
        new_size = (int(w * scale), int(h * scale))
        img = img.resize(new_size, Image.LANCZOS)
        print(
            f"[OCR] Rescaled from {w}x{h} to {img.size[0]}x{img.size[1]} "
            f"(scale {scale:.2f}x)"
        )

    gray = img.convert("L")
    gray = ImageOps.autocontrast(gray)
    smooth = gray.filter(ImageFilter.MedianFilter(size=3))

    bw = smooth.point(lambda x: 255 if x > 160 else 0)
    bw_inv = ImageOps.invert(bw)

    # Try several preprocess variants
    pre_variants = [
        ("gray", gray),
        ("bw", bw),
        ("bwinv", bw_inv),
    ]

    # Try both language priorities
    lang_orders = ["eng+rus", "rus+eng"]

    # Config: keep spaces, set DPI, and blacklist '$' and a bit of obvious trash.
    base_config = (
        "--psm 6 --oem 1 "
        "-c preserve_interword_spaces=1 "
        "-c user_defined_dpi=300 "
        "-c tessedit_char_blacklist=$@#"
    )

    best_text = ""
    best_score = -1.0
    t0_global = time.perf_counter()

    for lang in lang_orders:
        for name, im in pre_variants:
            t0 = time.perf_counter()
            try:
                txt = pytesseract.image_to_string(im, lang=lang, config=base_config)
            except Exception as e:
                print(f"[OCR] ERROR in variant {name} lang={lang}: {e}")
                continue
            dt = time.perf_counter() - t0
            score = _alnum_ratio(txt)
            print(
                f"[OCR] {name} | {lang}: {dt:.3f}s, score={score:.3f}, "
                f"len={len(txt.strip())}"
            )
            if score > best_score:
                best_score = score
                best_text = txt

    print(
        f"[OCR] Done in {time.perf_counter() - t0_global:.3f}s, "
        f"best_score={best_score:.3f}"
    )
    return best_text.strip()

def _is_cyrillic(ch: str) -> bool:
    return ("А" <= ch <= "я") or ch in "Ёё"


def _is_latin(ch: str) -> bool:
    return ("A" <= ch <= "Z") or ("a" <= ch <= "z")


CUSTOM_DICT_PATTERNS = []


RU_WORD_FIXES = {
    "A": "я",
    "a": "я",
    "Я": "я",   # normalize case a bit
    "OH": "Он",
    "Oh": "Он",
    "ON": "Он",
    "On": "Он",
}


def fix_ru_specific_words(text: str) -> str:
    """
    Fix very short Latin-only tokens that in Russian context are almost
    certainly 'я', 'Он', etc. This runs only for lang='ru' segments.
    """
    if not text:
        return text

    parts = re.split(r"(\s+)", text)
    for i, part in enumerate(parts):
        if not part or part.isspace():
            continue
        repl = RU_WORD_FIXES.get(part)
        if repl:
            parts[i] = repl
    return "".join(parts)


def split_ru_en_segments(text: str):
    tokens = WORD_OR_SPACE_RE.findall(text)
    segments = []
    cur_lang = None
    buf = []

    def detect_token_lang(tok, prev):
        cyr = sum(1 for ch in tok if _is_cyrillic(ch))
        lat = sum(1 for ch in tok if _is_latin(ch))
        if cyr == 0 and lat == 0:
            return prev
        if cyr > lat:
            return "ru"
        if lat > cyr:
            return "en"
        return prev or "ru"

    for tok in tokens:
        tok_lang = detect_token_lang(tok, cur_lang)
        if cur_lang is None and tok_lang is None:
            tok_lang = "ru"
        if cur_lang is None:
            cur_lang = tok_lang or "ru"
            buf.append(tok)
            continue
        if tok_lang is None or tok_lang == cur_lang:
            buf.append(tok)
        else:
            seg_text = "".join(buf).strip()
            if seg_text:
                segments.append((seg_text, cur_lang))
            buf = [tok]
            cur_lang = tok_lang

    if buf:
        seg_text = "".join(buf).strip()
        if seg_text:
            segments.append((seg_text, cur_lang or "ru"))

    return segments


# ---------------------------------------------------------------------------
#  Custom dictionary with tolerant matching (Tarkov etc.)
# ---------------------------------------------------------------------------

# Characters that often get confused between LATIN / CYRILLIC / symbols.
AMBIG_CHAR_GROUPS = {
    # Uppercase pairs
    "A": "AА",
    "B": "BВ",
    "C": "CС",
    "E": "EЕ",
    "H": "HН",
    "K": "KК",
    "M": "MМ",
    "O": "OО0",
    "P": "PР",
    "T": "TТ",
    "X": "XХ",
    "Y": "YУ",

    # Lowercase
    "a": "aа",
    "c": "cс",
    "e": "eе",
    "o": "oо0",
    "p": "pр",
    "x": "xх",
    "y": "yу",

    # Extra for TerraGroup style garbage
    "R": "RГ",
    "r": "rг",
    "G": "GБ",
    "g": "gб",
    "U": "UИ",
    "u": "uи",

    # S vs $ vs 5
    "S": "S$5",
    "s": "s$5",
}


def _build_ambiguous_pattern(original: str) -> str:
    """
    Build regex fragment tolerant to Cyrillic/Latin lookalikes and common OCR
    substitutions.

    TerraGroup -> [TТ][eе][rг][rг][aа][GБ][rг][oо0][uи][pр]
    HHS-1      -> [HН][HН][S$5]-1
    """
    parts: list[str] = []
    for ch in original:
        group = AMBIG_CHAR_GROUPS.get(ch)
        if group:
            parts.append("[" + re.escape(group) + "]")
        else:
            parts.append(re.escape(ch))
    return "".join(parts)


def load_custom_dict():
    global CUSTOM_DICT_PATTERNS
    path = BASE_DIR / "lumika_dict.txt"
    if not path.exists():
        print("[DICT] No lumika_dict.txt found, skipping custom dictionary")
        CUSTOM_DICT_PATTERNS = []
        return

    patterns = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = [p.strip() for p in line.split("|")]
                if len(parts) < 3:
                    continue
                original, ru, en = parts[0], parts[1], parts[2]
                if not original:
                    continue

                pattern_body = _build_ambiguous_pattern(original)
                pat = re.compile(r"(?i)\b" + pattern_body + r"\b")
                patterns.append((pat, ru, en))
    except Exception as e:
        print(f"[DICT] Failed to load lumika_dict.txt: {e}")
        patterns = []

    CUSTOM_DICT_PATTERNS = patterns
    print(f"[DICT] Loaded {len(CUSTOM_DICT_PATTERNS)} entries from lumika_dict.txt")


def apply_custom_dict(text: str, lang: str) -> str:
    if not CUSTOM_DICT_PATTERNS:
        return text
    for pat, ru, en in CUSTOM_DICT_PATTERNS:
        repl = ru if lang == "ru" else en
        text = pat.sub(repl, text)
    return text


# ---------------------------------------------------------------------------
#  Fix typical OCR confusions in numbers
# ---------------------------------------------------------------------------

def _fix_digit_like_chars(token: str) -> str:
    """
    Fix typical OCR confusions only when the token is mostly digits.
    e.g. '5O0' -> '500', 'l0' -> '10', 'НН$-1' still matches HHS-1 via dict.
    """
    if not token:
        return token

    digits = sum(ch.isdigit() for ch in token)
    nonspace = sum(1 for ch in token if not ch.isspace())
    if nonspace == 0:
        return token

    if digits / nonspace < 0.60:
        return token

    trans = str.maketrans(
        {
            "O": "0",
            "o": "0",
            "О": "0",  # Cyrillic O
            "I": "1",
            "l": "1",
            "S": "5",
            "s": "5",
        }
    )
    return token.translate(trans)


def fix_ocr_confusions(text: str) -> str:
    """
    Run _fix_digit_like_chars on each token, preserving whitespace.
    """
    if not text:
        return text

    parts = re.split(r"(\s+)", text)
    for i, part in enumerate(parts):
        if not part or part.isspace():
            continue
        parts[i] = _fix_digit_like_chars(part)
    return "".join(parts)


def replace_numbers_with_words(text: str, lang: str) -> str:
    num_lang = "ru" if lang == "ru" else "en"

    def _repl(match: re.Match) -> str:
        raw = match.group(0)
        normalized = raw.replace(",", ".")
        try:
            value = int(round(float(normalized))) if "." in normalized else int(normalized)
        except Exception:
            return raw
        try:
            spoken = num2words.num2words(value, lang=num_lang)
        except Exception:
            return raw
        return spoken

    return NUMBER_RE.sub(_repl, text)


def preprocess_for_tts(text: str, lang: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    text = NON_SPEECH_RE.sub("", text)
    text = WHITESPACE_RE.sub(" ", text)

    # Numbers (0/O/S etc.) -> more sane digits
    text = fix_ocr_confusions(text)

    # Russian-specific small word fixes like A -> я, OH -> Он
    if lang == "ru":
        text = fix_ru_specific_words(text)

    text = replace_numbers_with_words(text, lang)
    text = WHITESPACE_RE.sub(" ", text).strip()
    if lang == "ru" and text and text[-1] not in ".!?…":
        text += "."
    return text



# ---------------------------------------------------------------------------
#  Audio helpers
# ---------------------------------------------------------------------------

def trim_silence(buf: np.ndarray, threshold: int = 700, margin: int = 600) -> np.ndarray:
    if buf.size == 0:
        return buf
    abs_buf = np.abs(buf)
    idx = np.where(abs_buf > threshold)[0]
    if idx.size == 0:
        return buf
    start = max(0, int(idx[0]) - margin)
    end = min(buf.size, int(idx[-1]) + 1 + margin)
    return buf[start:end]


def show_speed_popup(speed_percent: int, lang: str | None = None):
    def _popup():
        try:
            root = tk.Tk()
            root.overrideredirect(True)
            root.attributes("-topmost", True)
            root.configure(bg="black")

            label_text = (
                f"TTS {lang.upper()} speed: {speed_percent}%"
                if lang else f"TTS speed: {speed_percent}%"
            )
            label = tk.Label(
                root,
                text=label_text,
                fg="white",
                bg="black",
                font=("Segoe UI", 14, "bold"),
                padx=20,
                pady=10,
            )
            label.pack()

            root.update_idletasks()
            w, h = root.winfo_width(), root.winfo_height()
            sw, sh = root.winfo_screenwidth(), root.winfo_screenheight()
            x, y = (sw - w) // 2, (sh - h) // 2
            root.geometry(f"{w}x{h}+{x}+{y}")

            root.after(700, root.destroy)
            root.mainloop()
        except Exception as e:
            print(f"[UI] Speed popup error: {e}")

    threading.Thread(target=_popup, daemon=True).start()


# ---------------------------------------------------------------------------
#  Silero TTS engine
# ---------------------------------------------------------------------------

class SileroTTSEngine:
    BILINGUAL_KEY = "bi"

    def __init__(self):
        self.device = "cpu"
        self.sample_rate = 24000
        self.speed = {self.BILINGUAL_KEY: 100}
        self.extra = {self.BILINGUAL_KEY: {"put_accent": True, "put_yo": True}}
        self.models = {}
        self._load_models()

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._audio_queue = queue.Queue(maxsize=8)
        self._current_play_obj = None
        self._player_thread = threading.Thread(
            target=self._player_loop,
            name="LumikaAudioPlayer",
            daemon=True,
        )
        self._player_thread.start()
        print("[TTS] Warming up models...")
        self._warmup()
        self._thread = None

    @staticmethod
    def _filter_extra_kwargs(model, extras: dict) -> dict:
        extras = dict(extras or {})
        try:
            sig = inspect.signature(model.apply_tts)
        except Exception:
            return extras
        params = sig.parameters
        if any(p.kind == inspect.Parameter.VAR_KEYWORD for p in params.values()):
            return extras
        allowed = {k: v for k, v in extras.items() if k in params}
        if len(allowed) != len(extras):
            removed = sorted(set(extras) - set(allowed))
            if removed:
                print(
                    "[TTS] Stripping unsupported apply_tts extras: "
                    + ", ".join(removed)
                )
        return allowed

    def _load_models(self):
        """
        Load a bilingual Silero model to synthesize RU+EN text in one pass.
        This is the only supported mode now; failure to load is fatal.
        """

        try:
            print("[TTS] Loading Silero bilingual model via torch.hub (CPU)...")
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-models",
                model="silero_tts",
                language="multi",
                speaker="multi_v2",
            )
            if hasattr(model, "to"):
                model.to(self.device)
            if hasattr(model, "eval"):
                model.eval()
            voice = None
            if hasattr(model, "speakers") and model.speakers:
                voice = model.speakers[0]
            if not voice:
                voice = "en_0"
            self.models[self.BILINGUAL_KEY] = {"model": model, "voice": voice}
            self.speed = {self.BILINGUAL_KEY: 100}
            extras = {"put_accent": True, "put_yo": True}
            extras = self._filter_extra_kwargs(model, extras)
            self.extra = {self.BILINGUAL_KEY: extras}
            print("[TTS] Bilingual Silero model loaded successfully.")
            return
        except Exception as e:
            msg = (
                "[TTS] Bilingual model load failed and no fallback is allowed. "
                "Please ensure the Silero multi_v2 speaker is available."
            )
            print(f"{msg} Error: {e}")
            _msg_box("Lumika - TTS init error", msg)
            os._exit(1)

    def _warmup(self):
        warm = {
            self.BILINGUAL_KEY: "This warmup line mixes русский и English so the bilingual model primes both alphabets before real playback.",
        }
        for lang in self.models:
            txt = warm.get(lang, warm[self.BILINGUAL_KEY])
            model = self.models[lang]["model"]
            voice = self.models[lang]["voice"]
            for i in range(2):
                t0 = time.time()
                try:
                    with self._lock, torch.no_grad():
                        text = preprocess_for_tts(txt, "ru" if lang == self.BILINGUAL_KEY else lang)
                        kwargs = {"speaker": voice, "sample_rate": self.sample_rate}
                        kwargs.update(self.extra.get(lang, {}))
                        _ = self._apply_model_tts(model, text, kwargs)
                    dt = time.time() - t0
                    print(f"[TTS] Warmup {lang} pass {i+1} done in {dt:.3f}s")
                except Exception as e:
                    print(f"[TTS] Warmup {lang} pass {i+1} failed: {e}")
                    break

    def _apply_model_tts(self, model, text: str, kwargs: dict):
        """
        Call Silero's apply_tts handling legacy ``text=``/``texts=`` keywords
        as well as positional-only signatures (e.g., multi_v2).
        """

        kwargs = dict(kwargs or {})
        speaker = kwargs.pop("speaker", None)
        sample_rate = kwargs.pop("sample_rate", None)
        base_kwargs = dict(kwargs)

        attempts = []

        def add_attempts(extra_kwargs: dict):
            extra_kwargs = dict(extra_kwargs or {})
            full_kw = dict(extra_kwargs)
            if sample_rate is not None:
                full_kw["sample_rate"] = sample_rate
            if speaker is not None:
                full_kw["speaker"] = speaker

            # Keyword first (both text= and texts=) with provided extras.
            attempts.append(((), {**full_kw, "text": text}))
            attempts.append(((), {**full_kw, "texts": [text]}))

            # Retry without speaker keyword in case it's positional-only.
            no_speaker_kw = dict(full_kw)
            no_speaker_kw.pop("speaker", None)
            attempts.append(((), {**no_speaker_kw, "text": text}))
            attempts.append(((), {**no_speaker_kw, "texts": [text]}))

            # Positional fallbacks (text, speaker[, sample_rate], **extras).
            if speaker is not None:
                pos_args = [text, speaker]
                attempts.append((tuple(pos_args), {**extra_kwargs}))
                if sample_rate is not None:
                    attempts.append((tuple(pos_args + [sample_rate]), {**extra_kwargs}))
            # Pure positional text with remaining kwargs.
            attempts.append(((text,), {**extra_kwargs}))

        # Try with full extras first, then a minimal set (e.g., for multi_v2 which rejects put_accent).
        add_attempts(base_kwargs)
        if base_kwargs:
            add_attempts({})

        last_exc = None
        for args, kw in attempts:
            try:
                return model.apply_tts(*args, **kw)
            except TypeError as exc:
                last_exc = exc
            except Exception:
                # Preserve the most relevant TypeError chain if all fail.
                raise
        if last_exc:
            raise last_exc
        raise RuntimeError("apply_tts invocation failed without exception.")

    @staticmethod
    def _build_atempo_filters(speed_ratio: float) -> str:
        s = float(speed_ratio)
        if s <= 0:
            return "atempo=1.0"
        factors = []
        while s > 2.0:
            factors.append(2.0)
            s /= 2.0
        while s < 0.5:
            factors.append(0.5)
            s /= 0.5
        factors.append(s)
        parts = [f"{f:.3f}".rstrip("0").rstrip(".") for f in factors if abs(f - 1.0) > 1e-3]
        if not parts:
            parts = ["1.0"]
        return ",".join(f"atempo={p}" for p in parts)

    def _time_stretch_ffmpeg(self, buf: np.ndarray, speed_percent: int) -> np.ndarray:
        speed_percent = max(SPEED_MIN_PERCENT, min(SPEED_MAX_PERCENT, int(speed_percent)))
        ratio = speed_percent / 100.0
        if abs(ratio - 1.0) < 1e-3:
            return buf
        filter_str = self._build_atempo_filters(ratio)
        cmd = [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "-i",
            "pipe:0",
            "-filter:a",
            filter_str,
            "-f",
            "s16le",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(self.sample_rate),
            "-ac",
            "1",
            "pipe:1",
        ]
        try:
            res = subprocess.run(
                cmd,
                input=buf.tobytes(),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            if res.returncode != 0 or not res.stdout:
                err = res.stderr.decode("utf-8", "ignore") if res.stderr else ""
                msg = f"ffmpeg atempo failed (code={res.returncode}):\n{err}"
                print("[TTS]", msg)
                _msg_box("Lumika - ffmpeg error", msg)
                os._exit(1)
            out = np.frombuffer(res.stdout, dtype=np.int16)
            if out.size == 0:
                msg = "ffmpeg atempo produced empty output buffer."
                print("[TTS]", msg)
                _msg_box("Lumika - ffmpeg error", msg)
                os._exit(1)
            return out
        except FileNotFoundError:
            msg = (
                "ffmpeg.exe was not found at runtime, but is required.\n\n"
                "Place ffmpeg.exe next to Lumika.exe or add it to PATH."
            )
            print("[TTS]", msg)
            _msg_box("Lumika - ffmpeg missing", msg)
            os._exit(1)
        except Exception as e:
            msg = f"Unexpected ffmpeg atempo exception: {e}"
            print("[TTS]", msg)
            _msg_box("Lumika - ffmpeg error", msg)
            os._exit(1)

    def _player_loop(self):
        while True:
            item = self._audio_queue.get()
            try:
                lang, buf = item
                if self._stop_event.is_set() or buf.size == 0:
                    self._audio_queue.task_done()
                    continue
                with self._lock:
                    speed_percent = self.speed.get(lang, 100)
                buf_to_play = self._time_stretch_ffmpeg(buf, speed_percent)
                play_obj = sa.play_buffer(
                    buf_to_play.tobytes(),
                    num_channels=1,
                    bytes_per_sample=2,
                    sample_rate=self.sample_rate,
                )
                with self._lock:
                    self._current_play_obj = play_obj
                while play_obj.is_playing():
                    if self._stop_event.is_set():
                        play_obj.stop()
                        break
                    time.sleep(0.01)
                self._audio_queue.task_done()
            except Exception as e:
                print(f"[TTS] Player error: {e}")
                self._audio_queue.task_done()

    def stop(self, clear_flag: bool = False):
        self._stop_event.set()
        with self._lock:
            play_obj = self._current_play_obj
        if play_obj is not None:
            try:
                if play_obj.is_playing():
                    play_obj.stop()
            except Exception:
                pass
        try:
            while True:
                self._audio_queue.get_nowait()
                self._audio_queue.task_done()
        except queue.Empty:
            pass
        t = self._thread
        if t and t.is_alive():
            try:
                t.join(timeout=0.5)
            except RuntimeError:
                pass
        if clear_flag:
            self._stop_event.clear()

    def speak_async(self, text: str):
        text = (text or "").strip()
        if not text:
            return
        self.stop(clear_flag=True)
        t = threading.Thread(target=self._worker, args=(text,), daemon=True)
        self._thread = t
        t.start()

    def set_speed_percent(self, lang: str, value: int):
        lang = self._resolve_lang_key(lang)
        value = int(round(value / SPEED_STEP_PERCENT) * SPEED_STEP_PERCENT)
        value = max(SPEED_MIN_PERCENT, min(SPEED_MAX_PERCENT, value))
        with self._lock:
            self.speed[lang] = value
        print(f"[TTS] Playback speed set to {value}%")
        show_speed_popup(value)

    def change_speed_step(self, lang: str, delta_step: int):
        lang = self._resolve_lang_key(lang)
        with self._lock:
            base = self.speed[lang]
        self.set_speed_percent(lang, base + delta_step * SPEED_STEP_PERCENT)

    def _worker(self, text: str):
        if self._stop_event.is_set():
            return
        segments = split_ru_en_segments(text)
        if not segments:
            return
        merged = self._merge_segments_for_bilingual(segments)
        if not merged:
            return
        lang_key = self.BILINGUAL_KEY
        print("[TTS] Using bilingual model for merged text (preview):")
        preview = merged.replace("\n", " ")[:80]
        print(f"   - [{lang_key}] '{preview}'")
        buf = self._synth_segment(merged, lang_key)
        if buf is not None and buf.size:
            try:
                self._audio_queue.put((lang_key, buf), timeout=0.1)
            except queue.Full:
                print("[TTS] Audio queue full, dropping synthesis result.")

    def _synth_segment(self, text: str, lang: str):
        text = self._apply_lang_specific_preprocessing(text, lang)
        if not text or self._stop_event.is_set():
            return None
        model = self.models[lang]["model"]
        voice = self.models[lang]["voice"]
        print(f"[TTS] Synth | lang={lang}, speaker={voice}, len={len(text)}")
        t0 = time.time()
        try:
            with self._lock, torch.no_grad():
                kwargs = {"speaker": voice, "sample_rate": self.sample_rate}
                kwargs.update(self.extra.get(lang, {}))
                audio = self._apply_model_tts(model, text, kwargs)
        except Exception as e:
            print(f"[TTS] Synth error ({lang}): {e}")
            return None
        print(f"[TTS] Synth done in {time.time() - t0:.3f}s")
        if isinstance(audio, torch.Tensor):
            audio = audio.detach().cpu().numpy()
        if isinstance(audio, (list, tuple)) and audio:
            audio = audio[0]
        audio = np.array(audio, dtype=np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        if not audio.size:
            return None
        max_val = float(np.max(np.abs(audio)))
        if max_val > 0:
            audio = audio / max_val
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767.0).astype(np.int16)
        audio_int16 = trim_silence(audio_int16)
        return audio_int16

    def _resolve_lang_key(self, _requested: str) -> str:
        return self.BILINGUAL_KEY

    def _merge_segments_for_bilingual(self, segments):
        normalized_parts = []
        for seg_text, lang in segments:
            prepared = self._apply_lang_specific_preprocessing(seg_text, lang)
            if prepared:
                normalized_parts.append(prepared)
        merged = " ".join(normalized_parts).strip()
        return merged

    def _apply_lang_specific_preprocessing(self, text: str, lang: str) -> str:
        lang_key = self._resolve_lang_key(lang)
        # Even with bilingual synthesis we still want per-language cleanup for
        # dictionary replacements and number naming.
        if lang_key == self.BILINGUAL_KEY:
            lang_for_cleanup = "ru" if any(_is_cyrillic(ch) for ch in text) else "en"
        else:
            lang_for_cleanup = lang_key
        text = apply_custom_dict(text, lang_for_cleanup)
        text = preprocess_for_tts(text, lang_for_cleanup)
        return text


# ---------------------------------------------------------------------------
#  Tray icon
# ---------------------------------------------------------------------------

def _load_tray_image() -> Image.Image:
    ico_path = BASE_DIR / "lumika.ico"
    if ico_path.exists():
        try:
            return Image.open(ico_path)
        except Exception as e:
            print(f"[TRAY] Failed to load lumika.ico: {e}")
    img = Image.new("RGBA", (64, 64), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img)
    draw.rectangle((8, 8, 56, 56), outline="white", width=2)
    draw.text((18, 18), "L", fill="white")
    return img


# ---------------------------------------------------------------------------
#  Hotkeys
# ---------------------------------------------------------------------------

tts_engine = None
_last_speed_hotkey_time = 0.0
_SPEED_HOTKEY_DEBOUNCE = 0.25


def handle_hotkey():
    global tts_engine
    print("\n[STATE] Capture hotkey pressed → capture region...")
    img = capture_region()
    if img is None:
        print("[STATE] Selection cancelled.")
        return
    text = ocr_image(img)
    if not text:
        print("[OCR] Empty result.")
        return
    print("\n--- RECOGNIZED (SELECTED AREA) ---")
    print(text)
    print("----------------------------------\n")
    if tts_engine is None:
        print("[TTS] Engine not initialized yet.")
        return
    tts_engine.speak_async(text)


def handle_stop_hotkey():
    global tts_engine
    if tts_engine is None:
        return
    print("[TTS] Stop requested via ESC.")
    tts_engine.stop(clear_flag=False)


def _handle_speed_change(lang: str, delta_steps: int):
    global tts_engine, _last_speed_hotkey_time
    now = time.time()
    if now - _last_speed_hotkey_time < _SPEED_HOTKEY_DEBOUNCE:
        return
    _last_speed_hotkey_time = now
    if tts_engine is None:
        return
    tts_engine.change_speed_step(lang, delta_steps)


def handle_speed_up():
    _handle_speed_change(SileroTTSEngine.BILINGUAL_KEY, +1)


def handle_speed_down():
    _handle_speed_change(SileroTTSEngine.BILINGUAL_KEY, -1)


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    global tts_engine
    setup_tesseract()
    ensure_ffmpeg_available()
    load_custom_dict()
    print("Lumika — screen reader (OCR + neural TTS, bilingual RU+EN, CPU)")
    print("Scroll Lock — capture screen (monitor under cursor) and read text.")
    print("Esc — stop current TTS.")
    print("Alt+PageUp / Alt+PageDown — change playback speed (10–400%, step 25%).")
    print("Close tray icon (right-click → Quit) to exit.")
    print("OCR: Tesseract eng+rus; TTS: Silero (bilingual RU+EN, CPU).")
    print("Speed uses ffmpeg 'atempo' to keep pitch constant; ffmpeg.exe must be in PATH.\n")
    print("[TTS] Initializing Silero models via torch.hub (CPU)...")
    tts_engine = SileroTTSEngine()

    keyboard.add_hotkey("scroll lock", handle_hotkey)
    keyboard.add_hotkey("esc", handle_stop_hotkey)
    keyboard.add_hotkey("alt+page up", handle_speed_up)
    keyboard.add_hotkey("alt+page down", handle_speed_down)

    tray_image = _load_tray_image()

    def on_quit(icon, _item):
        print("\n[STATE] Exiting from tray...")
        if tts_engine is not None:
            tts_engine.stop(clear_flag=True)
        icon.stop()
        os._exit(0)

    menu = Menu(MenuItem("Quit Lumika", on_quit))
    tray_icon = Icon("Lumika", tray_image, "Lumika", menu)
    print("\n[STATE] Lumika is running. Use Scroll Lock to read, ESC to stop.")
    print("[STATE] You can exit via tray icon menu (bottom-right system tray).")
    tray_icon.run()


if __name__ == "__main__":
    main()