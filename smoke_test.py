# smoke_test.py
import time

from lumika import SileroTTSEngine

def main():
    print("[SMOKE] Initializing SileroTTSEngine...")
    engine = SileroTTSEngine()

    texts = [
        ("ru", "Это тестовая фраза для проверки русской озвучки Лумики."),
        ("en", "This is a test sentence for the English Lumika voice."),
    ]

    for lang, text in texts:
        print(f"[SMOKE] Synthesizing {lang} text...")
        buf = engine._synth_segment(text, lang)
        assert buf is not None and buf.size > 0
        print(f"[SMOKE] {lang} buffer length: {buf.size}")

    # Дадим потокам чуть-чуть пожить (на всякий случай)
    time.sleep(1)
    print("[SMOKE] Done. TTS pipeline works in headless mode.")

if __name__ == "__main__":
    main()
