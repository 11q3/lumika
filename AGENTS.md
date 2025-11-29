\# Lumika – instructions for agents



\## What this project is



Lumika is a Windows screen-reader app (OCR + Silero TTS, ru+en).



In cloud containers (like Codex), only \*\*headless tests\*\* should be run.

Do \*\*not\*\* try to run `python lumika.py` there: it expects Windows hotkeys

and a system tray.



\## Setup



The code execution environment installs:



\- System: `tesseract-ocr`, `tesseract-ocr-eng`, `tesseract-ocr-rus`, `ffmpeg`

\- Python: `pip install -r requirements.txt`



\## How to run tests



Use the smoke test that avoids GUI and audio playback:



```bash

python smoke\_test.py



