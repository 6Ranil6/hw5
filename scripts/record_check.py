#!/usr/bin/env python3
"""Record the real make check output; optionally render that transcript to MP4.

uv run --with pillow python scripts/record_check.py --render
Requires ffmpeg only for rendering. No Git mutations or pushes are performed.
"""
import argparse
import codecs
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def render(events, duration, target):
    from PIL import Image, ImageDraw, ImageFont
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg не найден; текстовая запись уже сохранена")
    font_paths = [Path('/System/Library/Fonts/Menlo.ttc'), Path('/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf')]
    font_path = next((p for p in font_paths if p.exists()), None)
    if font_path is None:
        raise RuntimeError("Не найден моноширинный шрифт для видео")
    font = ImageFont.truetype(str(font_path), 18)
    width, height, fps = 1800, 1120, 2
    command = [ffmpeg, '-hide_banner', '-loglevel', 'error', '-y', '-f', 'rawvideo', '-pixel_format', 'rgb24', '-video_size', f'{width}x{height}', '-framerate', str(fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264', '-preset', 'fast', '-crf', '24', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', str(target)]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    index, text = 0, ''
    try:
        for frame in range(math.ceil((duration + 8) * fps)):
            moment = frame / fps
            while index < len(events) and events[index][0] <= moment:
                text += events[index][2]
                index += 1
            lines = []
            for line in ANSI.sub('', text).replace('\r', '').splitlines():
                lines.extend(textwrap.wrap(line, width=155, replace_whitespace=False) or [''])
            picture = Image.new('RGB', (width, height), '#0b1220')
            draw = ImageDraw.Draw(picture)
            draw.text((24, 16), 'MLOps / ДЗ 5 — фактический вывод make check', font=font, fill='#94a3b8')
            for i, line in enumerate(lines[-46:]):
                color = '#4ade80' if '✓' in line or 'Все проверки пройдены' in line else '#e2e8f0'
                if '✗' in line or 'Провалено проверок' in line:
                    color = '#f87171'
                draw.text((24, 64 + 22 * i), line, font=font, fill=color)
            process.stdin.write(picture.tobytes())
    finally:
        process.stdin.close()
    if process.wait() != 0:
        raise RuntimeError('ffmpeg завершился с ошибкой')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'submission')
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    events = [[0.0, 'o', '$ make check\r\n']]
    header = {'version': 2, 'width': 155, 'height': 46, 'timestamp': int(time.time()), 'command': 'make check', 'title': 'HW5: actual make check run', 'env': {'TERM': 'xterm-256color'}}
    child_env = os.environ.copy()
    # Pillow may run in uv's temporary environment; tests use the project .venv.
    child_env.pop('VIRTUAL_ENV', None)
    process = subprocess.Popen(['make', 'check'], cwd=ROOT, env=child_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    decoder = codecs.getincrementaldecoder('utf-8')()
    chunks = []
    while True:
        chunk = os.read(process.stdout.fileno(), 65536)
        if not chunk:
            break
        text = decoder.decode(chunk)
        if text:
            chunks.append(text)
            events.append([round(time.monotonic() - started, 4), 'o', text])
    tail = decoder.decode(b'', final=True)
    if tail:
        chunks.append(tail)
        events.append([round(time.monotonic() - started, 4), 'o', tail])
    code = process.wait()
    duration = time.monotonic() - started
    output = ''.join(chunks)
    cast = args.output_dir / 'make-check.cast'
    cast.write_text('\n'.join(json.dumps(item, ensure_ascii=False) for item in [header, *events]) + '\n', encoding='utf-8')
    log = ANSI.sub('', output).replace(str(ROOT), 'hw5-broken')
    (ROOT / 'docs/check.log').write_text(log, encoding='utf-8')
    info = {'command': 'make check', 'exit_code': code, 'duration_seconds': round(duration, 3), 'cast_sha256': hashlib.sha256(cast.read_bytes()).hexdigest(), 'capture': 'actual subprocess stdout/stderr; video renders the recorded transcript', 'video_rendered': False}
    if args.render:
        video = args.output_dir / 'make-check.mp4'
        render(events, duration, video)
        info['video_rendered'] = True
        info['video_sha256'] = hashlib.sha256(video.read_bytes()).hexdigest()
    (ROOT / 'docs/recording.json').write_text(json.dumps(info, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(info, ensure_ascii=False, indent=2))
    if code:
        raise SystemExit(code)
    if 'Все проверки пройдены.' not in output:
        raise SystemExit('Нет итоговой строки успеха в записи')


if __name__ == '__main__':
    main()
