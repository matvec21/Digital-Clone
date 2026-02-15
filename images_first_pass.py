from pathlib import Path

import cv2
import os

from PIL import Image

from lottie.importers.core import import_tgs
from lottie.exporters.cairo import export_png

import shutil

def tgs_to_png(tgs_path, output_path):
    try:
        animation = import_tgs(tgs_path)
        export_png(animation, output_path, frame = 0)
    except:
        print('error')

def webp_to_png(path, output_path):
    with Image.open(path) as img:
        img.save(output_path, 'PNG')

def extract_frame(video_path, output_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames // 2)
    
    ret, frame = cap.read()
    if ret:
        cv2.imwrite(output_path, frame)
    
    cap.release()
    return ret

source = Path("data")
destination = Path("images")

for file in source.rglob("*"):
    output_path = destination / file.relative_to(source)
    if file.is_dir():
        output_path.mkdir(parents=True, exist_ok=True)

    if not file.is_file():
        continue

    print(f"Обрабатываю: {file}")

    _output_path = output_path.with_suffix('.png')

    if file.suffix == '.webm' or file.suffix == '.mp4':
        extract_frame(str(file), str(_output_path))
    elif file.suffix == '.tgs':
        tgs_to_png(str(file), str(_output_path))
    elif file.suffix == '.webp':
        webp_to_png(str(file), str(_output_path))
    elif file.suffix == '.json':
        print('SKIP', file)
    else:
        shutil.copy(str(file), str(output_path))
