"""
Маршруты и логика Flask-приложения.
"""
import os
import cv2
import numpy as np
from flask import render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
from PIL import Image

from core.device import get_default_device
from core.keyframes import KeyFrameExtractor
from core.detection import YOLODetector
from core.matting import ModNetMatting
from core.diffusion import Img2ImgProcessor
from core.filters import SimilarityFilter
from core.wallpapers import WallpaperProcessor
from web import app

# Конфигурация
UPLOAD_FOLDER = os.path.join(app.static_folder, 'uploads')
ALLOWED_EXTENSIONS = {'mp4', 'mov', 'avi'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


def allowed_file(filename: str) -> bool:
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/result', methods=['POST'])
def result():
    video = request.files.get('video')
    if video is None or not allowed_file(video.filename):
        return redirect(url_for('index'))

    filename = secure_filename(video.filename)
    video_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(video_path)

    device = get_default_device()

    # Экстрактор ключевых кадров
    extractor = KeyFrameExtractor()
    frames = extractor.extract(video_path)

    # Детектор
    detector = YOLODetector(device=device)
    person_frames = [f for f in frames if any(d['name']=='person' for d in detector.detect(f))]

    # Маттирование
    mat = ModNetMatting(weights_path='models/modnet.pth', device=device)
    mattes = [mat.apply(Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))) for f in person_frames]

    # Стилизация
    diff = Img2ImgProcessor(model_name='CompVis/stable-diffusion-v1-4', device=device)
    stylized = [diff.run(m, prompt='wallpaper') for m in mattes]

    # Фильтрация похожих
    sim = SimilarityFilter()
    unique_frames = sim.filter([np.array(img.convert('RGB')) for img in stylized])

    # Сохранение обоев
    wp = WallpaperProcessor(output_dir=os.path.join(app.static_folder, 'wallpapers'), size=(1920,1080))
    paths = wp.process([Image.fromarray(f) for f in unique_frames], prefix=filename.rsplit('.',1)[0])

    # Преобразуем пути в относительные для шаблона
    rel_paths = [os.path.relpath(p, app.static_folder) for p in paths]
    return render_template('result.html', images=rel_paths)

