"""
Пакет web: Flask-приложение.
"""
from flask import Flask

app = Flask(
    __name__,
    template_folder="templates",
    static_folder="static"
)

# маршруты
import web.app  # noqa
