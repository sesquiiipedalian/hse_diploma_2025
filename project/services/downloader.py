"""
Скачивание файлов из Google Drive через gdown.
"""
import os
import gdown


class ModelDownloader:
    """
    Класс для скачивания моделей и ресурсов по Google Drive ID или URL.

    Аргументы:
        cache_dir (str): директория для кеширования загрузок.
    """
    def __init__(self, cache_dir: str = 'models'):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)

    def download(self, url_or_id: str, filename: str | None = None, quiet: bool = True) -> str:
        """
        Скачивает файл из Google Drive.

        Аргументы:
            url_or_id (str): URL или ID файла.
            filename (str|None): имя сохраняемого файла; по умолчанию берётся из URL.
            quiet (bool): отключает прогресс-бар gdown.

        Возвращает полный путь к скачанному файлу.
        """
        url = url_or_id
        if len(url_or_id) == 33 and not url_or_id.startswith('http'):
            # считаем, что передан только ID
            url = f'https://drive.google.com/uc?id={url_or_id}'
        output = filename or url.split('id=')[-1]
        dest = os.path.join(self.cache_dir, output)
        if not os.path.exists(dest):
            gdown.download(url, output=dest, quiet=quiet)
        return dest
