import cv2
import numpy as np
import pytesseract
import re
import yaml

def load_image(path):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Не удалось загрузить изображение: {path}")
    return img

def save_image(img, path):
    cv2.imwrite(path, img)

def show_image(img, title="Image", wait=True):
    cv2.imshow(title, img)
    if wait:
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def load_config(config_path="config/settings.yaml"):
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def ocr_extract_metadata(image, config):
    """
    Пытается извлечь длительность и диапазон амплитуды из текста на изображении.
    Возвращает (duration_sec, amp_range) или (None, None) если не удалось.
    """
    if not config['timeseries']['use_ocr']:
        return None, None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Улучшаем качество для OCR
    gray = cv2.resize(gray, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    text = pytesseract.image_to_string(gray, lang=config['ocr']['language'])
    # Ищем длительность: число и единицы (с, сек, sec)
    duration_pattern = r'(\d+(?:\.\d+)?)\s*(?:с|сек|sec|s)'
    match = re.search(duration_pattern, text, re.IGNORECASE)
    duration = float(match.group(1)) if match else None
    # Ищем амплитуду: например "10 мВ" или "0.05 mV"
    amp_pattern = r'(\d+(?:\.\d+)?)\s*(мВ|mV|мкВ|uV)'
    matches = re.findall(amp_pattern, text, re.IGNORECASE)
    amp_range = None
    if matches:
        # берём первое попавшееся значение как масштаб (предполагаем, что это цена деления)
        scale_val = float(matches[0][0])
        unit = matches[0][1].lower()
        if unit in ['мв', 'mv']:
            amp_range = (-scale_val*5, scale_val*5)  # грубое предположение
        else:
            amp_range = (-scale_val*5, scale_val*5)
    return duration, amp_range