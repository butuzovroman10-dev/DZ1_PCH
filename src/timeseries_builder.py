import numpy as np
import cv2
from scipy.interpolate import interp1d

def extract_curve_points(graph_img):
    """
    Извлекает точки кривой из изображения графика.
    Возвращает список (x, y) в пиксельных координатах.
    """
    gray = cv2.cvtColor(graph_img, cv2.COLOR_BGR2GRAY)
    # Инвертируем, чтобы кривая была белой на чёрном
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    h, w = binary.shape
    points = []
    for col in range(w):
        # Находим все белые пиксели в столбце
        col_pixels = np.where(binary[:, col] == 255)[0]
        if len(col_pixels) > 0:
            # Берём среднюю точку (если кривая толстая)
            y = int(np.mean(col_pixels))
            points.append((col, y))
    return points

def pixel_to_value(points, img_width, img_height, time_range, amp_range, flip_y=True):
    """
    Преобразует пиксельные координаты в физические величины.
    time_range = (t_min, t_max) секунд
    amp_range = (a_min, a_max) в единицах сигнала
    flip_y: инвертировать ли Y (обычно на изображении 0 сверху)
    """
    t_vals = []
    a_vals = []
    for x_px, y_px in points:
        t = time_range[0] + (x_px / img_width) * (time_range[1] - time_range[0])
        if flip_y:
            y_norm = 1.0 - (y_px / img_height)
        else:
            y_norm = y_px / img_height
        amp = amp_range[0] + y_norm * (amp_range[1] - amp_range[0])
        t_vals.append(t)
        a_vals.append(amp)
    return t_vals, a_vals

def resample_timeseries(t, v, target_fs, t_min, t_max):
    """
    Передискретизирует временной ряд на равномерную сетку.
    target_fs – частота дискретизации (Гц).
    """
    t_uniform = np.linspace(t_min, t_max, int((t_max - t_min) * target_fs) + 1)
    f = interp1d(t, v, kind='linear', fill_value='extrapolate')
    v_uniform = f(t_uniform)
    return t_uniform, v_uniform

def build_timeseries(graph_img, duration, amp_range, sampling_rate=100):
    """
    Основной метод: строит временной ряд для одного графика.
    duration – длительность исследования (сек)
    amp_range – кортеж (min_amp, max_amp) в физических единицах
    """
    h, w = graph_img.shape[:2]
    points = extract_curve_points(graph_img)
    if not points:
        raise ValueError("Не удалось извлечь кривую из графика")
    t_vals, a_vals = pixel_to_value(
        points, w, h,
        time_range=(0, duration),
        amp_range=amp_range,
        flip_y=True
    )
    t_uniform, a_uniform = resample_timeseries(t_vals, a_vals, sampling_rate, 0, duration)
    return t_uniform, a_uniform