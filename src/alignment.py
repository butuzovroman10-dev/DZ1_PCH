import cv2
import numpy as np
import math

def detect_rotation_angle(image, config):
    """
    Определяет угол поворота изображения по горизонтальным линиям (сетка).
    Возвращает угол в градусах (отрицательный = поворот по часовой).
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)

    lines = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=config['image_processing']['hough_lines_threshold'],
        minLineLength=config['image_processing']['min_line_length'],
        maxLineGap=config['image_processing']['max_line_gap']
    )

    if lines is None:
        return 0.0

    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
        # Оставляем только почти горизонтальные линии (допуск ±10°)
        if abs(angle) < 10:
            angles.append(angle)

    if not angles:
        return 0.0

    median_angle = np.median(angles)
    return -median_angle  # поворачиваем в обратную сторону

def rotate_image(image, angle):
    """Поворачивает изображение на заданный угол (градусы)."""
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos = abs(M[0, 0])
    sin = abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_CUBIC)
    return rotated

def align_image(image, config):
    """Основная функция выравнивания."""
    angle = detect_rotation_angle(image, config)
    tolerance = config['image_processing']['rotation_tolerance']
    if abs(angle) > tolerance:
        aligned = rotate_image(image, angle)
        print(f"Изображение повёрнуто на {angle:.2f} градусов")
        return aligned
    else:
        print("Изображение уже горизонтально")
        return image