import cv2
import numpy as np
from src.utils import load_image, save_image

def find_graph_regions(image, config):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cv2.imwrite("data/processed/debug_binary.png", binary)

    kernel = np.ones((5,5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    cv2.imwrite("data/processed/debug_closed.png", closed)

    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    print(f"[DEBUG] Найдено контуров: {len(contours)}")

    regions = []
    min_area = config['graph_extraction']['min_graph_area']
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        area = w * h
        if area > min_area and h > 50:
            regions.append((x, y, w, h))
            print(f"[DEBUG] Кандидат: x={x}, y={y}, w={w}, h={h}, area={area}")

    regions.sort(key=lambda r: r[1])
    expected = config['graph_extraction']['expected_graphs_count']
    print(f"[DEBUG] Отобрано областей: {len(regions)}")
    
    # Если нашли ровно 2 – отлично
    if len(regions) >= expected:
        return regions[:expected]
    # Если нашли 1 – пробуем разделить
    elif len(regions) == 1:
        print("[DEBUG] Найдена одна область, пробуем разделить на две")
        splitted = split_region_vertically(image, regions[0], config)
        if splitted:
            return list(splitted)
        else:
            return regions
    else:
        return regions

def extract_graphs(image, config):
    """
    Возвращает два изображения (первый и второй график).
    Если метод manual – использует координаты из конфига.
    """
    method = config['graph_extraction']['method']
    if method == "manual":
        boxes = config['graph_extraction']['manual_boxes']
        first = image[boxes[0][1]:boxes[0][3], boxes[0][0]:boxes[0][2]]
        second = image[boxes[1][1]:boxes[1][3], boxes[1][0]:boxes[1][2]]
        return first, second
    else:
        regions = find_graph_regions(image, config)
        if len(regions) < 2:
            raise RuntimeError(f"Найдено только {len(regions)} областей, ожидалось 2. Попробуйте ручную настройку.")
        # Предполагаем, что первый график – выше, второй – ниже
        x1, y1, w1, h1 = regions[0]
        x2, y2, w2, h2 = regions[1]
        graph1 = image[y1:y1+h1, x1:x1+w1]
        graph2 = image[y2:y2+h2, x2:x2+w2]
        if config['visualization']['save_graph_crops']:
            save_image(graph1, "data/processed/first_graph.png")
            save_image(graph2, "data/processed/second_graph.png")
        return graph1, graph2
def split_region_vertically(image, region, config):
    """
    Если найдена одна большая область, пробует разделить её на две части
    по горизонтальной линии, где мало чёрных пикселей (промежуток между графиками).
    Возвращает две области (x,y,w,h) или None.
    """
    x, y, w, h = region
    roi = image[y:y+h, x:x+w]
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    
    # Суммируем чёрные пиксели по горизонтали (проекция на Y)
    vertical_proj = np.sum(binary, axis=1) / 255
    # Ищем минимальные значения (промежутки)
    # Сглаживаем для устойчивости
    kernel = np.ones(5)
    smoothed = np.convolve(vertical_proj, kernel, mode='same')
    
    # Определяем порог: 10% от максимальной суммы
    threshold = np.max(smoothed) * 0.1
    gaps = np.where(smoothed < threshold)[0]
    
    if len(gaps) == 0:
        # Если нет явного промежутка, делим пополам
        mid = h // 2
        return (x, y, w, mid), (x, y+mid, w, h-mid)
    
    # Берём самый широкий промежуток в середине
    # (упрощённо: берём среднее арифметическое между мин и макс)
    gap_start = gaps[0]
    gap_end = gaps[-1]
    split_y = (gap_start + gap_end) // 2 + y
    return (x, y, w, split_y - y), (x, split_y, w, y + h - split_y)