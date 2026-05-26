import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.alignment import align_image
from src.graph_extractor import extract_graphs
from src.timeseries_builder import build_timeseries
from src.csv_exporter import save_to_csv
from src.utils import load_image, save_image, load_config, ocr_extract_metadata

def main():
    # Конфигурация
    config = load_config()
    image_path = "data/raw/1-1.png"
    output_csv = "data/output/timeseries_output.csv"
    
    # 1. Загрузка и выравнивание
    img = load_image(image_path)
    aligned = align_image(img, config)
    if config['visualization']['save_aligned']:
        save_image(aligned, "data/processed/aligned.png")
    
    # 2. Определение размерности и длительности исследования
    # Пытаемся через OCR, если не удаётся – берём из конфига
    duration, amp_range = ocr_extract_metadata(aligned, config)
    if duration is None:
        duration = 10.0  # значение по умолчанию (длительность 10 секунд)
        print("Длительность не распознана, используется 10 с")
    if amp_range is None:
        amp_range = (-1.0, 1.0)  # по умолчанию амплитуда от -1 до 1
        print("Диапазон амплитуды не распознан, используется (-1, 1)")
    
    # 3-4. Выделение двух графиков
    graph1, graph2 = extract_graphs(aligned, config)
    
    # 5. Построение временных рядов
    sampling_rate = config['timeseries']['output_sampling_rate']
    t1, s1 = build_timeseries(graph1, duration, amp_range, sampling_rate)
    t2, s2 = build_timeseries(graph2, duration, amp_range, sampling_rate)
    
    # Используем общую временную ось (первого графика)
    # (можно также интерполировать второй на сетку первого)
    time_axis = t1
    
    # 6. Сохранение в CSV
    save_to_csv(time_axis, s1, s2, output_csv)
    import matplotlib.pyplot as plt
    plt.figure()
    plt.plot(time_axis, s1, label='Signal 1')
    plt.plot(time_axis, s2, label='Signal 2')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.legend()
    plt.savefig('data/output/signals_plot.png')
    plt.show()
    print("Обработка завершена.")
        # Визуализация результатов
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 6))
    plt.subplot(2,1,1)
    plt.plot(time_axis, s1, 'b-', linewidth=1)
    plt.title('Signal 1')
    plt.grid(True)
    plt.subplot(2,1,2)
    plt.plot(time_axis, s2, 'r-', linewidth=1)
    plt.title('Signal 2')
    plt.xlabel('Time (s)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/output/signals_plot.png')
    plt.show()

if __name__ == "__main__":
    main()