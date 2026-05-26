import pandas as pd
import os

def save_to_csv(time, signal1, signal2, output_path):
    """
    Сохраняет два временных ряда в один CSV-файл.
    time – массив времен (общий для обоих сигналов)
    signal1, signal2 – массивы значений
    """
    # Убедимся, что длины совпадают
    min_len = min(len(time), len(signal1), len(signal2))
    df = pd.DataFrame({
        'time_s': time[:min_len],
        'signal1': signal1[:min_len],
        'signal2': signal2[:min_len]
    })
    # Создаём папку, если её нет
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, sep=';', decimal='.')
    print(f"CSV сохранён: {output_path}")