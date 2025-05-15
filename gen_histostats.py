import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import json

def zero_cost(data):
    return data[data != 0]

def ch_size(data, size):
    current_mean = np.mean(data)
    scale_factor = size / current_mean
    scaled_arr = data * scale_factor
    return scaled_arr.astype(int)

def calculate_statistics(channel):
    mean = np.mean(channel)
    median = np.median(channel)
    std_dev = np.std(channel)
    mode_result = stats.mode(channel, axis=None, keepdims=False)
    mode = mode_result.mode if isinstance(mode_result.mode, (int, float, np.integer)) else mode_result.mode[0]
    return mean, median, std_dev, mode

# Директории
os.makedirs('dataset/train/images', exist_ok=True)
os.makedirs('real_im_hsm', exist_ok=True)
os.makedirs('stats', exist_ok=True)
os.makedirs('histograms', exist_ok=True)

# Сбор статистик
r_means, g_means, b_means = [], [], []

# Перебор всех изображений
for filename in os.listdir('dataset/train/images'):
    if not filename.lower().endswith(('.jpg', '.png', '.jpeg')):
        continue

    img_path = os.path.join('dataset/train/images', filename)
    image = cv2.imread(img_path)

    if image is None:
        print(f"Ошибка загрузки изображения {filename}")
        continue

    print(f"Обработка: {filename}")

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    r_channel = image[:, :, 0]
    g_channel = image[:, :, 1]
    b_channel = image[:, :, 2]

    r_data = zero_cost(r_channel.ravel())
    g_data = zero_cost(g_channel.ravel())
    b_data = zero_cost(b_channel.ravel())

    r_scaled = ch_size(r_data, 99.82282989220326)
    g_scaled = ch_size(g_data, 124.5)
    b_scaled = ch_size(b_data, 41.7)

    # Статистика
    r_stats = calculate_statistics(r_data)
    g_stats = calculate_statistics(g_data)
    b_stats = calculate_statistics(b_data)

    r_means.append(r_stats[0])
    g_means.append(g_stats[0])
    b_means.append(b_stats[0])

    # Сохранение статистики
    stat_path = os.path.join('stats', f"{os.path.splitext(filename)[0]}_stats.txt")
    with open(stat_path, 'w') as f:
        f.write(f"Файл: {filename}\n")
        f.write(f"Красный канал: Среднее = {r_stats[0]:.2f}, Медиана = {r_stats[1]:.2f}, "
                f"СКО = {r_stats[2]:.2f}, Мода = {r_stats[3]}\n")
        f.write(f"Зелёный канал: Среднее = {g_stats[0]:.2f}, Медиана = {g_stats[1]:.2f}, "
                f"СКО = {g_stats[2]:.2f}, Мода = {g_stats[3]}\n")
        f.write(f"Синий канал: Среднее = {b_stats[0]:.2f}, Медиана = {b_stats[1]:.2f}, "
                f"СКО = {b_stats[2]:.2f}, Мода = {b_stats[3]}\n")

    # Гистограмма
    plt.figure(figsize=(10, 6))
    plt.hist(r_data, bins=256, alpha=0.5, color='red', label='Red')
    plt.hist(g_data, bins=256, alpha=0.5, color='green', label='Green')
    plt.hist(b_data, bins=256, alpha=0.5, color='blue', label='Blue')
    plt.title(f'Гистограмма RGB: {filename}')
    plt.xlim([0, 256])
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join('histograms', f"{os.path.splitext(filename)[0]}_hist.jpg")
    plt.savefig(hist_path, dpi=300)
    plt.close()

# Подсчёт и сохранение средних значений
if r_means:
    avg_r = float(np.mean(r_means))
    avg_g = float(np.mean(g_means))
    avg_b = float(np.mean(b_means))

    stats_dict = {
        "average_r": avg_r,
        "average_g": avg_g,
        "average_b": avg_b
    }

    with open(os.path.join('stats', 'average_stats.json'), 'w') as json_file:
        json.dump(stats_dict, json_file, indent=4)

    print(f"\nОбщая статистика сохранена в stats/average_stats.json:")
    print(stats_dict)
