import cv2
import numpy as np

def zero_cost(data):
    return data[data != 0]

def ch_size(original_data, filtered_data, target_mean):
    scale_factor = target_mean / np.mean(filtered_data)
    scaled = original_data * scale_factor
    return np.clip(scaled, 0, 255).astype(np.uint8)

# Загрузка изображения
image = cv2.imread('test_3.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Исходные размеры изображения
height, width = image.shape[:2]

# Обработка каналов
r_channel = image[:, :, 0]
g_channel = image[:, :, 1]
b_channel = image[:, :, 2]

# Сохраняем оригинальные данные перед изменением
r_data = r_channel.ravel().copy()
g_data = g_channel.ravel().copy()
b_data = b_channel.ravel().copy()

# Фильтрация нулей
new_r = zero_cost(r_data)
new_g = zero_cost(g_data)
new_b = zero_cost(b_data)

r_scale = 101
g_scale = 124.5
b_scale = 41.7

# Масштабирование с сохранением исходной формы
r_scaled = ch_size(r_data, new_r, r_scale).reshape(height, width)
g_scaled = ch_size(g_data, new_g, g_scale).reshape(height, width)
b_scaled = ch_size(b_data, new_b, b_scale).reshape(height, width)

# Сборка изображения
merged = cv2.merge([b_scaled, g_scaled, r_scaled])
cv2.imwrite("new_test_3.jpg", merged)