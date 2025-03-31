import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import re

def extract_values(filename):
    pattern = r'tile_(\d+)-(\d+) x (\d+)-(\d+)\.tif'
    
    match = re.match(pattern, filename)
    
    if match:
        x_start = int(match.group(1))
        y_start = int(match.group(2))
        x_end = int(match.group(3))
        y_end = int(match.group(4))
        
        return x_start, y_start, x_end, y_end
    else:
        raise ValueError("Vlad pidor")

def build_rgb_histogram(image_path):
    img = Image.open(image_path)
    
    pixels = np.array(img)
    
    height, width, channels = pixels.shape
    
    red_values = []
    green_values = []
    blue_values = []
    
    for i in range(height):
        for j in range(width):
            r, g, b = pixels[i][j]
            red_values.append(r)
            green_values.append(g)
            blue_values.append(b)
            
    return red_values, green_values, blue_values

if __name__ == "__main__":
    folder_path = r'templates/Tree Detection.v5i.yolov8/train/images'
    print(folder_path)
    
    image_files = [f for f in os.listdir(folder_path) if f.endswith(('jpg', 'tif'))]
    image_files.sort()
    print(image_files)

    show_hist = False
    hist_in_file = False
    hist_params = True

    for idx, file_name in enumerate(image_files):
        print(f'saved {file_name}')
        full_file_path = os.path.join(folder_path, file_name)
            
        red, green, blue = build_rgb_histogram(full_file_path)

        if hist_params:
            coords = extract_values(file_name)
            mean_red = np.mean(red)
            mean_green = np.mean(green)
            mean_blue = np.mean(blue)

            std_red = np.std(red)
            std_green = np.std(green)
            std_blue = np.std(blue)

            with open(os.path.join(folder_path, 'params.txt'), 'a') as file:
                combined_data = list(coords)
                combined_data.extend([mean_red, mean_green, mean_blue, std_red, std_green, std_blue])
                
                data_string = ' '.join(map(str, combined_data)) + '\n'
                
                file.write(data_string)
        
        if show_hist or hist_in_file: 
            fig, ax = plt.subplots(figsize=(10,6), constrained_layout=True)

            ax.hist(red, bins=256, range=(0, 255), color='r', alpha=0.4, label=f'{file_name} Red')
            ax.hist(green, bins=256, range=(0, 255), color='g', alpha=0.4, label=f'{file_name} Green')
            ax.hist(blue, bins=256, range=(0, 255), color='b', alpha=0.4, label=f'{file_name} Blue')
            if show_hist:
                plt.show()
            if hist_in_file:
                plt.savefig(full_file_path + '.png')
            plt.close(fig)
