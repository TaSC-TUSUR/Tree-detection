import cv2
import numpy as np

def adjust_brightness(image, adjustments):
    """
    vlad pidor
    """
    img = image.astype(np.float32)
    
    b, g, r = cv2.split(img)
    
    b *= adjustments[0]
    g *= adjustments[1]
    r *= adjustments[2]
    
    adjusted_img = cv2.merge((b, g, r))
    
    adjusted_img = np.clip(adjusted_img, 0, 255).astype(np.uint8)
    
    return adjusted_img


img_path = r'D:\hist\77_G.jpg' 
output_path = r'D:\hist\77_G_out.jpg'
image = cv2.imread(img_path)

adjustments = [ 29.77328303850156 / 62.27501826150475,  138.3551638917794 / 142.65850986121256 , 100.10009105098855 / 140.7972972972973]

result = adjust_brightness(image, adjustments)

cv2.imwrite(output_path, result)

#cv2.imshow('Original Image', image)
#cv2.imshow('Adjusted Image', result)
#cv2.waitKey(0)
