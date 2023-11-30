import numpy as np
import math
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
class Generate_Triangles:
    def __init__(self,image_height: int, image_width):
        self._image_height = image_height
        self._image_width = image_width
    def _generate_image(self)-> np.ndarray[(np.short)]:
        new_image = Image.new("L", (self._image_width, self._image_height), "white")
        draw = ImageDraw.Draw(new_image)
        while True:
            points_in_space = np.random.randint(low=0, high=self._image_height, size=(3, 2))
            x1, x2, x3 = points_in_space[:, 0]
            y1, y2, y3 = points_in_space[:, 1]

            m1 = (y2 - y1) / (x2 - x1)
            m2 = (y3 - y1) / (x3 - x1)
            m3 = (y3 - y2) / (x3 - x2)

            A1 = math.degrees(math.atan(abs((m2 - m1) / (1 + m1 * m2))))
            A2 = math.degrees(math.atan(abs((m3 - m1) / (1 + m1 * m3))))
            A3 = math.degrees(math.atan(abs((m3 - m2) / (1 + m2 * m3))))

            # Check if the angles add up to 180 degrees
            if math.isclose(A1 + A2 + A3, 180, abs_tol=1e-6):
                break  # Exit the loop when valid points are found
        draw.line([(x1,y1), (x2,y2)], fill=0, width=1)  # Draw outline
        draw.line([(x2,y2), (x3,y3)], fill=0, width=1)
        draw.line([(x3,y3), (x1,y1)], fill=0, width=1)
        draw.polygon([(x1, y1), (x2, y2), (x3, y3)], outline=0, fill=0)
        image_array = np.array(new_image)
        return image_array

    def generate_dataset(self, n_triangles : int) -> np.ndarray[(np.short)]:
        dataset = np.zeros([n_triangles,self._image_height,self._image_width], dtype=np.short)
        for idx in range (n_triangles):
            dataset[idx,:,:]= self._generate_image()
        return dataset
    
        
