## Try to map the solution to the original image.
import numpy as np
import cv2

class MapBack():
    def __init__(self, sudoku, sol, shape, H, img):
        self.sudoku = sudoku
        self.sol = sol
        self.shape = shape
        self.H = H
        self.arr = np.zeros((9, 9), dtype = int)
        self.mask = self.sudoku == 0
        self.arr[self.mask] = self.sol[self.mask]
        self.img = img

    def SolMap(self):
        # Parameters for the image
        block_size = 224
        image_size = block_size * self.arr.shape[0]

        # Font settings for OpenCV
        font = cv2.FONT_HERSHEY_SIMPLEX
        # Adjusted code with smaller font size and BGR image format
        font_scale = 6  # Reduced font scale for better fitting
        text_color = (83, 0, 127)  # Color of our choice, BGR format!
        font_thickness = 15
        
        # Create a black image in BGR format
        final_image = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Update the text rendering with new settings
        for i in range(self.arr.shape[0]):
            for j in range(self.arr.shape[1]):
                if self.arr[i, j] != 0:
                    text = str(int(self.arr[i, j]))
                    # Calculate text size to center it in the block
                    text_size = cv2.getTextSize(text, font, font_scale, font_thickness)[0]
                    bottom_left_corner_x = j * block_size + (block_size - text_size[0]) // 2
                    bottom_left_corner_y = i * block_size + (block_size + text_size[1]) // 2
        
                    # Place the text on the image
                    cv2.putText(final_image, text, (bottom_left_corner_x, bottom_left_corner_y), 
                                font, font_scale, text_color, font_thickness, lineType=cv2.LINE_AA)
                    
        self.impose_this = cv2.warpPerspective(final_image, self.H, (self.shape[1], self.shape[0]), flags=cv2.WARP_INVERSE_MAP)
        
    def ImposeSol(self):
        save_this = np.where(np.any(self.impose_this != 0, axis=-1, keepdims=True), self.impose_this, self.img)
        return save_this