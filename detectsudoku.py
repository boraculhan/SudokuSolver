import cv2
import numpy as np

class DetectSudoku:
    """
    
    Detects the sudoku puzzle from an image, extracts the cells, and processes them.
    Usage: Create an instance with the image path and use the GetCells method to process and extract Sudoku cells.
    
    """
    def __init__(self, path):
        self.img = cv2.imread(path)

    def preprocess_image(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
        return thresh

    def find_sudoku_grid(self, binary_image):
        contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        max_area = 0
        best_cnt = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10:  # Filter out small contours
                if area > max_area:
                    max_area = area
                    best_cnt = cnt
        return best_cnt

    def perspective_transform(self, image, contour):
        epsilon = 0.05 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        if len(approx) == 4:
            pts1 = np.float32([approx[0][0], approx[1][0], approx[2][0], approx[3][0]])
            pts1 = pts1[np.argsort(pts1[:, 1])]
            if pts1[0][0] > pts1[1][0]:
                pts1[[0, 1]] = pts1[[1, 0]]
            if pts1[2][0] < pts1[3][0]:
                pts1[[2, 3]] = pts1[[3, 2]]
            side = 224*9
            pts2 = np.float32([[0, 0], [side-1, 0], [side-1, side-1], [0, side-1]])
            corresponding_pairs = []
            for i in range(4):
                corresponding_pairs.append((pts1[i], pts2[i]))
            A = np.zeros((8, 9))
            for i, pair in enumerate(corresponding_pairs):
                a, o = pair
                A[2*i] =   [-a[0], -a[1], -1, 0, 0, 0, a[0]*o[0], a[1]*o[0], o[0]]
                A[2*i+1] = [0, 0, 0, -a[0], -a[1], -1, a[0]*o[1], a[1]*o[1], o[1]]
            _, _, vt = np.linalg.svd(A)
            H = vt[-1].reshape((3,3)); H = H/H[-1, -1]
            dst = cv2.warpPerspective(image, H, (int(side), int(side)))
            return cv2.resize(dst, (28*9, 28*9)), H, (pts1, pts2)
        return None, None, (None, None)
    
    def split_into_cells(self, grid_image):
        cells = []
        size = grid_image.shape[0] // 9
        for i in range(9):
            row = []
            for j in range(9):
                cell = grid_image[i*size:(i+1)*size, j*size:(j+1)*size]
                row.append(cell)
            cells.append(row)
        return cells

    def remove_outer_ring(self, input_images_matrix):
        images_matrix = input_images_matrix.copy()
        ring_width = 5
        images_matrix[:, :, :ring_width, :] = 0
        images_matrix[:, :, -ring_width:, :] = 0
        images_matrix[:, :, :, :ring_width] = 0
        images_matrix[:, :, :, -ring_width:] = 0
        return images_matrix

    def apply_blur_to_cells(self, input_images_matrix):
        matrix = input_images_matrix.copy()
        for row in range(matrix.shape[0]):
            for col in range(matrix.shape[1]):
                matrix[row, col] = cv2.GaussianBlur(matrix[row, col], (3, 3), 0)
        return matrix

    def GetCells(self):
        processed_image = self.preprocess_image(self.img)
        contour = self.find_sudoku_grid(processed_image)
        if contour is not None:
            flat_grid_image, H, (pts1, pts2) = self.perspective_transform(self.img, contour)
            if flat_grid_image is not None: 
                gray = cv2.cvtColor(flat_grid_image, cv2.COLOR_BGR2GRAY)
                flat_grid_image_thresh = 255 - cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 10)
                cells_thresh = self.split_into_cells(flat_grid_image_thresh)
                cells_reshaped_thresh = np.array(cells_thresh)
                cells_reshaped_thresh_cropped = self.remove_outer_ring(cells_reshaped_thresh)
                cells_reshaped_thresh_cropped_blurred = self.apply_blur_to_cells(cells_reshaped_thresh_cropped)
                return cells_reshaped_thresh_cropped_blurred, H, self.img, self.img.shape[:2]
            else:
                return None, None, self.img, None
        else:
            return None, None, self.img, None
