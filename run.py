import os
import sys
import tensorflow as tf
import cv2
import numpy as np
from solver import SolveSudoku
from detectsudoku import DetectSudoku
from mapback import MapBack

## Some code to silence tensorflow messages.
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

text = """
Please follow the instructions in README.md carefully!
"""
dirs = os.listdir()
if not ('BMBnet_DIGITS' in dirs and 'puzzles' in dirs and 'solver.py' in dirs and 'detectsudoku.py' in dirs and 'mapback.py' in dirs):
    print(text)
    sys.exit()

# Creates 'solved' directory if it doesn't exist.
path = os.getcwd()
os.makedirs(path+'\solved', exist_ok=True)

# Make sure to operate on images only if there are other file formats in the directory.
puzzle_list = os.listdir(path+'\puzzles')
puzzle_list = [file for file in puzzle_list if file.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.webp'))]  # Filter only image files

model = tf.keras.models.load_model('BMBnet_DIGITS')

# Main loop of the code. For each puzzle in puzzles folder:
#   1. Detect the sudoku cells.
#   2. Using BMBnet_DIGITS, predict the numbers in those cells.
#   3. With IP (Integer Programming) solve the sudoku puzzle.
#   4. Map the solution back to the original image and save it to '\solved' directory.

for puzzle in puzzle_list:
    print(f'Attempting to solve:\t{puzzle}:')
    detect = DetectSudoku(path+f'\puzzles\{puzzle}')
    cells, H, img, shape = detect.GetCells() 
    if cells is None:
        print('Could not detect the cells.\n')
        continue
    
    input_data = list()
    for i in range(9):
        for j in range(9):
            aux = cells[i][j]
            aux = np.expand_dims(aux, axis=-1); aux = np.expand_dims(aux, axis=0)
            if cv2.countNonZero(aux) <= 70:
                pass
            else:
                predict = np.argmax(model.predict(aux, verbose = None))+1
                input_data.append((predict, i+1, j+1))

    sudoku = SolveSudoku(input_data)
    sudoku.BuildConstraints()
    status, solved = sudoku.SolveLP(); solved = np.array(solved, dtype = int)

    
    sudoku_matrix = np.zeros((9, 9), dtype=int)
    
    # Fill in the matrix using the data from the list of tuples
    for number, row, column in input_data:
        sudoku_matrix[row-1, column-1] = number  # Adjust for 0-based indexing
    
    if status == 'Infeasible':
        # Some cells are incorrectly classified so the IP couldn't solve it. 
        print('Cannot solve sudoku.\n')
        continue
        
    
    ## Map the solution back to the original image
    mapBack = MapBack(sudoku_matrix, solved, shape, H, img)
    mapBack.SolMap()
    save_this = mapBack.ImposeSol()
    
    cv2.imwrite(path+f'\solved\solved_{puzzle}', save_this) 
    
    print('Solution is saved in "\solved" directory.\n')