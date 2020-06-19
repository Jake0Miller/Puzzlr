import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from skimage.io import imread
import sys
sys.path.insert(0, '.')
import Board
import pdb

MNIST_DIMENSION = (28, 28)

def read_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # plt.imshow(image, cmap='gray')
    # plt.show()

    resized_img = cv2.resize(image, (1200, 900), interpolation=cv2.INTER_AREA)
    proc_image = Board.process_board(resized_img.copy(), MNIST_DIMENSION)

    sudoku_board = np.zeros((9, 9), dtype=np.int8)

    model = load_model(os.path.dirname(os.path.realpath(__file__)) + "\Final model")
    pdb.set_trace()

if __name__ == "__main__":
    read_image(image)
