import cv2
import numpy as np

import Grid
import Number

def process_board(image, dimensions):
    corners = Grid.board_corners(preprocess_image(image.copy(), 3))
    cropped_grid = Grid.crop_reshape(preprocess_image(image.copy(), 1), corners)

    expand_ratio = 1.05
    squares = expanded_squares(cropped_grid.copy(), expand_ratio)

    centered_numbers = []
    for square in squares:
        number = Number.PreprocessNumber(square.copy())
        number.crop_feature()
        if number.is_number():
            centered_number = number.get_centered_number(dimensions)
        else:
            centered_number = np.zeros(dimensions)
        centered_numbers.append(centered_number)

    return centered_numbers

def preprocess_image(img, line_thickness=3):
    img = cv2.GaussianBlur(img.copy(), (9, 9), 0)
    threshold = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 4)

    inverted = cv2.bitwise_not(threshold, threshold)

    kernel = np.ones((line_thickness, line_thickness), np.uint8)
    return cv2.dilate(inverted, kernel)

def expanded_squares(img, expand_ratio):
    n_rows = 9
    expanded_squares = []
    width = int(np.floor(min(img.shape) / n_rows))
    height = width

    for row in range(1, 1 + n_rows):
        for col in range(1, 1 + n_rows):
            h_top = h_bot = w_left = w_right = 0
            if row != 1 and row != 9:
                h_top = int(height / expand_ratio) - height
            if row != 9:
                h_bot = int(height * expand_ratio) - height
            if col != 1:
                w_left = int(width / expand_ratio) - width
            if col != 9:
                w_right = int(width * expand_ratio) - width
            from_row = (row - 1) * height + h_top
            to_row = row * height + h_bot

            if row == 9:
                to_row = img.shape[0]
            expanded_square = np.zeros((int((to_row - from_row)), width + w_right - w_left))

            for index, k in enumerate(range(from_row, to_row)):
                expanded_square[index] = img[k][(col - 1) * width + w_left:col * width + w_right]
            expanded_squares.append(expanded_square)

    return expanded_squares
