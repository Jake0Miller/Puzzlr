import cv2
import numpy as np

def board_corners(board):
    largest_contour = get_largest_contour(board)
    quadrangle = largest_contour
    evals = np.zeros([3, len(quadrangle)])
    for index, coordinates in enumerate(quadrangle):
        coordinates = coordinates[0]
        evals[0, index] = sum(coordinates)
        evals[1, index] = coordinates[0] - coordinates[1]
        evals[2, index] = coordinates[1] - coordinates[0]
    top_left = quadrangle[np.argmin(evals, axis=1)[0]][0]
    bot_right = quadrangle[np.argmax(evals, axis=1)[0]][0]
    top_right = quadrangle[np.argmax(evals, axis=1)[1]][0]
    bot_left = quadrangle[np.argmax(evals, axis=1)[2]][0]
    return np.array([top_left, top_right, bot_right, bot_left])

def crop_reshape(board, corners):
    destination = get_dest(corners)
    side_length = int(np.ceil(get_dist(destination[0], destination[1])))

    source = np.array(corners, dtype='float32')
    transformation_matrix = cv2.getPerspectiveTransform(source, destination)

    return cv2.warpPerspective(board, transformation_matrix, (side_length, side_length))

def get_dist(p1, p2):
    diff = np.array([p2[i] - p1[i] for i in range(len(p1))])
    return np.sqrt(sum(diff ** 2))

def get_dest(corners):
    top_left, top_right, bot_right, bot_left = [i for i in corners]
    smallest_side = min([
        get_dist(top_left, top_right),
        get_dist(top_right, bot_right),
        get_dist(bot_right, bot_left),
        get_dist(bot_left, top_left)
    ])
    return np.array([[0,0], [smallest_side, 0], [smallest_side, smallest_side], [0, smallest_side]], dtype='float32')

def get_largest_contour(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[0]
