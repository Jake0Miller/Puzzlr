import numpy as np
from skimage.transform import resize

class PreprocessNumber:
    def __init__(self, image):
        self.NOT_BLACK_THRESHOLD = 20
        self.image = image
        self.extracted_feature = np.zeros_like(self.image)
        self.dimension = self.image.shape
        self.dct_groups = {}

    def crop_feature(self):
        self.group_features()
        self.extract_feature(self.group_number())

    def is_number(self):
        return self.validate_first_condition() and self.validate_second_condition()

    def get_centered_number(self, dimensions):
        left = self.extracted_feature.shape[0]
        right = 0
        top = self.extracted_feature.shape[1]
        bot = 0

        for row in range(self.extracted_feature.shape[0]):
            for col in range(self.extracted_feature.shape[1]):
                if self.extracted_feature[row][col] > self.NOT_BLACK_THRESHOLD:
                    if row < top:
                        top = row
                    elif row > bot:
                        bot = row
                    if col < left:
                        left = col
                    elif col > right:
                        right = col
        width = right - left + 1
        height = bot - top + 1

        extracted_number = np.zeros((height, width))

        for row in range(height):
            for col in range(width):
                extracted_number[row][col] = self.extracted_feature[top + row][left + col]

        if height != width:
            square_num = np.zeros((max((height, width)), max(height, width)))

            if height > width:
                diff = int((height - width) / 2)
                square_num[:, diff:diff + width] = extracted_number
            else:
                diff = int((width - height) / 2)

        else:
            square_num = extracted_number

        centered_number = np.zeros(dimensions)

        if np.any(np.array(square_num.shape) > 20):
            centered_number[4:24, 4:24] = resize(square_num, (20, 20), anti_aliasing=True)
        else:
            size = square_num.shape
            diff_height = int((dimension[0] - size[0]) / 2)
            diff_width = int((dimension[0] - size[1]) / 2)
            centered_number[diff_height:size[0] + diff_height, diff_width:size[1] + diff_width] = square_num
        return centered_number

    def group_features(self):
        new_group = 0
        for row in range(self.dimension[0]):
            for col in range(self.dimension[1]):
                if self.image[row][col] > self.NOT_BLACK_THRESHOLD:
                    surrounding_group = self.check_top_left_pixel(row, col)
                    if surrounding_group != -1:
                        self.dct_groups[str(row) + ":" + str(col)] = surrounding_group
                    else:
                        self.dct_groups[str(row) + ":" + str(col)] = new_group
                        new_group += 1
                else:
                    self.dct_groups[str(row) + ":" + str(col)] = -1

    def check_top_left_pixel(self, row, col):
        group_pixel_left = -1
        if col > 0:
            group_pixel_left = self.dct_groups[str(row) + ":" + str(col - 1)]
        if row > 0:
            group_pixel_above = self.dct_groups[str(row - 1) + ":" + str(col)]
            if group_pixel_left == -1:
                return group_pixel_above
            elif group_pixel_above == -1:
                return group_pixel_left
        else:
            return group_pixel_left

        if group_pixel_left != group_pixel_above:
            for item in self.dct_groups:
                if self.dct_groups[item] == group_pixel_above:
                    self.dct_groups[item] = group_pixel_left
        return group_pixel_left

    def group_number(self):
        center_row = int(self.dimension[0] / 2)
        center_col = int(self.dimension[1] / 2)
        nr_search_loops = min(self.dimension[0] - center_row, self.dimension[1] - center_row) - 1
        loop_directions = [[0, 1], [1, 0], [0, -1], [-1, 0]]

        if self.image[center_row][center_col] > self.NOT_BLACK_THRESHOLD:
            return self.dct_groups[str(center_row) + ":" + str(center_col)]
        else:
            for loop_nr in range(1, nr_search_loops):
                center_row -= 1
                center_col -= 1
                search_coord = [center_row, center_col]
                for direction in loop_directions:
                    vertical = direction[0]
                    horizontal = direction[1]
                    for i in range(loop_nr * 2):
                        if self.image[search_coord[0]][search_coord[1]] > self.NOT_BLACK_THRESHOLD:
                            return self.dct_groups[str(search_coord[0]) + ":" + str(search_coord[1])]
                        search_coord[0] += 1 * vertical
                        search_coord[1] += 1 * horizontal
        return -1

    def extract_feature(self, feature_group_number):
        for row in range(self.dimension[0]):
            for col in range(self.dimension[1]):
                if self.dct_groups[str(row) + ":" + str(col)] == feature_group_number:
                    self.extracted_feature[row][col] = self.image[row][col]

    def validate_first_condition(self):
        dot_count = 0
        x_box = int(self.extracted_feature.shape[0] / 9) * 4
        y_box = int(self.extracted_feature.shape[1] / 9) * 4
        for row in range(x_box, x_box + int(self.extracted_feature.shape[0] / 9) * 2):
            for col in range(y_box, y_box + int(self.extracted_feature.shape[1] / 9) * 2):
                if self.extracted_feature[row][col] > self.NOT_BLACK_THRESHOLD:
                    dot_count += 1
                    if dot_count >= 5: return True
        return False

    def validate_second_condition(self):
        count = 0
        for row in self.extracted_feature:
            for pixel in row:
                if pixel > self.NOT_BLACK_THRESHOLD:
                    count += 1
                    if count >= 25: return True
        return False
