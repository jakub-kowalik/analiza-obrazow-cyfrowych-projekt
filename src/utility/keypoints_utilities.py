from typing import List, Tuple

import cv2
import numpy as np


def generate_keypoints_dataframe():
    # @TODO refactor from notebook
    raise NotImplemented


def scale_points(size: Tuple, old_size: Tuple, points: List[Tuple]):
    old_y, old_x = old_size
    new_y, new_x = size

    ratio_x = new_x / old_x
    ratio_y = new_y / old_y

    output_list = []

    for point in points:
        p_y, p_x = point
        output_list.append((ratio_y * p_y, ratio_x * p_x))

    return output_list


def mark_points_on_image(image, points: List[Tuple], color: Tuple = (255, 255, 255), radius=5, thickness=5):
    copied_img = image.copy()

    for point in points:
        copied_img = cv2.circle(copied_img, (round(point[1]), round(point[0])), radius=radius, color=color,
                                thickness=thickness)

    return copied_img


def draw_skeleton(image, points: List[Tuple], color: Tuple = (255, 255, 255), radius=5, thickness=5):
    """
    Need to provide list of points according to current approach
    (0 - r ankle, 1 - r knee, 2 - r hip, 3 - l hip, 4 - l knee, 5 - l ankle, 6 - pelvis, 7 - thorax, 8 - upper neck, 9 - head top, 10 - r wrist, 11 - r elbow, 12 - r shoulder, 13 - l shoulder, 14 - l elbow, 15 - l wrist)
    :param image:
    :param points:
    :param color:
    :param radius:
    :param thickness:
    :return:
    """
    copied_img = image.copy()

    points = [(np.round(point[0]).astype(int), np.round(point[1]).astype(int)) for point in points]

    # left lower side
    copied_img = cv2.line(copied_img, (points[0][1], points[0][0]), (points[1][1], points[1][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[1][1], points[1][0]), (points[2][1], points[2][0]), color=color,
                          thickness=thickness)

    # right lower side
    copied_img = cv2.line(copied_img, (points[5][1], points[5][0]), (points[4][1], points[4][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[4][1], points[4][0]), (points[3][1], points[3][0]), color=color,
                          thickness=thickness)

    # pelvis
    copied_img = cv2.line(copied_img, (points[2][1], points[2][0]), (points[6][1], points[6][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[3][1], points[3][0]), (points[6][1], points[6][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[6][1], points[6][0]), (points[7][1], points[7][0]), color=color,
                          thickness=thickness)

    # thorax
    copied_img = cv2.line(copied_img, (points[7][1], points[7][0]), (points[8][1], points[8][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[7][1], points[7][0]), (points[12][1], points[12][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[7][1], points[7][0]), (points[13][1], points[13][0]), color=color,
                          thickness=thickness)

    # head
    copied_img = cv2.line(copied_img, (points[8][1], points[8][0]), (points[9][1], points[9][0]), color=color,
                          thickness=thickness)

    # left arm
    copied_img = cv2.line(copied_img, (points[13][1], points[13][0]), (points[14][1], points[14][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[14][1], points[14][0]), (points[15][1], points[15][0]), color=color,
                          thickness=thickness)

    # right arm
    copied_img = cv2.line(copied_img, (points[12][1], points[12][0]), (points[11][1], points[11][0]), color=color,
                          thickness=thickness)
    copied_img = cv2.line(copied_img, (points[11][1], points[11][0]), (points[10][1], points[10][0]), color=color,
                          thickness=thickness)

    for point in points:
        copied_img = cv2.circle(copied_img, (round(point[1]), round(point[0])), radius=radius, color=color,
                                thickness=thickness)

    return copied_img
