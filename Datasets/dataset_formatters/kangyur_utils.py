import cv2
import numpy as np


def get_bbox_from_contour(points):
    left = 999999
    right = -999999
    top = 999999
    bottom = -999999

    for x, y in points:
        left = min(left, x)
        right = max(right, x)
        top = min(top, y)
        bottom = max(bottom, y)
    x = left
    y = top
    w = right - left
    h = bottom - top
    return [x, y, w, h]

def get_bbox_from_contours(contours):
    left = 999999
    right = -999999
    top = 999999
    bottom = -999999
    for contour in contours:
        x, y, w, h = get_bbox_from_contour(contour)
        left = min(left, x)
        right = max(right, x + w)
        top = min(top, y)
        bottom = max(bottom, y + h)
    x = left
    y = top
    w = right - left
    h = bottom - top
    return [x, y, w, h]

def get_contours_from_line_seps(line_seps):
    assert len(line_seps) >= 2
    contours = []
    for i in range(1, len(line_seps)):
        contour = line_seps[i-1] + list(reversed(line_seps[i]))
        contours.append(contour)
    return contours




def cut_image_from_contour(img, points):
    points = [[int(x), int(y)] for x, y in points]
    points = np.array(points)
    """通过contour分割出文本行图像"""
    mask = np.zeros_like(img)
    fill_color = (1, 1, 1) if len(mask.shape) == 3 else 1
    cv2.fillPoly(mask, pts=[points], color=fill_color)
    inverse_mask = np.ones_like(img) - mask
    # 白色背景
    canvas = np.full_like(img, 255)
    masked = img * mask + canvas * inverse_mask
    x, y, w, h = get_bbox_from_contour(points)
    if len(img.shape) == 2:
        line_img = masked[y:y + h, x:x + w]
    else:
        line_img = masked[y:y + h, x:x + w, :]
    return line_img