import cv2
import numpy as np
import os


# Region of Interest Mask


def region_of_interest(img):
    height, width = img.shape[:2]
    mask = np.zeros_like(img)

    polygon = np.array(
        [
            [
                (50, height),
                (width - 50, height),
                (width - 200, int(height * 0.55)),
                (200, int(height * 0.55)),
            ]
        ],
        np.int32,
    )

    cv2.fillPoly(mask, polygon, 255)
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image


# Averaging slope intercept


def average_slope_intercept(lines):
    left, right = [], []
    if lines is None:
        return None, None
    for line in lines:
        x1, y1, x2, y2 = line.reshape(4)
        if x2 - x1 == 0:
            continue
        slope, intercept = np.polyfit((x1, x2), (y1, y2), 1)

        # filtering out nearly flat lines
        if abs(slope) < 0.3:
            continue

        if slope < 0:
            left.append((slope, intercept))
        else:
            right.append((slope, intercept))

    left_avg = np.average(left, axis=0) if len(left) > 0 else None
    right_avg = np.average(right, axis=0) if len(right) > 0 else None
    return left_avg, right_avg


def make_points(y1, y2, line_params):
    if line_params is None:
        return None
    slope, intercept = line_params
    if slope == 0:
        return None
    x1 = int((y1 - intercept) / slope)
    x2 = int((y2 - intercept) / slope)
    return np.array([x1, y1, x2, y2])


def draw_lanes(img, lines):
    left, right = average_slope_intercept(lines)
    y1 = img.shape[0]
    y2 = int(y1 * 0.6)
    overlay = img.copy()

    drawn = False
    if left is not None:
        left_line = make_points(y1, y2, left)
        if left_line is not None:
            cv2.line(
                overlay,
                (left_line[0], left_line[1]),
                (left_line[2], left_line[3]),
                (0, 255, 0),
                10,
            )
            drawn = True
    if right is not None:
        right_line = make_points(y1, y2, right)
        if right_line is not None:
            cv2.line(
                overlay,
                (right_line[0], right_line[1]),
                (right_line[2], right_line[3]),
                (0, 255, 0),
                10,
            )
            drawn = True

    if not drawn and lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)

    return overlay


# Lane Detection Pipeline


def process_image(image):

    # grayscale + blur
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # edges
    edges = cv2.Canny(blur, 40, 120)

    # mask ROI
    roi_edges = region_of_interest(edges)

    # hough transform
    lines = cv2.HoughLinesP(
        roi_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=30,
        minLineLength=40,
        maxLineGap=100,
    )

    # drawing lanes

    lane_img = draw_lanes(image, lines)
    return lane_img


if __name__ == "__main__":
    input_dir = "Dataset"
    output_dir = "Output"
    os.makedirs(output_dir, exist_ok=True)

    for root, _, files in os.walk(input_dir):
        for file in files:
            if file.lower().endswith((".jpg", ".png", ".jpeg")):
                img_path = os.path.join(root, file)
                img = cv2.imread(img_path)
                if img is None:
                    continue

                lane_img = process_image(img)

                out_path = os.path.join(output_dir, file)
                cv2.imwrite(out_path, lane_img)

    print(f" Lane detection completed. Results saved in '{output_dir}'")
