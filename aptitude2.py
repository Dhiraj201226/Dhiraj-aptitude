import cv2 as cv
import numpy as np

def resize_scale(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)

def is_filled_white(contour, gray_img, min_mean=200):
    mask = np.zeros(gray_img.shape, dtype=np.uint8)
    cv.drawContours(mask, [contour], -1, 255, -1)
    inside = gray_img[mask == 255]
    if len(inside) == 0:
        return False
    mean_val = np.mean(inside)
    return mean_val > min_mean

def get_column(x):
    if x < 150:
        return 'A'
    elif x < 250:
        return 'B'
    elif x < 400:
        return 'C'
    else:
        return 'D'

def get_row(y):
    if y < 100:
        return '1'
    elif y < 150:
        return '2'
    elif y < 190:
        return '3'
    elif y < 230:
        return '4'
    elif y < 270:
        return '5'
    elif y < 300:
        return '6'
    elif y < 350:
        return '7'
    elif y < 380:
        return '8'
    elif y < 400:
        return '9'
    else:
        return '10'


img1 = cv.imread('Bhaskar (1).jpg')
img2 = cv.imread('Ganshyam (1).jpg')
img3 = cv.imread('Raghav (1).jpg')

gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
gray3 = cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

gray1 = cv.resize(gray1, (500, 500))
gray2 = cv.resize(gray2, (500, 500))
gray3 = cv.resize(gray3, (500, 500))

bright1 = cv.convertScaleAbs(gray1, alpha=2.5, beta=50)
bright3 = cv.convertScaleAbs(gray3, alpha=1.0, beta=50)

merged = cv.merge([bright1, gray2, bright3])

final_img = cv.cvtColor(merged, cv.COLOR_BGR2GRAY)

blurred = cv.GaussianBlur(final_img, (5, 5), 1)
_, thresh = cv.threshold(blurred, 28, 255, cv.THRESH_BINARY)
cv.imshow("Threshold", thresh)

contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
output = cv.cvtColor(final_img.copy(), cv.COLOR_GRAY2BGR)

for cnt in contours:
    x, y, w, h = cv.boundingRect(cnt)
    if x > 50 and y > 50:
        area = cv.contourArea(cnt)
        if area > 300 and is_filled_white(cnt, thresh):
            cx, cy = x + w // 2, y + h // 2
            col = get_column(cx)
            row = get_row(cy)
            print(f"{row}{col}")
            cv.drawContours(thresh, [cnt], -1, (0, 255, 0), 2)
            cv.putText(output, f"{row}{col}", (x, y - 5),
                       cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)

cv.imshow("Detected Filled Shapes", output)

cv.waitKey(0)
cv.destroyAllWindows()