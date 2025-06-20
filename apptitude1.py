import cv2 as cv
import numpy as np

def resize_scale(frame, scale):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    return cv.resize(frame, (width, height), interpolation=cv.INTER_AREA)
def get_edge_count(contour):
    perimeter = 0.01*cv.arcLength(contour, True)
    approx = cv.approxPolyDP(contour,  perimeter, True)
    return len(approx)

img1 = cv.imread("Ganshyam.jpg")
new_img1=cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

img2 = cv.imread("Raghav.jpg")
new_img2=cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

img3 = cv.imread("Bhaskar.jpg")
new_img3=cv.cvtColor(img3, cv.COLOR_BGR2GRAY)

gray1 = resize_scale(new_img1, 0.5)
gray2 = resize_scale(new_img2, 0.5)
gray3 = resize_scale(new_img3, 0.5)

merged = cv.merge([gray1, gray2, gray3])
merged = np.uint8(cv.normalize(merged, None, 0, 255, cv.NORM_MINMAX))
merged1 = resize_scale(merged, 0.5)

gray = cv.cvtColor(merged1, cv.COLOR_BGR2GRAY)
blurred = cv.GaussianBlur(gray, (5, 5), 1)

bright = cv.convertScaleAbs(blurred, alpha=1.8, beta=40)

_, thresh = cv.threshold(bright, 60, 255, cv.THRESH_BINARY)

contours, _ = cv.findContours(thresh, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)

def get_column(x):
    if x < 100:
        return 'A'
    elif x < 200:
        return 'B'
    elif x < 300:
        return 'C'
    else:
        return 'D'

def get_row(y):
    if y<100:
        return '1'
    elif y<160:
        return '2'
    elif y<220:
        return '3'
    elif y<260:
        return '4'
    elif y<300:
        return '5'
    elif y<360:
        return '6'
    elif y<400:
        return '7'
    else:
        return '8'

output = bright.copy()

for cnt in contours:
    edges=get_edge_count(cnt)
    if edges>10:
     area = cv.contourArea(cnt)
     if 500 < area < 2500:
        perimeter = cv.arcLength(cnt, True)
        circularity = 4 * np.pi * (area / (perimeter ** 2 ))
        if circularity > 0.7:
            (x, y), radius = cv.minEnclosingCircle(cnt)
            center = (int(x), int(y))
            radius = int(radius)

            cv.circle(output, center, radius, (255, 0, 255), 2)

            col = get_column(x)
            row = get_row(y)
            print(f"{row}{col}")
            

cv.imshow("Detected Circles", output)
cv.waitKey(0)
cv.destroyAllWindows()