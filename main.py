import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

def lane_region(img):
    height = img.shape[0]
    region = np.array([[(150, height), (600, height), (300, 200)]])
    black = np.zeros_like(img)
    cv.fillPoly(black, region, 255)
    region_lane = cv.bitwise_and(img, black)
    return region_lane

def display_line(img, lines):
    lane_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(lane_image, (x1, y1), (x2, y2),
                    (0, 255, 0), 2)
    return lane_image

img = cv.imread('road.jpg')
img = cv.resize(img, (700, 500))
copy_img = np.copy(img)
gray_img = cv.cvtColor(copy_img, cv.COLOR_BGR2GRAY)
blur = cv.GaussianBlur(gray_img, (5,5), 0)
canny = cv.Canny(blur, 50, 150)
#plt.imshow(canny)
#plt.show()
lane = lane_region(canny)
lines = cv.HoughLinesP(lane, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=5)
display_line_image = display_line(copy_img, lines)
img = cv.bitwise_or(img, display_line_image)



cv.imshow('image', img)
cv.waitKey(0)

cap = cv.VideoCapture('test2.mp4')
while True:
    _, frame = cap.read()
    canny = cv.Canny(frame, 50, 150)
    lane = lane_region(canny)

    lines = cv.HoughLinesP(lane, 2, np.pi / 180, 100, np.array([]), minLineLength=20, maxLineGap=5)
    display_line_image = display_line(frame, lines)

    frame = cv.bitwise_or(frame, display_line_image)

    cv.imshow('frame', frame)
    if cv.waitKey(0) == 27:
        break

cap.release()
cv.destroyAllWindows()