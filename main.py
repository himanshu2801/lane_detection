import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt

# this function is for separating or defining our main region of interest which is of trigular shape.

def lane_region(img):
    height = img.shape[0]
    #here we got the point 150,600,300 and 200 using matplotlib library
    region = np.array([[(150, height), (600, height), (300, 200)]])
    black = np.zeros_like(img)
    cv.fillPoly(black, region, 255)
    region_lane = cv.bitwise_and(img, black)
    return region_lane

#This function is to display the lane over our real image.

def display_line(img, lines):
    lane_image = np.zeros_like(img)
    if lines is not None:
        for line in lines:
            #line.reshape(4) return the array contaning the points for line and 4 here displaying the no of values in the single dimension array
            #which you should know after checking the value using print function.
            x1, y1, x2, y2 = line.reshape(4)
            cv.line(lane_image, (x1, y1), (x2, y2),
                    (0, 255, 0), 2)
    return lane_image

#reading the image
img = cv.imread('road.jpg')

#resizing it according to your comfort
img = cv.resize(img, (700, 500))

#storing the copy of image in other variable
copy_img = np.copy(img)

#converting into gray scale as it contain only 1 channel which make it lot easier to implements function on our image.
gray_img = cv.cvtColor(copy_img, cv.COLOR_BGR2GRAY)

blur = cv.GaussianBlur(gray_img, (5,5), 0)

#detecting the edges in our image
canny = cv.Canny(blur, 50, 150)
#plt.imshow(canny) To check the points for our region of interest
#plt.show()

#Getting the only region that we required 
lane = lane_region(canny)

# using hough lines function we get the plotting points of our lines (return in the form of 2d array)
lines = cv.HoughLinesP(lane, 2, np.pi/180, 100, np.array([]), minLineLength=20, maxLineGap=5)
display_line_image = display_line(copy_img, lines)

#using bitwise or for displaying the lines on our real image
img = cv.bitwise_or(img, display_line_image)



cv.imshow('image', img)
cv.waitKey(0)

# for video capturing we just need a continous loop until we break it
#below code is for the Video capturing purpose
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
