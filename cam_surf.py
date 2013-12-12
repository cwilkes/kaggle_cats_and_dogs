import cv2
import sys
import numpy as np

#Create object to read images from camera 0
cam = cv2.VideoCapture(0)

#Initialize SURF object
surf = cv2.SURF(85)

#Set desired radius
rad = 2


def do_hough_lines(img, gray):
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
    for x1, y1, x2, y2 in lines[0]:
        cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)


def do_surf(img, gray):
    #Detect keypoints and descriptors in greyscale image
    #keypoints, descriptors = surf.detect(gray)
    keypoints = surf.detect(gray)

    #Draw a small red circle with the desired radius
    #at the (x, y) location for each feature found
    for kp in keypoints:
        x = int(kp.pt[0])
        y = int(kp.pt[1])
        cv2.circle(img, (x, y), rad, (0, 0, 255))


def main(args):
    if len(args) <= 1:
        action = do_surf
    else:
        action = do_hough_lines
    while True:
        #Get image from webcam and convert to greyscale
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        action(img, gray)
        cv2.imshow("features", img)
        #Sleep infinite loop for ~10ms
        #Exit if user presses <Esc>
        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    sys.exit(main(sys.argv))