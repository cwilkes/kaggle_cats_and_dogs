import cv2
import sys
import numpy as np
import argparse

#Create object to read images from camera 0

#Initialize SURF object
surf = cv2.SURF(85)

#Set desired radius
rad = 2


class ImageMethods(object):
    def do_line(self, img, gray):
        edges = cv2.Canny(gray, 50, 150, apertureSize = 3)
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength, maxLineGap)
        for x1, y1, x2, y2 in lines[0]:
            cv2.line(img,(x1,y1),(x2,y2),(0,255,0),2)

    def do_surf(self, img, gray):
        #Detect keypoints and descriptors in greyscale image
        #keypoints, descriptors = surf.detect(gray)
        keypoints = surf.detect(gray)

        #Draw a small red circle with the desired radius
        #at the (x, y) location for each feature found
        for kp in keypoints:
            x = int(kp.pt[0])
            y = int(kp.pt[1])
            cv2.circle(img, (x, y), rad, (0, 0, 255))

    def do_edge(self, img, gray):
        edges = cv2.Canny(img, 100, 200)
        for channel in (0, 1, 2):
            img[:, :, channel] = cv2.max(img[:, :, channel], edges)


def main(args):
    known_actions = set(['line', 'surf', 'edge'])
    actions = []
    im = ImageMethods()
    for _ in args[1:]:
        if not _ in known_actions:
            print 'Unknown actions', _
            return 1
        actions.append(getattr(im, 'do_%s' % (_, )))

    cam = cv2.VideoCapture(0)
    action_pos = 0
    while True:
        #Get image from webcam and convert to greyscale
        ret, img = cam.read()
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        if action_pos == len(actions):
            action_pos = 0
        actions[action_pos](img, gray)
        action_pos+=1
        cv2.imshow("features", img)
        #Sleep infinite loop for ~10ms
        #Exit if user presses <Esc>
        if cv2.waitKey(10) == 27:
            break


if __name__ == '__main__':
    sys.exit(main(sys.argv))