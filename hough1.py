import cv2
import numpy as np
import sys


def normal_hough(src_img, output_file):
    gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges,1,np.pi/180,200)
    for rho,theta in lines[0]:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
    cv2.line(src_img, (x1,y1), (x2,y2), (0,0,255), 2)
    cv2.imwrite(output_file, src_img)


def prob_hough(src_img, output_file):
    gray = cv2.cvtColor(src_img,cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray,50,150,apertureSize = 3)
    minLineLength = 100
    maxLineGap = 10
    lines = cv2.HoughLinesP(edges,1,np.pi/180,100,minLineLength,maxLineGap)
    for x1,y1,x2,y2 in lines[0]:
        cv2.line(src_img,(x1,y1),(x2,y2),(0,255,0),2)
    cv2.imwrite(output_file, src_img)


def main(args):
    img = cv2.imread(args[1])
    prob_hough(img, args[2])


if __name__ == '__main__':
    sys.exit(main(sys.argv))