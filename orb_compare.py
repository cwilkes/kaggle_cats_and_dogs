import cv2
import sys


def do_orb(img, output_file):
    # Initiate STAR detector
    orb = cv2.ORB()

    # find the keypoints with ORB
    kp = orb.detect(img,None)

    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)

    # draw only keypoints location,not size and orientation
    img2 = cv2.drawKeypoints(img,kp,color=(0,255,0), flags=0)
    cv2.imwrite(output_file, img2)
    #plt.imshow(img2)
    #plt.show()


def main(args):
    img1, img2 = cv2.imread(args[1], 0), cv2.imread(args[2], 0)
    orb = cv2.ORB()
    kps = orb.detect(img1, None)
    for kp in kps:
        print kp.pt, kp.size, kp.angle


if __name__ == '__main__':
    sys.exit(main(sys.argv))