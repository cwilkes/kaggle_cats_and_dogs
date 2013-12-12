import cv2
import sys


#https://opencv-python-tutroals.readthedocs.org/en/latest/py_tutorials/py_feature2d/py_fast/py_fast.html#fast

def do_fast(img, file_true, file_false):
    # Initiate FAST object with default values
    fast = cv2.FastFeatureDetector(50)

    # find and draw the keypoints
    kp = fast.detect(img, None)
    img2 = cv2.drawKeypoints(img, kp, color=(255, 0, 0))

    # Print all default params
    #print "Threshold: ", fast.getInt('threshold')
    #print "nonmaxSuppression: ", fast.getBool('nonmaxSuppression')
    #print "neighborhood: ", fast.getInt('type')
    #print "Total Keypoints with nonmaxSuppression: ", len(kp)

    cv2.imwrite(file_true, img2)

    # Disable nonmaxSuppression
    fast.setBool('nonmaxSuppression',0)
    kp = fast.detect(img,None)

    print "Total Keypoints without nonmaxSuppression: ", len(kp)

    img3 = cv2.drawKeypoints(img, kp, color=(255,0,0))

    cv2.imwrite(file_false, img3)


def main(args):
    img = cv2.imread(args[1])
    do_fast(img, args[2], args[3])


if __name__ == '__main__':
    sys.exit(main(sys.argv))