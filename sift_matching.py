import cv2
import numpy as np
import itertools
import sys
import argparse
import os


def findKeyPoints(img, template, distance=200):
    detector = cv2.FeatureDetector_create("SIFT")
    descriptor = cv2.DescriptorExtractor_create("SIFT")

    skp = detector.detect(img)
    skp, sd = descriptor.compute(img, skp)

    tkp = detector.detect(template)
    tkp, td = descriptor.compute(template, tkp)

    flann_params = dict(algorithm=1, trees=4)
    flann = cv2.flann_Index(sd, flann_params)
    idx, dist = flann.knnSearch(td, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    skp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            skp_final.append(skp[i])

    flann = cv2.flann_Index(td, flann_params)
    idx, dist = flann.knnSearch(sd, 1, params={})
    del flann

    dist = dist[:,0]/2500.0
    dist = dist.reshape(-1,).tolist()
    idx = idx.reshape(-1).tolist()
    indices = range(len(dist))
    indices.sort(key=lambda i: dist[i])
    dist = [dist[i] for i in indices]
    idx = [idx[i] for i in indices]
    tkp_final = []
    for i, dis in itertools.izip(idx, dist):
        if dis < distance:
            tkp_final.append(tkp[i])

    return skp_final, tkp_final


def drawKeyPoints(img, template, skp, tkp, num=-1, horizontal_alignment=True):
    hI, wI = img.shape[:2]
    hT, wT = template.shape[:2]
    if horizontal_alignment:
        nWidth = wT + wI
        wT_start = 0
        wI_start = wT
        if hT >= hI:
            hT_start = 0
            hI_start = (hT-hI)/2
            nHeight = hT
        else:
            hT_start = (hI-hT)/2
            hI_start = 0
            nHeight = hI
    else:
        nHeight = hT + hI
        hT_start = 0
        hI_start = hT
        if wT >= wI:
            wT_start = 0
            wI_start = (wT-wI)/2
            nWidth = wT
        else:
            wT_start = (wI-wT)/2
            wI_start = 0
            nWidth = wI
    #print 'template: %dx%d (%d:%d), image: %dx%d (%d:%d)' % (wT, hT, wT_start, hT_start, wI, hI, wI_start, hI_start)
    newimg = np.zeros((nHeight, nWidth, 3), np.uint8)
    newimg[hT_start:hT_start+hT, wT_start:wT_start+wT] = template
    newimg[hI_start:hI_start+hI, wI_start:wI_start+wI] = img
    maxlen = min(len(skp), len(tkp))
    if num < 0 or num > maxlen:
        num = maxlen
    for i in range(num):
        pt_a = (wT_start + int(tkp[i].pt[0]), hT_start + int(tkp[i].pt[1]))
        pt_b = (wI_start + int(skp[i].pt[0]), hI_start + int(skp[i].pt[1]))
        cv2.line(newimg, pt_a, pt_b, (255, 0, 0))
    return newimg


def match(img, temp, dist, num, horizontal_alignment=True):
    skp, tkp = findKeyPoints(img, temp, dist)
    newimg = drawKeyPoints(img, temp, skp, tkp, num, horizontal_alignment)
    cv2.imshow("image", newimg)
    cv2.waitKey(0)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', required=True, help='first image')
    parser.add_argument('--template', required=True, help='second image')
    parser.add_argument('--align', default='horizontal', choices=['horizontal', 'vertical'], help='alignment of images')
    parser.add_argument('--dist', type=int, default=200, help='distance measurement')
    parser.add_argument('--num', type=int, default=-1, help='number matches')
    parsed_args = parser.parse_args()
    if not os.path.isfile(parsed_args.image):
        print >>sys.stderr, 'image not a file:', parsed_args.image
        sys.exit(1)
    if not os.path.isfile(parsed_args.template):
        print >>sys.stderr, 'template not a file:', parsed_args.template
        sys.exit(1)
    return parsed_args


def main(args):
    pa = parse_args()
    img, template = cv2.imread(pa.image), cv2.imread(pa.template)
    match(img, template, pa.dist, pa.num, pa.align == 'horizontal')


if __name__ == '__main__':
    sys.exit(main(sys.argv))