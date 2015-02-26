import numpy as np
import cv2

lk_params = dict(winSize=(15, 15), maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
feature_params = dict(maxCorners=1000,
                      qualityLevel=0.01,
                      minDistance=8,
                      blockSize=3)
subpix_params = dict(zeroZone=(-1, -1), winSize=(10, 10),
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))


def identity_homography():
    # return np.zeros((3,3), dtype=np.float32)
    return np.eye(3)


def curl(h):
    return abs(h[1, 0] - h[0, 1])


def deformation(h):
    return abs(h[0, 0] - h[1, 1])


def calculate_points_lk(srcgray, dstgray):
    points0 = cv2.goodFeaturesToTrack(dstgray, **feature_params)
    cv2.cornerSubPix(dstgray, points0, **subpix_params)
    if points0 is None:
        return None, None

    points1, status, err = cv2.calcOpticalFlowPyrLK(srcgray, dstgray, points0, None, **lk_params)
    return points0, points1


def filter_matches(kp1, kp2, matches, ratio=0.75):
    mkp1, mkp2 = [], []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            m = m[0]
            mkp1.append(kp1[m.queryIdx])
            mkp2.append(kp2[m.trainIdx])
    kp_pairs = zip(mkp1, mkp2)
    return kp_pairs


def calculate_points_surf(srcgray, dstgray):
    detector = cv2.SURF(400, 5, 5)
    matcher = cv2.BFMatcher(cv2.NORM_L2)

    kp1, desc1 = detector.detectAndCompute(srcgray, None)
    kp2, desc2 = detector.detectAndCompute(dstgray, None)
    # print 'img1 - %d features, img2 - %d features' % (len(kp1), len(kp2))

    raw_matches = matcher.knnMatch(desc1, trainDescriptors=desc2, k=2)  #2
    kp_pairs = filter_matches(kp1, kp2, raw_matches)
    mkp1, mkp2 = zip(*kp_pairs)
    p1 = np.float32([kp.pt for kp in mkp1])
    p2 = np.float32([kp.pt for kp in mkp2])
    if len(kp_pairs) >= 4:
        return p1, p2
    else:
        return None, None


def read(cam, curl_threshold=0.03, deformation_threshold=0.02):
    H = [identity_homography()]
    count = 0
    ret, src = cam.read()
    srcgray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    srcgray = cv2.equalizeHist(srcgray)
    while True:
        ret, dst = cam.read()
        dstgray = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        dstgray = cv2.equalizeHist(dstgray)

        # if count % 4 == 0:
        #     points0, points1 = calculate_points_surf(srcgray, dstgray)
        # else:
        points0, points1 = calculate_points_lk(srcgray, dstgray)

        if points0 is None or points1 is None:
            continue

        h, status_homo = cv2.findHomography(points0, points1, cv2.RANSAC, 1)
        height, width = dst.shape[:2]
        prev_h = H[count]
        new_h = np.dot(h, prev_h)
        c = curl(new_h)
        d = deformation(new_h)
        # print c,d
        if c > curl_threshold or d > deformation_threshold:
            new_h = identity_homography()

        dstwarp = cv2.warpPerspective(dst, prev_h, (width, height))
        graywarp = cv2.warpPerspective(dstgray, prev_h, (width, height))
        # overlay = cv2.addWeighted(dst, 0.5, dstwarp, 0.5, 0)
        # overlay = cv2.absdiff(dst, dstwarp)
        # cv2.imshow("rov", overlay)
        # print prev_h
        yield dstwarp, graywarp, prev_h
        # yield dst, dstgray, prev_h

        srcgray = dstgray
        src = dst
        H.append(new_h)
        count += 1


class Scaler(object):
    def __init__(self, cam):
        self.cam = cam

    def read(self):
        ret, frame = self.cam.read()
        scale = 0.8
        frame = cv2.resize(frame, (int(frame.shape[1] * scale), int(frame.shape[0] * scale)))
        height, width = frame.shape[:2]
        # scale = frame.shape * 2
        scaled = (height * 2, width * 2, 3)
        canvas = np.zeros(scaled, dtype=np.uint8)

        x_offset = int(width * scale) / 4
        y_offset = int(height * scale) / 4
        # x_offset = int(width / scale)
        # y_offset = int(height / scale)
        canvas[y_offset:y_offset + frame.shape[0], x_offset:x_offset + frame.shape[1]] = frame
        return ret, canvas


class Reader(object):
    def __init__(self, cam, threshold=(0.03, 0.02)):
        self.threshold = threshold
        self.reader = read(cam, threshold[0], threshold[1])

    def read(self):
        return next(self.reader)
