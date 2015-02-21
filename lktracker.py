import numpy as np
import time
from itertools import izip, islice
import cv2
import sys
import math
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs
from sklearn.cluster import DBSCAN
CLUSTER_FRAME = 10
STEP_SIZE = 12
def getComponents(normalised_homography):
    '''((translationx, translationy), rotation, (scalex, scaley), shear)'''
    a = normalised_homography[0, 0]
    b = normalised_homography[0, 1]
    c = normalised_homography[0, 2]
    d = normalised_homography[1, 0]
    e = normalised_homography[1, 1]
    f = normalised_homography[1, 2]

    p = math.sqrt(a * a + b * b)
    r = (a * e - b * d) / (p)
    q = (a * d + b * e) / (a * e - b * d)

    translation = (c, f)
    scale = (p, r)
    shear = q
    theta = math.atan2(b, a)

    return (translation, theta, scale, shear)


def camera_pose_from_homography(H):
    H1 = H[:, 0]
    H2 = H[:, 1]
    H3 = np.cross(H1, H2)

    norm1 = np.linalg.norm(H1)
    norm2 = np.linalg.norm(H1)
    tnorm = (norm1 + norm2) / 2.0;

    T = H[:, 2] / tnorm
    return np.mat([H1, H2, H3, T])


anchor_side = 40
# some constants and default parameters
lk_params = dict(winSize=(15, 15), maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
                 flags=cv2.OPTFLOW_LK_GET_MIN_EIGENVALS, minEigThreshold=0.01)

lk_params = dict(winSize=(15, 15), maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
)
# winSize=None, maxLevel=None, criteria=None, flags=None, minEigThreshold=None): # real signature unknown; restored from __doc__
subpix_params = dict(zeroZone=(-1, -1), winSize=(10, 10),
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))

feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10)

random_byte = lambda: random.randint(0, 255)

random_color = lambda: (random_byte(), random_byte(), random_byte())

COLORS = [
    (0, 0, 0),
    (255, 255, 255),
    (255, 0, 0),
    (0, 255, 0),
    (0, 0, 255),
    (255, 255, 0),
    (0, 255, 255),
    (255, 0, 255),
    (192, 192, 192),
    (128, 128, 128),
    (128, 0, 0),
    (128, 128, 0),
    (0, 128, 0),
    (128, 0, 128),
    (0, 128, 128),
    (0, 0, 128),
]


def make_angle(x1, y1, x2, y2):
    angle = math.atan2(y1 - y2, x1 - x2)
    return angle


def make_magnitude(x1, y1, x2, y2):
    return math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)

def flow_motions(flow):
    """
    h, w = img.shape[:2]
    y, x = np.mgrid[step/2:h:step, step/2:w:step].reshape(2,-1)
    y, x = np.mgrid[1/2:h:1, 1/2:w:1].reshape(2,-1)
    fx, fy = flow[y,x].T
    """
    h, w = flow.shape[:2]
    y, x = np.mgrid[1 / 2:h:1, 1 / 2:w:1]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    values = np.zeros((h, w, 2), np.float)
    # values[...,0] = x
    # values[...,1] = y
    values[..., 0] = ang
    values[..., 1] = v
    return values


def flow_motions(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    # ang = np.arctan2(fy, fx) + np.pi
    val = np.sqrt(fx * fx + fy * fy)
    # ang = np.int32(ang*(180/np.pi/2))
    # val = np.int32(np.minimum(val*4, 255))
    # hsv = np.zeros((h, w, 2), np.uint8)
    # hsv[...,0] = nang
    # hsv[...,1] = nval
    # vals = np.dstack((ang, val))
    # fa = ang.reshape(-1, 1)
    fv = val.reshape(-1, 1)
    av = np.average(fv, axis=0)
    # aa = np.average(fa, axis=0)

    return val, av[0], #aa[0],


def draw_hsv(flow):
    h, w = flow.shape[:2]
    fx, fy = flow[:, :, 0], flow[:, :, 1]
    ang = np.arctan2(fy, fx) + np.pi
    v = np.sqrt(fx * fx + fy * fy)
    hsv = np.zeros((h, w, 3), np.uint8)
    hsv[..., 0] = ang * (180 / np.pi / 2)
    hsv[..., 1] = np.minimum(v * 4, 255)
    hsv[..., 2] = 255
    bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return bgr

def neighbours_less_than(point, distance, data, delta):
    height, width = data.shape
    border_x = width - 1
    border_y = height - 1
    x, y = point
    min_x = x - distance
    min_y = y - distance
    max_x = x + distance
    max_y = y + distance
    col_left, col_right, row_top, row_bottom = True, True, True, True
    if min_x < 0:
        col_left = False
        min_x = 0
    if min_y < 0:
        row_top = False
        min_y = 0
    if max_x > border_x:
        col_right = False
        max_x = border_x
    if max_y > border_y:
        row_bottom = False
        max_y = border_y
    indexes_x = np.arange(min_x, max_x+1)
    indexes_y = np.arange(min_y, max_y+1)

    if col_left:
        segment = data[min_y:max_y+1, min_x]
        condition = segment < delta
        needle = indexes_y[condition]
        if len(needle):
            return np.array([min_x, needle[0]])
    if row_top:
        segment = data[min_y, min_x:max_x+1]
        condition = segment < delta
        needle = indexes_x[condition]
        if len(needle):
            return np.array([needle[0], min_y])
    if col_right:
        segment = data[min_y:max_y+1, max_x]
        condition = segment < delta
        needle = indexes_y[condition]
        if len(needle):
            return np.array([max_x, needle[0]])
    if row_bottom:
        segment = data[max_y, min_x:max_x+1]
        condition = segment < delta
        needle = indexes_x[condition]
        if len(needle):
            return np.array([needle[0], max_y])

ANGLE_DELTA = 0.1
MAGNITUDE_DELTA = 0.1
POSITION_DELTA = 16


def bounding_box(iterable):
    min_x = min(iterable, key=lambda x: x[0])[0]
    max_x = max(iterable, key=lambda x: x[0])[0]
    min_y = min(iterable, key=lambda x: x[1])[1]
    max_y = max(iterable, key=lambda x: x[1])[1]
    return min_x, min_y, max_x, max_y


class Block(object):
    def __init__(self, label, indexes, points, centroid):
        self.label = label
        if label >= len(COLORS):
            self.color = random_color()
        else:
            self.color = COLORS[label]
        self.centroid = centroid
        self.indexes = indexes
        # print points
        self.top = 0
        self.points = None
        self.bbox = None
        self.update(points)

    def update(self, points):
        self.points = points
        self.bbox = bounding_box(self.points)
        self.top = (self.bbox[0], self.bbox[1])

    def evaluate(self):
        pass

    def draw(self, image):
        for point in self.points:
            # if index == len(self.features) - 1:
            # print "I FOUND", index, block.label, block.top, point
            cv2.circle(image, (int(point[0]), int(point[1])), 3, self.color, -1)

        v1, v2 = self.rect()
        cv2.rectangle(image, (v1[0], v1[1]), (v2[0], v2[1]), self.color)

        mark = "%s" % str(np.round(self.centroid, 2))
        cv2.putText(image, mark, (v1[0], v1[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, self.color)

    def rect(self):
        x0, y0, x1, y1 = self.bbox
        return (x0, y0), (x1, y1)

    def __iter__(self):
        return izip(self.indexes, self.points)

    def __repr__(self):
        s = ""
        for i, p in self:
            s += str((i, p))
        s += "]"
        return s

class Detector(object):
    def __init__(self):
        self.points = []
        self.median = None
        self.tracks = []
        self.frames_left = 0
        self.blocks = []
        self.legend = np.zeros((1000,700,3), np.uint8)
        self.legend[:,:] = 255
    def init(self, features, motions, threshold):
        self.frames_left = CLUSTER_FRAME
        self.median = len(features)
        self.indexes = np.arange(self.median)
        self.delta_angles = [[] for i in range(self.median)]
        self.delta_mags = [[] for i in range(self.median)]
        self.weights = [[] for i in range(self.median)]
        self.weights2 = [[] for i in range(self.median)]
        self.results = np.zeros(self.median, dtype=np.float32)
        anchor_points = self.create_anchors(features, motions, 0.1)
        # for a,p in izip(anchor_points, features):
        #     print a
        #     print p
        # anchor_points = np.array([[p[0] - 20, p[1]+20] for p in features])
        points = np.append(features, anchor_points, axis=0)
        self.tracks = [points]
        self.etalons = self.measure(points)
        return points

    def create_anchors(self, features, motions, threshold):
        anchors = np.zeros_like(features, dtype=np.int32)
        sizes = range(30, 100)
        bad = np.array([-1,-1])
        # unique_sizes = []
        for i, point in enumerate(features):
            for size in sizes:
                anchor = neighbours_less_than(point, size, motions, threshold)
                if anchor is not None:
                    # unique_sizes.append(size)
                    anchors[i] = anchor
                    break
            else:
                anchors[i] = bad

        return anchors

    def split(self, points):
        targets = points[0:self.median]
        anchors = points[self.median:]
        return targets, anchors

    def measure(self, points):
        targets, anchors = self.split(points)

        def calculate(vec):
            return np.array([make_magnitude(*vec), make_angle(*vec)] )

        vectors = np.concatenate((targets, anchors), axis=1)
        values = np.apply_along_axis(calculate, 1, vectors)
        return values

    def update(self, points):
        self.update_blocks(points)
        self.tracks.append(points)
        values = self.measure(points)
        for i in range(self.median):
            etalon = self.etalons[i]
            current = values[i]
            delta_mags = self.delta_mags[i]
            delta_angles = self.delta_angles[i]
            weights = self.weights[i]
            weights2 = self.weights2[i]
            delta_mags.append(current[0] / etalon[0])
            delta_angles.append(current[1] / etalon[1])

            weights.append(np.average(delta_mags))
            weights2.append(np.average(delta_angles))
            self.results[i] = abs(1-weights[-1]) + abs(1-weights2[-1])

        self.frames_left -= 1
        if self.frames_left <= 0:
            self.clusterize()
            self.frames_left = CLUSTER_FRAME

    def update_blocks(self, points):
        # print "POINTS", len(points)
        for block in self.blocks:
            try:
                block_points = points[block.indexes]
                block.update(block_points)
            except Exception as e:
                print self.labels
                print len(self.blocks)
                print points
                print block
                raise e

    def meanshift(self, data, quantile):
        bandwidth = estimate_bandwidth(data, quantile=quantile)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(data)
        return ms


    def clusterize_by_weight(self, data,  delta):
        EMPTY_LABEL = -1
        centers = []
        center_values = []
        labels = np.random.randint(EMPTY_LABEL, EMPTY_LABEL+1, len(data))
        for i, item in enumerate(data):
            criteria = item
            for mark, center in enumerate(centers):
                if abs(center - criteria) < delta:
                    center_values[mark].append(criteria)
                    new_center = sum(center_values[mark]) / len(center_values[mark])
                    centers[mark] = new_center
                    labels[i] = mark
                    break
            else:
                centers.append(criteria)
                center_values.append([criteria])
                labels[i] = len(centers) - 1
        return centers,labels


    def clusterize_by_position(self, data, delta):
        db = DBSCAN(eps=100, min_samples=1).fit(X)
        core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        core_samples_mask[db.core_sample_indices_] = True
        labels = db.labels_
        return labels


    def clusterize(self):
        self.blocks = []
        points = self.tracks[-1]
        target, anchors = self.split(points)
        condition = self.results > 0
        indexes = self.indexes[condition]
        weights = self.results[indexes]
        block_points = points[indexes]
        np.save("blockpoints.np", block_points)
        # data = np.concatenate((block_points, weights.reshape(-1,1)), axis=1)
        # data = block_points

        centers, labels = self.clusterize_by_weight(weights, 0.02)
        # centers, labels = self.clusterize_by_position(block_points, 0.01)
        for i, centroid in enumerate(centers):
            # print labels.ravel() == i
            print centroid
            __indexes = indexes[labels.ravel() == i]
            __points = block_points[labels.ravel() == i]
            print __points
            block = Block(i, __indexes, __points, centroid)
            self.blocks.append(block)

        self.labels = labels

    def draw_legend(self):
        y = 0
        for i,block in enumerate(self.blocks):
            x = 10
            y += 20
            cv2.circle(self.legend, (x, y), 5, block.color, thickness=2)
            x += 20
            mark = "block %d  points : %d centroid: %s" % (i, len(block.points), str(np.round(block.centroid, 2)))
            # mark = "%s : %s" % (str(label), str(block.delta_distance))
            cv2.putText(self.legend, mark, (x,y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[0])
        cv2.imshow("Legend", self.legend)
        pass

    def draw_blocks(self, image):
        for block in self.blocks:
            block.draw(image)

    def draw_points(self, image):
        index = len(self.tracks) - 1
        if index < 1:
            return

        targets, anchors = self.split(self.tracks[index])

        for i, p0, p1 in izip(range(self.median), targets, anchors):
            w = self.weights[i][-1]
            w2 = self.weights2[i][-1]
            cv2.circle(image, (int(p0[0]), int(p0[1])), 5, COLORS[0], thickness=1)
            cv2.circle(image, (int(p1[0]), int(p1[1])), 5, COLORS[1], thickness=1)
            cv2.line(image, (int(p0[0]), int(p0[1])), (int(p1[0]), int(p1[1])), COLORS[0])
            mark = "%s : %s %s" % (str(i), str(round(w, 2)), str(round(w2, 2)))
            # mark = "%s : %s" % (str(label), str(block.delta_distance))
            cv2.putText(image, mark, (int(p0[0]), int(p0[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COLORS[0])

    def draw(self, image):
        self.draw_legend()
        self.draw_blocks(image)
        # self.draw_points(image)

class LKTracker(object):
    """    Class for Lucas-Kanade tracking with
        pyramidal optical flow."""

    def __init__(self, filename):
        """    Initialize with a list of image names. """
        self.filename = filename

        self.cap = cv2.VideoCapture()
        self.cap.open(filename)

        self.pause = False
        self.trace_tracks = False

        self.gray0 = None
        self.gray1 = None
        self.image = None
        self.size = [0, 0]
        self.current_frame = 0

        # self.video_writer = None
        self.indexes = []
        self.features = []
        self.initial_features = []
        self.tracks = []
        self.status = []

        self.angles = []
        self.distances = []
        self.weights = []


        self.cluster_frame = 0

        self.anchor_points = []

        self.blocks = []
        self.last_block = 0
        self.step_mode = True
        self.step_size = STEP_SIZE


    def init_tracking(self):
        # self.read()
        self.current_frame = 0
        self.image = self.read()
        # fourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
        # if self.video_writer != None:
        # self.video_writer.release()
        # self.video_writer = cv2.VideoWriter("output.avi", fourcc, 20, self.size)
        self.cluster_frame = 9
        # load the image and create grayscale
        self.gray0 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # search for good points
        self.features = cv2.goodFeaturesToTrack(self.gray0, **feature_params)
        # self.features = self.features.reshape(-1,  2)
        # refine the corner locations
        # cv2.cornerSubPix(self.gray0, self.features, **subpix_params)
        self.features = np.float32(self.features).reshape(-1, 2)
        self.detector = Detector()
        self.image = self.read()
        self.gray1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(self.gray0, self.gray1, 0.5, 3, 1, 3, 7, 1.5, 0)
        motions,avg1 = flow_motions(flow)
        self.features = self.detector.init(self.features, motions, avg1)

        self.indexes = [i for i in range(len(self.features))]
        self.indexes = np.array(self.indexes)
        self.tracks = [[p] for p in self.features]
        self.status = np.ones((len(self.features), 1))
        self.initial_features = self.features.copy()

        points0, points1 = self.next_features()
        self.detector.update(points1)

    def create_block_links(self):
        anchors = []
        index = len(self.blocks)
        point_index = len(self.features)
        labels = self.labels.tolist()
        for block in self.blocks:
            points = [[block.top[0] - 20, block.top[1] + 20]]
            self.features = np.append(self.features, np.array([points]), axis=0)
            anchor = Anchor(block, index, [point_index], points, [0], [0],
                            0, 0, 0, 0)
            anchors.append(anchor)
            labels.append(index)
            index += 1
            point_index += 1

        self.blocks += anchors
        self.labels = np.array(labels)
        self.indexes = [i for i in range(len(self.features))]
        self.indexes = np.array(self.indexes)
        self.tracks = [[p[0]] for p in self.features]
        self.status = np.ones((len(self.features), 1))
        self.initial_features = self.features.copy()


    def read(self):
        self.current_frame += 1
        ret, image = self.cap.read()

        image = cv2.resize(image,(image.shape[1],700))
        return image

    def next_features(self):
        self.image = self.read()
        self.gray1 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # affine = cv2.estimateRigidTransform(self.gray0, self.gray1, True)
        # res = cv2.matchTemplate(self.gray1, self.anchor_data, cv2.TM_SQDIFF)
        # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # top_left = min_loc
        # self.anchor = (top_left[0], top_left[1], top_left[0] + anchor_side, top_left[1] + anchor_side)
        # reshape to fit input format
        previous = np.float32(self.features).reshape(-1, 1, 2)

        # calculate optical flow
        self.features, self.status, track_error = cv2.calcOpticalFlowPyrLK(self.gray0, self.gray1, previous, None,
                                                                           **lk_params)

        self.features = np.float32(self.features).reshape(-1, 2)
        # affine = cv2.getAffineTransform(previous, self.features)
        # H, status_homo = cv2.findHomography(previous, self.features, cv2.RANSAC, 1.0)
        # # print status_homo
        # h, w = self.image.shape[:2]
        # print "COMPS:", getComponents(H)
        # print "!!!!!", pos
        # print self.features[0][0]
        #
        # self.features2 = cv2.perspectiveTransform(self.features, H)
        # # print self.features[0][0]
        # for p1,p2 in zip(self.features, self.features2):
        # cv2.circle(self.image, (p2[0][0], p2[0][1]), 1, COLORS[0])
        # cv2.circle(self.image, (p1[0][0], p1[0][1]), 1, COLORS[1])
        #     # print p1,p2
        # self.image = cv2.warpPerspective(self.gray1, H, (w, h))
        # self.image = cv2.addWeighted(self.image, 0.5, self.gray0, 0.5, 0)
        # diff = cv2.absdiff(self.gray0, self.gray1)
        # cv2.imshow("diff", diff)
        # self.image = self.gray1
        # self.image = overlay
        # self.features, self.status, track_error = cv2.calcOpticalFlowPyrLK(self.gray0, overlay, previous, None,
        # **lk_params)
        # self.image = cv2.warpPerspective(self.image, H, (w,h))
        # for i,s in enumerate(self.status):
        # print "status",i,s
        # for i,s in enumerate(track_error):
        #     print "error",i,s
        # print "error", 146, track_error[146]
        # print "error", 7, track_error[7]
        # print "error", 15, track_error[15]
        # print "error", 26, track_error[26]
        # print self.features
        # print "LENGTH", len(previous), len(self.features)
        self.gray0 = self.gray1
        # reversed, sr, er = cv2.calcOpticalFlowPyrLK(self.gray, self.gray0, self.features, None, **lk_params)
        #
        # # Compute the distance between corresponding points in the two flows
        # d = abs(self.features - reversed).reshape(-1, 2).max(-1)
        # good = d < 1
        # for i, p, s, e, s1, e1 in zip(range(len(self.features)), d, status, track_error, sr, er):
        # print i, p, s, e,s1,e1
        # # If the distance between pairs of points is < 1 pixel, set
        # # a value in the "good" array to True, otherwise False
        return previous, self.features

    def track_points(self):
        # load the image and create grayscale
        # print self.current_features
        # remove points lost
        self.next_features()
        # self.calculate_distances(self.features)
        # self.features = [p for (st, p) in zip(status, self.current_features) if st]


        # # clean tracks from lost points
        features = np.array(self.features).reshape((-1, 2))

        # points = self.features.reshape(-1, 1, 2)
        self.detector.update(features)

        for i, f in enumerate(features):
            self.tracks[i].append(f)
            # ndx = [i for (i, st) in enumerate(status) if not st]
            # ndx.reverse()  # remove from back
            # for i in ndx:
            # self.tracks.pop(i)


    def track(self):
        """    Generator for stepping through a sequence."""
        while True:
            if self.step_mode:
                if self.current_frame != 0 and self.current_frame % self.step_size == 0:
                    self.pause = True
                    self.draw()
                    self.dispatch()
                    continue

            if self.pause:
                # time.sleep(1)
                self.dispatch()
                continue

            if self.features == []:
                self.init_tracking()
            else:
                # self.create_blocks()
                self.track_points()

            self.cluster_frame -= 1
            self.draw()
            self.dispatch()

    def draw(self):
        cv2.putText(self.image, str(self.current_frame) + "-" + str(round(len(self.blocks), 2)), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[0])
        self.detector.draw(self.image)
        # cv2.rectangle(self.image, (self.initial_anchor[0], self.initial_anchor[1]), (self.initial_anchor[2], self.initial_anchor[3]), COLORS[0])
        # cv2.imshow("Anchor", self.anchor_data)
        for block in self.blocks:
            # if label not in [54, 79]:
            # continue
            block.draw(self.image)

        cv2.imshow('LKtrack', self.image)
        # self.video_writer.write(self.image)

    def dispatch(self):
        ch = 0xFF & cv2.waitKey(10)
        # ch = cv2.waitKey(0)
        if ch == ord(' '):
            self.pause = not self.pause
            self.current_frame = 0
        if ch == ord('t'):
            self.trace_tracks = not self.trace_tracks
        if ch == ord('r'):
            self.pause = False
            self.current_frame = 0
            self.init_tracking()

tracker = LKTracker(sys.argv[1])
tracker.track()