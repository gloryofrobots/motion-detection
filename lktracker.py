import numpy as np
import time
from itertools import izip, islice
import cv2
import sys
import math
import random
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.datasets.samples_generator import make_blobs


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


ANGLE_DELTA = 0.1
MAGNITUDE_DELTA = 0.1
POSITION_DELTA = 16


class Block(object):
    def __init__(self):
        self.indexes = []

        self.center_x = 0
        self.center_y = 0
        self.xs = []
        self.ys = []
        self.angles = []
        self.magnitudes = []
        self.center_angle = 0
        self.center_magnitude = 0

    def rect(self):
        size = POSITION_DELTA / 2
        return np.int32([self.center_x - size, self.center_y - size]), np.int32(
            [self.center_x + size, self.center_y + size])

    def set_point(self, index, x, y, angle, magnitude):
        self.indexes.append(index)
        self.angles.append(angle)
        self.xs.append(x)
        self.ys.append(y)
        self.center_x = np.mean(self.xs)
        self.center_y = np.mean(self.ys)
        self.magnitudes.append(magnitude)
        self.center_angle = np.mean(self.angles)
        self.center_magnitude = np.mean(self.magnitudes)

    def is_fit(self, x, y, angle, magnitude):
        if abs(x - self.center_x) > POSITION_DELTA:
            return False
        if abs(y - self.center_y) > POSITION_DELTA:
            return False
        if abs(self.center_angle - angle) > ANGLE_DELTA:
            return False
        if abs(self.center_magnitude - magnitude) > MAGNITUDE_DELTA:
            return False

        return True

    def __iter__(self):
        return izip(self.indexes, self.angles, self.magnitudes)

    def __repr__(self):
        s = "angle = %f mag = %f points = [" % (self.center_angle, self.center_magnitude)
        for i, a, m in self:
            s += str((i, a, m))
        s += "]"
        return s


def bounding_box(iterable):
    min_x = min(iterable, key=lambda x: x[0])[0]
    max_x = max(iterable, key=lambda x: x[0])[0]
    min_y = min(iterable, key=lambda x: x[1])[1]
    max_y = max(iterable, key=lambda x: x[1])[1]
    return min_x, min_y, max_x, max_y


class Block(object):
    def __init__(self, label, indexes, points, angles, magnitudes, angle, magnitude, x, y):
        self.label = label
        if label >= len(COLORS):
            self.color = random_color()
        else:
            self.color = COLORS[label]

        if self.label == 54:
            self.color = COLORS[0]

        self.indexes = indexes
        self.angles = angles
        self.magnitudes = magnitudes
        self.x = x
        self.y = y
        self.center_angle = angle
        self.center_magnitude = magnitude

        self.top = 0
        self.distance = 0
        self.initial_points = points
        self.init_bbox = bounding_box(self.initial_points)
        self.points = None
        self.bbox = None
        self.track = []
        self.links = []
        self.distances = []
        self.set_points(points)
        self.weight = 0
        self.weight2 = 0
        self.weights2 = []
        self.weights = []
        self.averages = []
        self.initial_weight = 0
        self.angles = []
        self.dxdy = []
    def get_shifts(self):
        total_magintude = 0
        total_angle = 0
        total_dx = 0
        total_dy = 0
        for block in self.links:
            total_magintude += make_magnitude(self.top[0], self.top[1], block.top[0], block.top[1])
            # total_magintude += make_magnitude(self.top[0], self.top[1], self.init_bbox[0], self.init_bbox[1])
            total_angle += make_angle(self.top[0], self.top[1], block.top[0], block.top[1])
            total_dx += self.top[0] - block.top[0]
            total_dy += self.top[1] - block.top[1]
        total_magintude /= len(self.links)
        total_angle /= len(self.links)
        self.angles.append(total_angle)
        self.dxdy.append(np.round([total_dx,total_dy]))
        return total_magintude, total_angle

    def calculate_distance(self):
        mag, angle = self.get_shifts()
        self.weights.append(mag)
        try:
            delta = (self.weights[len(self.weights) - 2] - mag)
        except KeyError:
            delta = 0
        self.averages.append(delta)
        self.weight = np.sum(self.averages)
        # if self.label in [54, 1, 42, 83]:
        #     # print self.label, self.indexes, self.initial_weight, self.init_bbox, self.top, np.round(
        #     #     self.weights), np.round(self.averages)
        #
        #     print self.label, self.track, self.angles, self.dxdy
        #     # print np.round(self.track)

    def set_links(self, links):
        self.links = links

        mag, angle = self.get_shifts()
        self.initial_weight = mag
        self.initial_weight2 = angle

    def set_points(self, points):
        self.points = points
        self.bbox = bounding_box(self.points)
        previous = self.top
        self.top = (self.bbox[0], self.bbox[1])
        self.track.append(self.top)

        if previous:
            self.distances.append(make_magnitude(previous[0], previous[1], self.top[0], self.top[1]))

    def rect(self):
        x0, y0, x1, y1 = self.bbox
        return (x0, y0), (x1, y1)

    def __iter__(self):
        return izip(self.indexes, self.angles, self.magnitudes)

    def __repr__(self):
        s = "angle = %f mag = %f points = [" % (self.center_angle, self.center_magnitude)
        for i, a, m in self:
            s += str((i, a, m))
        s += "]"
        return s


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

        self.anchor = None
        # self.video_writer = None
        self.indexes = []
        self.features = []
        self.initial_features = []
        self.tracks = []
        self.status = []
        self.blocks = []
        self.point_to_block = dict()
        self.step_mode = False
        self.step_size = 20

    def create_anchor(self):
        self.size = (np.size(self.image, 1), np.size(self.image, 0))
        side = 40
        x0 = self.size[0] / 2
        y0 = self.size[1] / 2
        # x0 = self.size[0] - 100
        # y0 = 100
        x1 = x0 + side
        y1 = y0 + side
        self.initial_anchor = [x0, y0, x1, y1]
        self.anchor = self.initial_anchor
        self.anchor_data = self.gray0[y0:y1, x0:x1]

    def detect_corners(self):
        # self.read()
        self.current_frame = 0
        self.image = self.read()
        # fourcc = cv2.cv.CV_FOURCC('X', '2', '6', '4')
        # if self.video_writer != None:
        # self.video_writer.release()
        # self.video_writer = cv2.VideoWriter("output.avi", fourcc, 20, self.size)

        # load the image and create grayscale
        self.gray0 = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.create_anchor()

        # search for good points
        self.features = cv2.goodFeaturesToTrack(self.gray0, **feature_params)
        # self.features = self.features.reshape(-1,  2)
        anch = np.array([[[self.anchor[0], self.anchor[1]]]])
        # anch.reshape(-1, 1, 2)
        # print self.features
        self.features = np.append(self.features, anch, axis=0)
        # refine the corner locations
        # cv2.cornerSubPix(self.gray0, self.features, **subpix_params)
        self.indexes = [i for i in range(len(self.features))]
        self.indexes = np.array(self.indexes)
        self.tracks = [[p[0]] for p in self.features]
        self.status = np.ones((len(self.features), 1))
        self.initial_features = self.features.copy()

        EMPTY_LABEL = -1

    def create_blocks(self):
        self.blocks = []
        self.detect_corners()
        points0, points1 = self.next_features()

        points0 = points0.reshape((-1, 2))
        points1 = points1.reshape((-1, 2))

        def calculate(vec):
            return np.array([make_angle(*vec), make_magnitude(*vec)])

        vectors = np.concatenate((points0, points1), axis=1)
        values = np.apply_along_axis(calculate, 1, vectors)
        input_data = np.concatenate((points0, values), axis=1)
        bandwidth = estimate_bandwidth(input_data, quantile=0.01, n_samples=len(input_data))
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(input_data)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique = np.unique(labels)
        # print labels_unique
        for i, centroid in enumerate(cluster_centers):
            # print labels.ravel() == i
            block_indexes = self.indexes[labels.ravel() == i]
            block_points = points0[labels.ravel() == i]
            block_values = values[labels.ravel() == i]
            angles = block_values[:, 0]
            magnitudes = block_values[:, 1]
            block = Block(i, block_indexes, block_points, angles, magnitudes,
                          centroid[0], centroid[1], centroid[2], centroid[3])
            self.blocks.append(block)

        # for block in self.blocks:
        # block.links = np.random.choice(self.blocks, 6).tolist()
        # if block in block.links:
        # block.links.remove(block)

        self.labels = labels
        self.create_block_links()
        # for index, block in enumerate(self.blocks):
        # print index, block

        # for index, data in self.point_to_block.items():
        # print index, data

    def create_block_links(self):
        static_index = len(self.features) - 1

        for block in self.blocks:
            if static_index in block.indexes:
                static = block
                print "STATIC INDEX", block.label

        for i, block in enumerate(self.blocks):
            block.set_links([static])
        static.set_links([self.blocks[47]])

    def create_block_links2(self):
        # static_index = len(self.features) - 1
        #
        # for block in self.blocks:
        #     if static_index in block.indexes:
        #         static = block
        #         print "STATIC INDEX", block.label

        for i, block in enumerate(self.blocks):
            # block.set_links([static])
            distances = []
            indexes2 = range(len(self.blocks))
            indexes2.remove(i)

            for j in indexes2:
                block2 = self.blocks[j]
                distance = make_magnitude(block.top[0], block.top[1], block2.top[0], block2.top[1])
                distances.append((distance, j, block2))
                distances = sorted(distances, key=lambda t: t[0])
                block.set_links([link[2] for link in distances[0:8]])

        # static.set_links([self.blocks[47]])

    def update_blocks(self, points):
        # print "POINTS", len(points)
        for index, block in enumerate(self.blocks):
            try:
                block_points = points[self.labels.ravel() == index]
                block.set_points(block_points)
                block.calculate_distance()
            except Exception as e:
                print self.labels
                print len(self.blocks)
                print points
                print block
                raise e

    def read(self):
        self.current_frame += 1
        ret, image = self.cap.read()
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


        # affine = cv2.getAffineTransform(previous, self.features)
        H, status_homo = cv2.findHomography(previous, self.features, cv2.RANSAC, 1.0)
        # print status_homo
        h, w = self.image.shape[:2]
        print "COMPS:", getComponents(H)
        # print "!!!!!", pos
        # print self.features[0][0]
        #
        # self.features2 = cv2.perspectiveTransform(self.features, H)
        # # print self.features[0][0]
        # for p1,p2 in zip(self.features, self.features2):
        #     cv2.circle(self.image, (p2[0][0], p2[0][1]), 1, COLORS[0])
        #     cv2.circle(self.image, (p1[0][0], p1[0][1]), 1, COLORS[1])
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
        self.update_blocks(features)

        for i, f in enumerate(features):
            self.tracks[i].append(f)
            # ndx = [i for (i, st) in enumerate(status) if not st]
            # ndx.reverse()  # remove from back
            # for i in ndx:
            # self.tracks.pop(i)


    def track(self):
        """    Generator for stepping through a sequence."""
        count = 0
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

            count += 1
            if self.features == [] or (count % 800) == 0:
                self.create_blocks()
            else:
                # self.create_blocks()
                self.track_points()

            self.draw()
            self.dispatch()

    def draw(self):
        cv2.putText(self.image, str(self.current_frame) + "-" + str(round(len(self.blocks), 2)), (100, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[0])

        cv2.rectangle(self.image, (self.anchor[0], self.anchor[1]), (self.anchor[2], self.anchor[3]), COLORS[0])
        # cv2.rectangle(self.image, (self.initial_anchor[0], self.initial_anchor[1]), (self.initial_anchor[2], self.initial_anchor[3]), COLORS[0])
        # cv2.imshow("Anchor", self.anchor_data)
        for label, block in enumerate(self.blocks):
            # if label not in [54, 79]:
            # continue

            for index, angle, mag in block:
                point = self.features[index]
                # if index == len(self.features) - 1:
                # print "I FOUND", index, block.label, block.top, point
                cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, block.color, -1)

            v1, v2 = block.rect()
            cv2.rectangle(self.image, (v1[0], v1[1]), (v2[0], v2[1]), block.color)

            for link in block.links:
                cv2.line(self.image, link.top, block.top, block.color)
            # cv2.rectangle(self.image, (block.init_bbox[0], block.init_bbox[1]),
            # (block.init_bbox[2], block.init_bbox[3]), block.color)
            mark = "%s : %s %s" % (str(label), str(round(block.weight, 2)), str(len(block.links)))
            # mark = "%s : %s" % (str(label), str(block.delta_distance))
            cv2.putText(self.image, mark, (v2[0], v2[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, block.color)
            # for point in block.track:
            #     cv2.circle(self.image, (int(point[0]), int(point[1])), 3, block.color, -1)

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
            self.create_blocks()

    def create_links(self, points):
        length = len(points)
        self.average = 0
        self.averages = [i for i in range(length)]
        self.velocities = [[0, 0] for i in range(length)]
        indexes = [i for i in range(length)]
        self.links = [None for i in indexes]

        for i in indexes:
            distances = []
            point0 = points[i][0]
            # left = indexes[i:i-5:-1]
            # right = indexes[i:10-len(left):1]
            #
            # indexes2 = left + right
            indexes2 = range(length)
            indexes2.remove(i)
            for j in indexes2:
                point1 = points[j][0]
                distance = make_distance(point0, point1)
                distances.append((distance, j))
            distances = sorted(distances, key=lambda t: t[0])
            self.links[i] = [link[1] for link in distances[0:12]]
            # print i, self.links[i]

    def calculate_distances(self, points):
        self.average = 0
        for i, point, link in zip(range(len(points)), points, self.links):
            total = 0
            move = self.distances[i]
            for linked in link:
                point1 = points[linked]

                distance = make_distance(point[0], point1[0])
                total += distance

            move.append(total)
            way = sum(move) / len(link)
            timing = len(move)
            velocity = way / timing
            delta, last_velocity = self.velocities[i]
            if last_velocity != 0:
                new_delta = (abs(last_velocity - velocity) + delta) / 2
                self.velocities[i][0] = new_delta
                self.average += new_delta
                # if i == 249:
                # print i, velocity, self.velocities[i], new_delta
            # if self.velocities[i][1] == 0:
            self.velocities[i][1] = velocity

            acceleration = (2 * way) / (timing ** 2)
            # total = total/len(link)
            #
            # deltas = []
            # for k in range(len(move) - 1):
            # first = move[k]
            # second = move[k + 1]
            # deltas.append(abs(first - second))
            #
            # totals = sum(deltas)
            # print totals,self.current_frame, totals / self.current_frame, deltas
            # totals = sum(self.distances[i])
            # velocity = 2*totals / self.current_frame
            # acceleration = velocity / self.current_frame
            # self.averages[i] = velocity
            self.averages[i] = self.velocities[i][0]
            # print velocity,acceleration
        self.average /= len(points)


tracker = LKTracker(sys.argv[1])
tracker.track()