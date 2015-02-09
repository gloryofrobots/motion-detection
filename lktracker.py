import numpy as np
import time
from itertools import izip, islice
import cv2
import sys
import math
# some constants and default parameters
lk_params = dict(winSize=(15, 15), maxLevel=5,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

subpix_params = dict(zeroZone=(-1, -1), winSize=(10, 10),
                     criteria=(cv2.TERM_CRITERIA_COUNT | cv2.TERM_CRITERIA_EPS, 20, 0.03))

feature_params = dict(maxCorners=500, qualityLevel=0.01, minDistance=10)

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

def make_distance(point0, point1):
    x1, y1 = point0
    x2, y2 = point1
    return math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)


class LKTracker(object):
    """    Class for Lucas-Kanade tracking with
        pyramidal optical flow."""

    def __init__(self, filename):
        """    Initialize with a list of image names. """
        self.filename = filename
        self.features = []
        self.tracks = []
        self.cap = cv2.VideoCapture()
        self.cap.open(filename)
        self.pause = False
        self.trace_tracks = False
        self.distances = []
        self.prev_gray = None
        self.image = None
        self.gray = None
        self.counter = 20
        self.size = [0, 0]
        self.status = []
        self.links = []
        self.current_frame = 0

    def detect_points(self):
        """    Detect 'good features to track' (corners) in the current frame
            using sub-pixel accuracy. """
        # load the image and create grayscale
        self.read()
        self.prev_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.size[0] = np.size(self.image, 1)
        self.size[1] = np.size(self.image, 0)
        # search for good points
        self.features = cv2.goodFeaturesToTrack(self.prev_gray, **feature_params)
        self.current_frame = 0
        # refine the corner locations
        cv2.cornerSubPix(self.prev_gray, self.features, **subpix_params)
        # self.cut_homography()

        # print features
        # m = 0
        # n = 0
        def comparator(p0, p1):
            x1, y1 = p0[0]
            x2, y2 = p1[0]
            if y1 == y2:
                if x1 > x2:
                    return 1
                elif x2 < x1:
                    return -1
                return 0
            if y1 > y2:
                return 1
            elif y1 < y2:
                return -1

        def sorter(p):
            x, y = p[0]
            return math.sqrt(((self.size[0] - x) ** 2) + ((self.size[1] - y) ** 2))

        self.features = sorted(self.features, key=sorter)
        self.tracks = [[p[0]] for p in self.features]
        self.distances = [[] for i in range(len(self.features))]
        self.status = np.ones((len(self.features), 1))

        self.create_links(self.features)
        self.calculate_distances(self.features)
        # print self.distances

    def read(self):
        self.current_frame += 1
        ret, self.image = self.cap.read()
        return ret

    def cut_homography(self):
        print "!!!!!!!!!!!!!!!!!"
        print len(self.features)
        previous, status = self.next_features()
        # print previous, self.features
        H, self.status = cv2.findHomography(previous, self.features, cv2.RANSAC, 1)
        self.features = [p for (st, p) in zip(self.status, self.features) if st]
        print len(self.features)

    def create_links(self, points):
        length = len(points)
        self.average = 0
        self.averages = [i for i in range(length)]
        self.velocities = [[0,0] for i in range(length)]
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
                #     print i, velocity, self.velocities[i], new_delta
            # if self.velocities[i][1] == 0:
            self.velocities[i][1] = velocity


            acceleration = (2 * way) / (timing**2)
            # total = total/len(link)
            #
            # deltas = []
            # for k in range(len(move) - 1):
            #     first = move[k]
            #     second = move[k + 1]
            #     deltas.append(abs(first - second))
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
    def next_features(self):
        self.read()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # reshape to fit input format
        previous = np.float32(self.features).reshape(-1, 1, 2)

        # calculate optical flow
        self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, previous, None,
                                                                      **lk_params)
        return previous, status

    def track_points(self):
        # load the image and create grayscale
        # print self.current_features
        # remove points lost
        self.next_features()
        self.calculate_distances(self.features)
        # self.features = [p for (st, p) in zip(status, self.current_features) if st]

        # # clean tracks from lost points
        features = np.array(self.features).reshape((-1, 2))
        for i, f in enumerate(features):
            self.tracks[i].append(f)
        # ndx = [i for (i, st) in enumerate(status) if not st]
        # ndx.reverse()  # remove from back
        # for i in ndx:
        # self.tracks.pop(i)

        self.prev_gray = self.gray

    def track(self):
        """    Generator for stepping through a sequence."""
        count = 0
        while True:
            if self.pause:
                # time.sleep(1)
                self.dispatch()
                continue
            count += 1
            if self.features == [] or (count % 800) == 0:
                self.detect_points()
            else:
                self.track_points()

            # create a copy in RGB
            # f = array(self.features).reshape(-1,2)
            # im = cv2.cvtColor(self.image,cv2.COLOR_BGR2RGB)
            # yield im,f

            self.draw()
            self.dispatch()

    def draw(self):
        self.counter -= 1
        """    Draw the current image with points using
            OpenCV's own drawing functions.
            Press ant key to close window."""

        cv2.putText(self.image, str(self.current_frame) + "-" + str(round(self.average,2)), (100,100),
                    cv2.FONT_HERSHEY_SIMPLEX, 2, COLORS[0])
        for point, track, move, index in zip(self.features, self.tracks, self.distances, range(len(self.features))):
            # if index not in [249, 251, 252, 253, 268, 269, 270, 271, 485, 486, 487, 488, 0, 469, 470, 471, 472]:
            #     continue
            # if index < 100 or index > 270:
            #     continue
            if index == len(self.features) - 1:
                continue
            if len(track) == 1:
                cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
                continue
            if len(move) < 2:
                cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
                continue

            average = self.averages[index]
            # print average
            # print self.averages
            # total = sum(move)
            # average = total / self.current_frame
            # average = total
            # deltas = []
            # for i in range(len(move) - 1):
            #     first = move[i]
            #     second = move[i + 1]
            #     deltas.append(abs(first - second))
            #
            # total = sum(deltas)
            #  average = total / len(deltas)
            average = abs(round(average, 2))
            # print index, average, self.average, self.averages[index]
            check = average - self.average
            if check > 0.02:
                color = COLORS[0]
            else:
                color = COLORS[1]

            cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, color, -1)
            cv2.putText(self.image, str(index) + ":" + str(average), (int(point[0][0]), int(point[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color)

            # if average > 0.5:
            #     cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
            #     cv2.putText(self.image, str(index) + ":" + str(average), (int(point[0][0]), int(point[0][1])),
            #                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
            #     add_next = True

            if self.trace_tracks:
                print index, total, average
                print track
                print deltas
                print min(move), max(move)

        cv2.imshow('LKtrack', self.image)

    def dispatch(self):
        ch = 0xFF & cv2.waitKey(10)
        # ch = cv2.waitKey(0)
        if ch == ord(' '):
            self.pause = not self.pause
        if ch == ord('t'):
            self.trace_tracks = not self.trace_tracks
        if ch == ord('r'):
            self.detect_points()


tracker = LKTracker(sys.argv[1])
tracker.track()