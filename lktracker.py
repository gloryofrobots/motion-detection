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
    def detect_points(self):
        """    Detect 'good features to track' (corners) in the current frame
            using sub-pixel accuracy. """
        # load the image and create grayscale
        ret, self.image = self.cap.read()
        self.prev_gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        self.size[0] = np.size(self.image, 1)
        self.size[1] = np.size(self.image, 0)
        # search for good points
        self.features = cv2.goodFeaturesToTrack(self.prev_gray, **feature_params)

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
        print self.distances

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
        indexes = [i for i in range(length)]
        self.links = [None for i in indexes]

        for i in indexes:
            distances = []
            x1, y1 = points[i][0]
            indexes2 = range(length)
            indexes2.remove(i)
            for j in indexes2:
                x2, y2 = points[j][0]
                distance = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
                distances.append((distance,j))
            distances = sorted(distances, key=lambda t:t[0])
            self.links[i] = distances[0:3]

    def calculate_distances(self, points):
        for i in range(len(points) - 1):
            x1, y1 = points[i][0]
            x2, y2 = points[i + 1][0]
            distance = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            self.distances[i].append(distance)

    def next_features(self):
        ret, self.image = self.cap.read()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # reshape to fit input format
        previous = np.float32(self.features).reshape(-1, 1, 2)

        # calculate optical flow
        self.features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, previous, None,
                                                                      **lk_params)
        return previous,status

    def track_points(self):
        # load the image and create grayscale
        # print self.current_features
        # remove points lost
        self.next_features()
        self.calculate_distances()
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
        self.counter-=1
        """    Draw the current image with points using
            OpenCV's own drawing functions.
            Press ant key to close window."""


        add_next = False
        for point, track, move, index in zip(self.features, self.tracks, self.distances, range(len(self.features))):
            if index not in [251,252,253,268,269,270,271,485,486,487,488,0, 469, 470, 471,472]:
                continue
            if index == len(self.features) - 1:
                continue
            if len(track) == 1:
                cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
                continue
            if len(move) < 2:
                cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
                continue

            deltas = []
            for i in range(len(move)-1):
                first = move[i]
                second = move[i+1]
                deltas.append(abs(first-second))

            total = sum(deltas)
            average = total / len(deltas)
            average = abs(round(average, 2))
            add_next = True
            if add_next:
                cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
                cv2.putText(self.image, str(index) + ":" + str(average), (int(point[0][0]), int(point[0][1])),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
                add_next = False

            # if average > 0.5:
            #     cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
            #     cv2.putText(self.image, str(index) + ":" + str(average), (int(point[0][0]), int(point[0][1])),
            #                 cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))
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