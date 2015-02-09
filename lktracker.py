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

    def detect_points(self):
        """    Detect 'good features to track' (corners) in the current frame
            using sub-pixel accuracy. """
        # load the image and create grayscale
        ret, self.image = self.cap.read()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # search for good points
        features = cv2.goodFeaturesToTrack(self.gray, **feature_params)

        # refine the corner locations
        cv2.cornerSubPix(self.gray, features, **subpix_params)

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
            return math.sqrt(x ** 2 + y ** 2)

        sf = sorted(features, key=sorter)
        print "!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!"
        for i, p in enumerate(sf):
            print i, p[0]
        self.features = sf
        self.tracks = [[p[0]] for p in self.features]
        self.distances = [[] for i in range(len(self.features))]

        self.calculate_distances()
        print self.distances
        self.prev_gray = self.gray

    def calculate_distances(self):
        points = self.features
        for i in range(len(points) - 1):
            x1, y1 = points[i][0]
            x2, y2 = points[i + 1][0]
            distance = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            self.distances[i].append(distance)

    def track_points(self):
        # load the image and create grayscale
        ret, self.image = self.cap.read()
        self.gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # reshape to fit input format
        tmp = np.float32(self.features).reshape(-1, 1, 2)

        # calculate optical flow
        self.current_features, status, track_error = cv2.calcOpticalFlowPyrLK(self.prev_gray, self.gray, tmp, None,
                                                                              **lk_params)
        # print self.current_features
        # remove points lost
        self.features = self.current_features
        # self.features = [p for (st, p) in zip(status, self.current_features) if st]

        # # clean tracks from lost points
        features = np.array(self.current_features).reshape((-1, 2))
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
        """    Draw the current image with points using
            OpenCV's own drawing functions.
            Press ant key to close window."""



        # draw points as green circles
        for point, track, index in zip(self.features, self.tracks, range(len(self.features))):
            if len(track) == 1:
                cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
                continue

            # pairs = izip(track[::2], track[1::2])
            # total = 0
            # for pair in pairs:
            # x1,y1 = pair[0]
            # x2, y2 = pair[1]
            # # distance = y2 - y1 + x2 - x1
            #     distance = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            #     angle = math.atan2(y1 - y2, x1 - x2)
            #     criteria = distance
            #     total += criteria
            # print pairs
            # print track
            x1, y1 = track[0]
            x2, y2 = track[len(track) - 1]
            # angle = math.atan2(y1 - y2, x1 - x2)
            total = math.sqrt((y1 - y2) ** 2 + (x1 - x2) ** 2)
            count = 1
            average = round(total / count, 1)
            if self.trace_tracks:
                print index, total, average
                print track
            # print (x1,y1), (x2,y2), angle, len(track)
            #cv2.polylines(self.image, [int32(track)], False, (255, 255, 255))
            cv2.circle(self.image, (int(point[0][0]), int(point[0][1])), 3, (0, 255, 0), -1)
            cv2.putText(self.image, str(index) + ":" + str(average), (int(point[0][0]), int(point[0][1])),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0))

        cv2.imshow('LKtrack', self.image)

    def dispatch(self):
        ch = 0xFF & cv2.waitKey(10)
        ch = cv2.waitKey(0)
        if ch == ord(' '):
            self.pause = not self.pause
        if ch == ord('t'):
            self.trace_tracks = not self.trace_tracks
        if ch == ord('r'):
            self.detect_points()


tracker = LKTracker(sys.argv[1])
tracker.track()