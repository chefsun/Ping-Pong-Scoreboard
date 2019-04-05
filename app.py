from collections import deque

import numpy as np
import cv2 as cv


gameState = ['unknown', 'newServe', 'onGame', 'dead']

BallPos = ['unknown', 'inAirOverTable', 'inAirBelowTable',
           'onGround', 'touchLeftTable', 'touchNet', 'touchRightTable']

_ball_buffer_size = 999999


class ScoreBoard:
    """Record/display points, serve reminder, and game point alert.

    TODO: Impl this class with OpenCV GUI.
    """

    def __init__(self):
        pass


class Ball:

    def __init__(self):

        self.traj = deque(maxlen=_ball_buffer_size)
        self.state = 'unknown'

    def update(self, fidx, x, y, h):

        self.traj.append((fidx, x, y, h))

        # TODO: Impl this method with the table geometry.
        self.state = BallPos[np.random.randint(len(BallPos))]

    def _clear(self):

        self.traj.clear()
        self.state = 'unknown'


class Game:

    def __init__(self, stream, ball, scoreboard, name='default'):

        self.stream = stream
        self.ball = ball
        self.scoreboard = scoreboard

        self.name = name
        self.state = 'unknown'

        # get the properties of video
        self.W = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.stream.get(cv.CAP_PROP_FPS))

        self.cnt = 0

        # self.params used to extract ball position: (x, y, h)
        pass

    def run(self):

        self._calibrate()

        while self.stream.isOpened():
            # get current frame
            success, frame = self.stream.read()
            if success:
                x, y, h = self._get_ball_pos(frame)
                self._update(self.cnt, x, y, h)

                self.cnt += 1

    def _calibrate(self):
        pass

    def _get_ball_pos(self, frame):
        return (0, 0, 0)

    def _update(self, fidx, x, y, h):

        self.ball.update(fidx, x, y, h)
        self._analyse()

    def _analyse(self):
        # TODO: Impl it with the game rules.
        pass

    def _clear(self):
        pass


# Init video stream
cap = cv.VideoCapture(0)
ball = Ball()
sb = ScoreBoard()

g = Game(cap, ball, sb)
g.run()
