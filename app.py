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

    def __init__(self, bH=1080, bW=1920):

        cv.startWindowThread()
        cv.namedWindow('Scoreboard')

        self.H = bH
        self.W = bW
        self.halfW = self.W // 2

        # style
        self.blue = (126, 77, 36)
        self.red = (125, 126, 246)
        self.font = cv.FONT_HERSHEY_SIMPLEX
        self.fontsize = 24
        self.fontthickness = 64

        # BG
        self.plain = np.zeros((self.H, self.W, 3), np.uint8)
        self.plain[:, 0:self.halfW] = self.red
        self.plain[:, self.halfW:] = self.blue

        # score
        self.LHS = 0
        self.RHS = 0

        self.curr = None
        self._draw()

    def _draw(self):
        self.curr = self.plain.copy()
        if self.LHS < 10:
            self.curr = cv.putText(self.curr, str(self.LHS), (self.W//8, self.H*3//4), fontFace=self.font,
                                   fontScale=self.fontsize, color=(255, 255, 255), thickness=self.fontthickness)
        else:
            self.curr = cv.putText(self.curr, str(self.LHS//10), (0, self.H*3//4), fontFace=self.font,
                                   fontScale=self.fontsize, color=(255, 255, 255), thickness=self.fontthickness)
            self.curr = cv.putText(self.curr, str(self.LHS % 10), (self.W//4, self.H*3//4), fontFace=self.font,
                                   fontScale=self.fontsize, color=(255, 255, 255), thickness=self.fontthickness)
        if self.RHS < 10:
            self.curr = cv.putText(self.curr, str(self.RHS), (self.W*5//8, self.H*3//4), fontFace=self.font,
                                   fontScale=self.fontsize, color=(255, 255, 255), thickness=self.fontthickness)
        else:
            self.curr = cv.putText(self.curr, str(self.RHS//10), (self.W//2, self.H*3//4), fontFace=self.font,
                                   fontScale=self.fontsize, color=(255, 255, 255), thickness=self.fontthickness)
            self.curr = cv.putText(self.curr, str(self.RHS % 10), (self.W*3//4, self.H*3//4), fontFace=self.font,
                                   fontScale=self.fontsize, color=(255, 255, 255), thickness=self.fontthickness)

    def update(self, winside):

        if winside == 'L':
            self.LHS += 1
        elif winside == 'R':
            self.RHS += 1
        else:
            raise ValueError("illegal update")

        self._draw()

        if self.LHS >= 11 or self.RHS >= 11:
            _delta = abs(self.LHS - self.RHS)
            if _delta >= 2:
                self.curr = cv.putText(self.curr, str(winside+' Wins!'), (self.H//3, self.H*2//3), fontFace=self.font,
                                       fontScale=self.fontsize//2, color=(0, 215, 255), thickness=self.fontthickness)
                self.reset()

    def display(self):
        cv.imshow('Scoreboard', self.curr)
        cv.waitKey(1)

    def reset(self):
        self.LHS = 0
        self.RHS = 0


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
            self.scoreboard.display()
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
