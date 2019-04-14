## TODO:
# ball guess for each frame - pick detection closest to guess
# confidence for bounces, can be updated with more frame history

from collections import deque
from enum import Enum, auto
from dataclasses import dataclass

import numpy as np
import cv2 as cv
from matplotlib import colors

#gameState = ['unknown', 'newServe', 'onGame', 'dead']
class gameState(Enum):
    UNKNOWN = auto()        # unknown state
    NEWPOINT = auto()       # waiting for someone to serve
    CHECKSERVE = auto()     # checking for a correct serve
    INPLAY = auto()         # point is being played (checking for end of point)

BallPos = ['unknown', 'inAirOverTable', 'inAirBelowTable',
           'onGround', 'touchLeftTable', 'touchNet', 'touchRightTable']

_ball_buffer_size = 999999
_possible_traj_size = 5

_ball_width = 0.040 # 40mm
fps = 1

class ScoreBoard:
    """Record/display points, serve reminder, and game point alert.

    TODO: Impl this class with OpenCV GUI.
    """

    def __init__(self):
        pass

@dataclass
class balldet:
    x: float
    y: float
    vel: float
    angle: float
    size_px: float
    # vx, vy instead of vel/angle ?

class Ball:

    def __init__(self):

        self.possible_traj = np.ndarray(_possible_traj_size, dtype=object)
        for i in range(_possible_traj_size):
            self.possible_traj[i] = deque(maxlen=_ball_buffer_size)
        self.best_traj_ix = -1
        self.traj = deque(maxlen=_ball_buffer_size)
        self.state = 'unknown'
        self.guess = balldet(300, 300, 0.5, 90, 50) # initial guess for ball position

    def update(self, fidx, balldet):

        self.traj.append((fidx, balldet))
        
        # update guess
        # TODO: average over a few past trajectory points, or use parabolic fit
        # TODO: determine sign of velocities based on past trajectory, or at worst, which end of table it's on
        a = self.traj[-1][1].angle
        vi = self.traj[-1][1].vel
        xi = self.traj[-1][1].x
        yi = self.traj[-1][1].y
        vx = vi*np.sin(a)
        vy = vi*np.cos(a)
        px_m = balldet.size_px / _ball_width
        
        print("last point: ", self.traj[-1][1])
        
        self.guess.x = xi + vx/fps*px_m
        self.guess.y = yi + vy/fps*px_m
        self.guess.angle = a
        self.guess.vel = vi
        
        print("guess: ", self.guess)
        
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
        self.state = gameState.NEWPOINT

        # get the properties of video
        self.W = int(self.stream.get(cv.CAP_PROP_FRAME_WIDTH))
        self.H = int(self.stream.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.FPS = int(self.stream.get(cv.CAP_PROP_FPS))
        global fps
        fps = self.FPS
        print("Source FPS: ", self.FPS)
        
        self.cnt = 0
        
        # save last 5 frames
        self.fbuf = np.ndarray(5, dtype=object)
        self.bufpos = -1

        # self.params used to extract ball position: (x, y, h)
        pass

    def run(self):

        self._calibrate()
        
        cv.namedWindow("view", cv.WINDOW_NORMAL)
        cv.resizeWindow("view", 1200,720)
        #cv.namedWindow("difference", cv.WINDOW_NORMAL)
        #cv.resizeWindow("difference", 1200,720)

        while self.stream.isOpened():
            # get current frame
            success, frame = self.stream.read()
            if success:
                newpos = (self.bufpos + 1) % 5
                self.fbuf[newpos] = cv.extractChannel(frame, 2) #cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   #frame.copy()
                self.bufpos = newpos
                
                balls = self._get_ball_pos(frame)
                self._update(self.cnt, balls)
                
                # testing: draw possible balls, selected ball, and prediction (guess)
                for b in balls:
                    cv.circle(frame, (int(round(b.x)), int(round(b.y))), int(round(b.size_px/2)), [0,200,200], 1, cv.LINE_AA)
                if len(self.ball.traj) > 0:
                    cv.circle(frame, (int(round(self.ball.traj[-1][1].x)), int(round(self.ball.traj[-1][1].y))), 5, [100,255,0], -1, cv.LINE_AA)
                cv.circle(frame, (int(round(self.ball.guess.x)), int(round(self.ball.guess.y))), 5, [0,100,255], -1, cv.LINE_AA)
            
                cv.imshow("view", frame)
                cv.waitKey(10)
                self.cnt += 1
            key = cv.waitKey(0) & 0xFF
            if key == ord('q'):
                break

    def _calibrate(self):
        # fill up the buffer of frames
        for i in range(0,5):
            success, frame = self.stream.read()
            if success:
                newpos = (self.bufpos + 1) % 5
                self.fbuf[newpos] = cv.extractChannel(frame, 2) #cv.cvtColor(frame, cv.COLOR_BGR2GRAY)   #frame.copy()
                self.bufpos = newpos

    def _get_ball_pos(self, frame):
        
        possible_balls = deque(maxlen=10)
        
        #hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        #h = hsv[:,:,0]
        #s = hsv[:,:,1]
        #colmask = ~(((hsv[:,:,0]>165)+(hsv[:,:,0]<15))*(hsv[:,:,1]>80))
        #frame[colmask] = [0,0,0]
        
        #framebw = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        
        diff1 = cv.absdiff(self.fbuf[self.bufpos], self.fbuf[(self.bufpos-3)%5])
        diff2 = cv.absdiff(self.fbuf[self.bufpos], self.fbuf[(self.bufpos-1)%5])
        mask = (diff1 > 7) * (diff2 > 7)  # previously used red channel instead of grayscale here
        
        diffmask = np.zeros((self.H, self.W), dtype=np.uint8)
        diffmask[mask] = 255
        
        el = cv.getStructuringElement(cv.MORPH_ELLIPSE, (9,9))
        diffmask = cv.erode(diffmask, el)
        diffmask = cv.dilate(diffmask, el)
        
        contours, hierarchy = cv.findContours(diffmask,cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
        
        #cv.drawContours(frame, contours, -1, (0,0,0), 1)
        
        for c in contours:            
            a_cont = cv.contourArea(c)
            per = cv.arcLength(c, True)
            
            # this should be calibrated at beginning
            if a_cont < 700:    
                # draw magenta to indicate filtered out by area
                #cv.drawContours(frame, [c], 0, (255,0,255), 1)
                continue
                
            # check convexity
            hull = cv.convexHull(c)
            a_hull = cv.contourArea(hull)
            #cv.putText(frame, str(round(a_hull/a_cont,2)), (int(c[0,0,0]),int(c[0,0,1])), cv.FONT_HERSHEY_PLAIN, 1, (100,50,50), 1, cv.LINE_AA)
            if a_hull/a_cont > 1.15:
                #cv.drawContours(frame, [c], 0, (50,150,0), 1)
                continue
            
            circularity = a_cont / (per**2/(16*np.pi))
            if circularity < 0.9:
                # draw blue to indicate filtered out by circularity
                #cv.drawContours(frame, [c], 0, (255,100,0), 1)
                continue
            
            if c.shape[0] >= 5:
                (x,y),(MA,ma),angle = cv.fitEllipse(c)
                #print("x: ", x, "y: ", y, "MA: ", MA, "ma: ", ma, "angle: ", angle, "\n")
                
                # filter by ratio of minor to major axis - should be between 1 and 5 (6 to be safe)
                ar = ma/MA
                if ar < 1 or ar > 6:
                    # draw red to indicate filtered by aspect ratio
                    #cv.drawContours(frame, [c], 0, (0,50,255), 1)
                    continue
                    
                # filter by minimum dimension (ball width)
                # this should be in a certain range for min and max distance to ball
                if MA < 30 or MA > 66:
                    # draw orange to indicate filtered by width
                    #cv.drawContours(frame, [c], 0, (0,0,0), 1)
                    #cv.putText(frame, str(round(MA,1)), (int(x),int(y)), cv.FONT_HERSHEY_PLAIN, 1, (0,150,255), 1, cv.LINE_AA)
                    continue
                
                # filter by color (hue and saturation):
                mask = np.zeros(diffmask.shape,np.uint8)
                cv.drawContours(mask,[c],0,255,-1)
                maskpoints = cv.findNonZero(mask)
                vals = np.ndarray((maskpoints.shape[0], 3), np.uint8)
                for i in range(0,maskpoints.shape[0]):
                    vals[i,:] = frame[maskpoints[i,0,1], maskpoints[i,0,0], :]
                #  calculate median value of the current contour region in original image
                median = np.median(vals, axis = 0)
                # change to rgb for conversion function
                swap = median[2]
                median[2] = median[0]
                median[0] = swap
                hsv = 255*colors.rgb_to_hsv(median/255)
                sat = hsv[1]
                hue = hsv[0]
                # sat between 55, 111
                # hue between 11, 103
                # median: 
                    # sat 37 65 144
                    # hue 251 1 14 21
                if (hue < 238 and hue > 28) or (sat < 30 or sat > 220):
                    # draw contour black and print hue/sat to indicate filtered by hsv
                    #cv.drawContours(frame, [c], 0, (0,0,0), 1)
                    #cv.putText(frame, str(int(sat)), (int(x),int(y)), cv.FONT_HERSHEY_PLAIN, 1.3, (255,255,0), 1, cv.LINE_AA)
                    #cv.putText(frame, str(int(hue)), (int(x),int(y)+10), cv.FONT_HERSHEY_PLAIN, 1.3, (0,255,255), 1, cv.LINE_AA)
                    continue
        
                # draw green to indicate remaining candidate balls:
                cv.drawContours(frame, [c], 0, (80, 255, 0), 2)
                possible_balls.append(balldet(x, y, (ma/MA-1)*_ball_width*self.FPS, angle, MA))
                #cv.putText(frame, str(round(a_cont,0)), (int(x),int(y)), cv.FONT_HERSHEY_PLAIN, 1, (100,50,50), 1, cv.LINE_AA)
            
        # TODO: try using matchShape with a straight blur or bounce blur template
        # try filtering on aspect ratio, max dimension, area
        # use ellipse fit to get angle / direction
        # filter by hue / color (chroma key)
        # convexity / convexity defects
        
        #cv.imshow("contours", frame)
        #cv.imshow("difference", diff2)
        #cv.waitKey(15)
        
        #print("\n")
        
        return possible_balls

    def _update(self, fidx, balls):
        if len(balls) == 0:
            return
            
        ## ROBUST METHOD
        # see if this ball matches any existing possible trajectory
        # pick the best match
        #   or
        # for each possible trajectory, choose the possible ball closest to its prediction
        # if it's too far out of predicted range, don't update it. keep track of stale frame count for trajectory
        
        # if trajectory is too stale (# of frames w/o update), remove it
            
        ## SIMPLE METHOD
        # choose the ball closest to the current prediction
        dist2 = np.ndarray(len(balls), dtype=object)
        ix = 0
        for b in balls:
            print(b)
            dist2[ix] = (b.x - self.ball.guess.x)**2 + (b.y - self.ball.guess.y)**2
            ix = ix + 1
        
        imin = np.argmin(dist2)
        self.ball.update(fidx, balls[imin])
        
        print("\n")
        
        #self.ball.update(fidx, balls)
        self._analyse()

    def _analyse(self):
        # TODO: Impl it with the game rules.
        pass

    def _clear(self):
        pass

# templates
temp1 = cv.imread("template1.png", cv.IMREAD_GRAYSCALE)
temp2 = cv.imread("template2.png", cv.IMREAD_GRAYSCALE)
temp1c,_ = cv.findContours(temp1, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)
temp2c,_ = cv.findContours(temp2, cv.RETR_TREE,cv.CHAIN_APPROX_SIMPLE)

# Init video stream
cap = cv.VideoCapture("/run/media/erich/CORSAIR/school/mini_side.mp4")
ball = Ball()
sb = ScoreBoard()

g = Game(cap, ball, sb)
g.run()
