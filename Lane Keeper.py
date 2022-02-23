import cv2
import numpy as np
import math
import RPi.GPIO as GPIO


from gpiozero import Servo
from gpiozero import Motor

# ---------------------------------camera calibration------------------------------------
mtx = np.array([[675.36635728,   0.,         333.38639369],
                [  0.,         703.53240948, 248.24724652],
                [  0.,           0.,           1.        ]])
dist = np.array([[ 0.15717633, -1.16912606, -0.00236505,  0.00283506,  1.29273117]])
newcameramtx = np.array([[649.50244141,   0.,         336.09837819],
                         [  0.,         671.5791626,  246.96360265],
                         [  0.,           0.,           1.        ]])
# -------------------------bird's eye view parameter, warp-------------------------------
height = 240
width = 320
up_left_x = 55
up_left_y = 80
pts1 = np.float32([(up_left_x, up_left_y), (width-up_left_x, up_left_y), (20, 180), (300, 180)])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])

line_top = 130
# -------------------------functions to get line coordinates-----------------------------

def get_fitline(img, f_lines):

    a = np.mean(f_lines, axis = 0)

    x1 = int(a[0])
    y1 = int(a[1])
    x2 = int(a[2])
    y2 = int(a[3])

    if(y1 >= y2):
        y1 = 240
        y2 = line_top
    else:
        y2 = 240
        y1 = line_top

    result = [x1, y1, x2, y2]

    return result

def left_line_top(lines):
    if(len(lines) == 0):
        return 0
    else:
        return lines[0]
def right_line_top(lines):
    if (len(lines) == 0):
        return 0
    else:
        return lines[0]

def left_line_bot(lines):
    if (len(lines) == 0):
        return 0
    else:
        return lines[2]
def right_line_bot(lines):
    if (len(lines) == 0):
        return 0
    else:
        return lines[2]

motor = Motor(forward=20, backward=21)
GPIO.setmode(GPIO.BCM)
GPIO.setup(17, GPIO.OUT)

p = GPIO.PWM(17, 50)
p.start(7.1)

vfile = cv2.VideoCapture(0)

if vfile.isOpened():
    while True:
        motor.forward(speed=0.7)
        vret, img = vfile.read()
        undistort = cv2.undistort(img, mtx, dist, None, newcameramtx)
        blurimg = cv2.GaussianBlur(undistort, (5, 5), 1)  # kernel size 5x5, standard deviation
        resize = cv2.resize(blurimg, dsize=(320, 240), interpolation=cv2.INTER_AREA)

        ROI = cv2.warpPerspective(resize, cv2.getPerspectiveTransform(pts1, pts2), (width, height))
        cannyimg = cv2.Canny(ROI, 80, 200)
        hough_img = cv2.HoughLinesP(cannyimg, 0.8, 1 * np.pi / 180, 30, np.array([]), 50, 15)  # img, rho, theta, threshold, min_line_len, max_line_gap

        line_arr = np.squeeze(hough_img)

        print(line_arr)


        for i in line_arr:
            if (i[1] > i[3]):
                i[1], i[3] = i[3], i[1]
                i[0], i[2] = i[2], i[0]

        # print(line_arr)

        L_lines = []
        R_lines = []

        for i in line_arr:

            if(i[2] <= 160):
                L_lines.append(i)
            else:
                R_lines.append(i)

        L_lines = np.array(L_lines)
        R_lines = np.array(R_lines)

        if len(L_lines) == 0:
            left_fitline = [0, line_top, 0, 240]
        else:
            left_fitline = get_fitline(cannyimg, L_lines)

        if len(R_lines) == 0:
            right_fitline = [320, line_top, 320, 240]
        else:
            right_fitline = get_fitline(cannyimg, R_lines)

        # print(left_fitline)
        # print(right_fitline)

        left_top_x = left_line_top(left_fitline)
        right_top_x = right_line_top(right_fitline)

        left_bot_x = left_line_bot(left_fitline)
        right_bot_x = right_line_bot(right_fitline)

        centerxtop = int((left_top_x + right_top_x) / 2)
        centerxbot = int((left_bot_x + right_bot_x) / 2)
        center = [centerxtop, line_top, centerxbot, 240]

        # print(center)

        theta1 = (180 / np.pi) * math.atan2(240 - line_top, centerxtop - centerxbot) - 90

        if (theta1 > 30):
            theta1 = 30

        if (theta1 < -30):
            theta1 = -30

        # print(theta1)

        duty = (( 2.0 * theta1+90)/(180))*10 +2.1

        if (duty > 10.8):
            duty = 10.8
        if (duty < 3.4):
            duty = 3.8

        temp = np.zeros((cannyimg.shape[0], cannyimg.shape[1], 3), dtype=np.uint8)

        cv2.line(temp, (left_top_x, line_top), (left_bot_x, 240), [0, 255, 255], 7)
        cv2.line(temp, (right_top_x, line_top), (right_bot_x, 240), [0, 255, 255], 7)
        cv2.line(temp, (center[0], center[1]), (center[2], center[3]), [0, 255, 255], 7)
        result = cv2.addWeighted(temp, 1, ROI, 1., 0.)

        print(duty)
        p.ChangeDutyCycle(duty)

        if vret:
        #     cv2.imshow('webcam', img)
        #     cv2.imshow('Resize', resize)
        #     cv2.imshow('blur', blurimg)
        #     cv2.imshow('ROI', ROI)
        #     cv2.imshow('Canny', cannyimg)
            cv2.imshow('Result', result)
            cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break

        print("\n")

    else:
        print('Frame is abnormal.')
else:
    print('Unable to open file.')

vfile.release()
cv2.destroyAllWindows