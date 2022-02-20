import cv2
import numpy as np
import math

vfile = cv2.VideoCapture(1, cv2.CAP_DSHOW)


# -------------------------bird's eye view parameter, warp-------------------------------
height = 240
width = 320
up_left_x = 0
up_left_y = 120
pts1 = np.float32([(up_left_x, up_left_y), (width-up_left_x, up_left_y), (0, 180), (320, 180)])
pts2 = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
# ---------------------------------------------------------------------------------------
def get_fitline(img, f_lines):

    a = np.mean(f_lines, axis = 0)

    x1 = int(a[0])
    y1 = int(a[1])
    x2 = int(a[2])
    y2 = int(a[3])

    if(y1 >= y2):
        y1 = 240
        y2 = 150
    else:
        y2 = 240
        y1 = 150

    result = [x1, y1, x2, y2]

    return result


def draw_fitline(img, lines, color=[255, 255, 255], thickness=7):  # 대표선 그리기
    cv2.line(img, (lines[0], lines[1]), (lines[2], lines[3]), color, thickness)

def left_line_top(lines):
    if (lines[0] >= lines[2]):
        return lines[0]
    else:
        return lines[2]

def right_line_top(lines):
    if (lines[0] <= lines[2]):
        return lines[0]
    else:
        return lines[2]

def left_line_bot(lines):
    if (lines[0] >= lines[2]):
        return lines[2]
    else:
        return lines[0]

def right_line_bot(lines):
    if (lines[0] <= lines[2]):
        return lines[2]
    else:
        return lines[0]

if vfile.isOpened():
    while True:
        vret, img = vfile.read()

        dst = cv2.resize(img, dsize=(320, 240), interpolation=cv2.INTER_AREA)   # image resize

        warp = cv2.getPerspectiveTransform(pts1, pts2)
        ROI = cv2.warpPerspective(dst, warp, (width, height))
        grayimg = cv2.cvtColor(ROI, cv2.COLOR_BGR2GRAY) # if you wanna use cropped image instead of warped one, replace ROI to ROI2
        blurimg = cv2.GaussianBlur(grayimg, (5, 5), 1) # kernel size 5x5, standard deviation
        cannyimg = cv2.Canny(blurimg, 50, 200)
        hough_img = cv2.HoughLinesP(cannyimg, 1, 1 * np.pi/180, 30, np.array([]), 80, 20) # img, rho, theta, threshold, min_line_len, max_line_gap

        line_arr = np.squeeze(hough_img)

        print(line_arr)

        L_lines = []
        R_lines = []

        # for i in line_arr:
        #     print(i)
        #     if (i[2] <= 160 and i[0] <= 160):
        #         L_lines.append(i)
        #     else:
        #         R_lines.append(i)

        for i in line_arr:
            if (i[1] < i[3]):
                outterx = i[2]
            else:
                outterx = i[0]

            if  (outterx <= 160):
                L_lines.append(i)
            else:
                R_lines.append(i)

        L_lines = np.array(L_lines)
        R_lines = np.array(R_lines)

        print(L_lines)
        print(R_lines)

        temp =  np.zeros((cannyimg.shape[0], cannyimg.shape[1], 3), dtype = np.uint8)

        # 왼쪽, 오른쪽 각각 대표선 구하기
        if len(L_lines) == 0:
            left_fitline = [0, 150, 0, 240]
        else:
            left_fitline = get_fitline(cannyimg, L_lines)

        if len(R_lines) == 0:
            right_fitline = [320, 150, 320, 240]
        else:
            right_fitline = get_fitline(cannyimg, R_lines)

        left_top_x = left_line_top(left_fitline)
        right_top_x = right_line_top(right_fitline)

        left_bot_x = left_line_bot(left_fitline)
        right_bot_x = right_line_bot(right_fitline)

        centerxtop = int((left_top_x + right_top_x) / 2)
        centerxbot = int((left_bot_x + right_bot_x) / 2)
        centerlist = [centerxtop, 150, centerxbot, 240]

        print(left_fitline)
        print(right_fitline)

        # 대표선 그리기
        draw_fitline(temp, left_fitline)
        draw_fitline(temp, right_fitline)
        cv2.line(temp, (centerlist[0], centerlist[1]), (centerlist[2], centerlist[3]), [0, 255, 0], 7)

        result = cv2.addWeighted(temp, 1, ROI, 1., 0.)

        if vret:
            cv2.imshow('webcam', img)
            cv2.imshow('Resize', dst)
            cv2.imshow('ROI', ROI)
            cv2.imshow('Canny', cannyimg)
            cv2.imshow('Result', result)

        if cv2.waitKey(70) == ord('q'):
            break

        print("\n\n\n\n")
    else:
        print('Frame is abnormal.')
else:
    print('Unable to open file.')

vfile.release()
cv2.destroyAllWindows
