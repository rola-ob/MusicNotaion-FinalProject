import math
import cv2
import mediapipe as mp
import numpy as np
import ctypes
from tkinter import Tk
from tkinter.filedialog import askopenfilename

yClick = 0


def empty(a):
    pass


def click_event(event, x, y, flags, param):
    global yClick
    if event == cv2.EVENT_LBUTTONDOWN:
        yClick = y
    return 0


def make_vector(lines_1):
    newlines = []
    for line_1 in lines_1:
        r, theta = line_1[0]
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a * r
        y0 = b * r
        newlines.append([int(x0 + 1000 * (-b)), int(y0 + 1000 * a), int(x0 - 1000 * (-b)), int(y0 - 1000 * a)])
    return newlines
# make_vector get the output list of houghlines transform and return a new list containing actual points


def bottom_line(lines_3):
    for line_3 in lines_3:
        if (line_3[1] > 130) and (line_3[1] < 170):
            return line_3[1]
    return 165
# bottom_line get a list of detected horizontal lines and return the closest line to the bottom of the white keys


def delete_extra_lines(lines_4):
    # assuming that the gap between two keys is 3, and we have 52 keys, then: (1280-(53*3))/52 =~ 21
    # 20 is the width of a single key
    d = -20
    c = 0
    deleted_index = [-1]
    for line_4 in lines_4:
        if line_4[0] - d < 20:
            deleted_index.append(c)
        else:
            d = line_4[0]
        c += 1
    deleted_index.reverse()
    for i in deleted_index:
        if i == -1:
            break
        lines_4.pop(i)
    return lines_4
# delete_extra_lines deletes all lines within 20 pixels of the first line then moves to the next one and do the same


def right_left(borders_):
    left_ = borders_[0][2]
    for border in borders_:
        empty(0)
    right_ = border[0]
    return right_, left_
# right_left returns the right and left borders of the keys


def start_frame(frames_, btm):
    tip_1 = tip_2 = tip_3 = tip_4 = tip_5 = btm + 1
    for frame_1 in frames_:
        if frame_1[0] == 4:
            tip_1 = frame_1[2]
        if frame_1[0] == 8:
            tip_2 = frame_1[2]
        if frame_1[0] == 12:
            tip_3 = frame_1[2]
        if frame_1[0] == 16:
            tip_4 = frame_1[2]
        if frame_1[0] == 20:
            tip_5 = frame_1[2]
        if tip_1 < btm and tip_2 < btm and tip_3 < btm and tip_4 < btm and tip_5 < btm:
            return frame_1[3]
# start_frame returns the frame where all the fingers of one of the hands are above the bottom line


def end_frame(frames_, start, btm):
    tip_1 = tip_2 = tip_3 = tip_4 = tip_5 = 0
    for frame_2 in frames_:
        if frame_2[3] < start:
            continue
        if frame_2[0] == 4:
            tip_1 = frame_2[2]
        if frame_2[0] == 8:
            tip_2 = frame_2[2]
        if frame_2[0] == 12:
            tip_3 = frame_2[2]
        if frame_2[0] == 16:
            tip_4 = frame_2[2]
        if frame_2[0] == 20:
            tip_5 = frame_2[2]
        if (tip_1 > btm) and (tip_2 > btm) and (tip_3 > btm) and (tip_4 > btm) and (tip_5 > btm):
            return frame_2[3]
# end_frame returns the frame where all the fingers of one of the hands are goes below the bottom line


def distance(finger_):
    distance_list = []
    for point_ in finger_:
        if point_[0] % 2 == 1:
            p1 = [point_[1], point_[2]]
            continue
        p2 = [point_[1], point_[2]]
        d = math.dist(p1, p2)
        distance_list.append([round(d), point_[3], point_[1], point_[2]])
    return distance_list
# distance converts the list of finger points by frame into length of the finger by frame, by calculating the distance
# of two points


def pressed_keys(finger_1, min_distance):
    flag = False
    pressed = []
    for line_d in finger_1:
        if line_d[0] < min_distance:
            if not flag:
                flag = True
                pressed.append([line_d[1], line_d[2], line_d[3]])
        else:
            flag = False
    return pressed
# pressed_keys returns a list containing all the frames where the finger pressed


def clear_frames(lst):
    threshold_error = 25
    new_lst = [lst[0]]
    for elem in lst:
        if elem[0] - new_lst[-1][0] > threshold_error:
            new_lst = new_lst + [elem]
    return new_lst
# clear_frames deletes 25 consecutive frames, cause we need just one frame to know that a key is pressed


def find_key(x, y):
    if y > btm_ln:
        return chr(64)
    i = 0
    for line_5 in vertical_lines:
        if x < line_5[0]:
            i -= 1
            break
        i += 1
    octave = (i // 7) + 1
    i %= 7
    if i == 0 or i == 1:
        octave -= 1
    if y > black_line:
        i += 65
        return chr(i)+str(octave)
    else:
        if i == 2 or i == 6:
            i -= 1
        i += 65
        return chr(i)+str(octave)+'s'
# find_key gets point returns the name of the key


def find_note(arr_frame):
    notation = []
    for frame_3 in arr_frame:
        notation.append([frame_3[0], find_key(frame_3[1], frame_3[2])])
    return notation
# find_note returns a list of frames and keys names


def delete_extra_notes(notes_):
    delete_index = [-1]
    i = 0
    for note_ in notes:
        if note_[1] == '@':
            delete_index.append(i)
        i += 1
    delete_index.reverse()
    for j in delete_index:
        if j == -1:
            break
        notes_ = np.delete(notes_, j, 0)
    return notes_
# delete_extra_notes delete all the lines where the name of the note is @, we choose @ if we detected a pressed key
# out of range


def color_key(notes_1):
    colors = []
    x1, x2 = 0, 0
    for note_1 in notes_1:
        y2 = btm_ln
        curr_frame = int(note_1[0])
        sharp_flag = False
        if len(note_1[1]) == 3:
            sharp_flag = True
        octave = ord(note_1[1][1]) - 48
        key = ord(note_1[1][0]) - 64
        if key > 2:
            octave -= 1
        line_num = (octave * 7) + key - 1
        if sharp_flag:
            y2 = black_line
            line_num += 1
        for line_5 in vertical_lines:
            if line_num == 0:
                x1 = line_5[0]
            if line_num == -1:
                x2 = line_5[0]
                break
            line_num -= 1
        k = curr_frame
        for i in range(30):
            colors.append([k, x1, x2, y2])
            k += 1
    colors.sort(key=lambda x: x[0])
    return colors
# color_key returns a list of frame and position to color


resizeWidth, resizeHeight = 1280, 400
desiredWidth, desiredHeight = 1280, 300
kernel = np.ones((5, 5), np.uint8)
Tk().withdraw()
filename = askopenfilename()
ctypes.windll.user32.MessageBoxW(0, "please click on the top border of the keys,\n"
                                    "    after that press any key to continue", "before we start", 0)
cap = cv2.VideoCapture(filename)
success, img = cap.read()
img = cv2.resize(img, (resizeWidth, resizeHeight))
cv2.imshow("Video", img)
cv2.setMouseCallback("Video", click_event)
cv2.waitKey(0)
cv2.destroyWindow("Video")
# transformation to cut the upper side of the piano
# pts1 is the point on the input image, pts2 is the output image
pts1 = np.float32([[0, yClick], [resizeWidth, yClick], [0, resizeHeight], [resizeWidth, resizeHeight]])
pts2 = np.float32([[0, 0], [desiredWidth, 0], [0, desiredHeight], [desiredWidth, desiredHeight]])
matrix = cv2.getPerspectiveTransform(pts1, pts2)
img_warp = cv2.warpPerspective(img, matrix, (desiredWidth, desiredHeight))
# detecting lines of the keys to get the right border and the left border of the keys
gray = cv2.cvtColor(img_warp, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 1)
canny = cv2.Canny(blur, 90, 110)
borders = cv2.HoughLines(canny, 1, np.pi, 45)
borders_vector = make_vector(borders)
borders_vector.sort(key=lambda x: x[0])
right, left = right_left(borders_vector)
# transformation to cut the edges of the piano
pts12 = np.float32([[left, 0], [right, 0], [left, desiredHeight], [right, desiredHeight]])
pts22 = np.float32([[0, 0], [desiredWidth, 0], [0, desiredHeight], [desiredWidth, desiredHeight]])
matrix2 = cv2.getPerspectiveTransform(pts12, pts22)
# colored image of the piano in final form
img_no_canny = cv2.warpPerspective(img_warp, matrix2, (desiredWidth, desiredHeight))
# the canny image of the piano in final form
gray2 = cv2.cvtColor(img_no_canny, cv2.COLOR_BGR2GRAY)
blur2 = cv2.GaussianBlur(gray2, (5, 5), 1)
img_final = cv2.Canny(blur2, 90, 110)
# hough lines detecting the horizontal lines in order to find the bottom border of the piano keys
imgDilate = cv2.dilate(img_final, kernel, iterations=1)
imgErode = cv2.erode(imgDilate, kernel, iterations=1)
white_keys_border = cv2.HoughLines(imgErode, 1, (np.pi / 180), 500)
horizontal_lines = make_vector(white_keys_border)
horizontal_lines.sort(key=lambda x: x[1])
btm_ln = bottom_line(horizontal_lines)
# hough lines detecting the vertical lines of the keys
vertical_lines = make_vector(cv2.HoughLines(imgErode, 1, np.pi, 45))
vertical_lines.sort(key=lambda x: x[0])
vertical_lines = delete_extra_lines(vertical_lines)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils
# after we got the bottom line and the right left edges we wrap the image by 4 points
pts12 = np.float32([[left, yClick], [right, yClick], [left, resizeHeight], [right, resizeHeight]])
pts22 = np.float32([[0, 0], [desiredWidth, 0], [0, desiredHeight], [desiredWidth, desiredHeight]])
matrix2 = cv2.getPerspectiveTransform(pts12, pts22)
# Black keys usually have a visible length of about 9 cm.
# White keys usually have a visible length of about 15 cm.
# The length of a black key is gonna be (btm_ln)*(9/15)
black_line = btm_ln * 0.6
hand_landmarks = []
f = 0
dis = 0
press_flag = 0
# f is a frame counter
while True:
    success, img = cap.read()
    if success != 1:
        break
    f += 1
    imgResize = cv2.resize(img, (resizeWidth, resizeHeight))
    imgOutPut = cv2.warpPerspective(imgResize, matrix2, (desiredWidth, desiredHeight))
    # getting image ready for hand detection
    imgRGB = cv2.cvtColor(imgOutPut, cv2.COLOR_BGR2RGB)
    results = hands.process(imgRGB)
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id, lm in enumerate(handLms.landmark):
                cx, cy = int(lm.x * desiredWidth), int(lm.y * desiredHeight)
                hand_landmarks.append([id, cx, cy, f])
            # print lines on image
            for line in vertical_lines:
                cv2.line(imgOutPut, (line[0], 0), (line[2], btm_ln), (0, 255, 0), 2)
            cv2.line(imgOutPut, (0, btm_ln), (1280, btm_ln), (0, 0, 255), 2)
            cv2.line(imgOutPut, (0, int(black_line)), (1280, int(black_line)), (255, 0, 255), 2)
            # draw hand landmarks on image
            mpDraw.draw_landmarks(imgOutPut, handLms, mpHands.HAND_CONNECTIONS)
    cv2.putText(imgOutPut, str(int(f)), (10, 70), cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    cv2.imshow("Video", imgOutPut)
    cv2.waitKey(1)
cv2.destroyWindow("Video")
# now we delete all frames out of the range of the start_frame and end_frame
start_f = start_frame(hand_landmarks, btm_ln)
end_f = end_frame(hand_landmarks, start_f, btm_ln)
hand_landmarks = list(filter(lambda x: (x[3] > start_f) and (x[3] < end_f), hand_landmarks))
# we delete all the points but the points of the beginning and the tip of each finger
thumb = list(filter(lambda x: (x[0] == 1) or (x[0] == 4), hand_landmarks))
index = list(filter(lambda x: (x[0] == 5) or (x[0] == 8), hand_landmarks))
middle = list(filter(lambda x: (x[0] == 9) or (x[0] == 12), hand_landmarks))
ring = list(filter(lambda x: (x[0] == 13) or (x[0] == 16), hand_landmarks))
pinky = list(filter(lambda x: (x[0] == 17) or (x[0] == 20), hand_landmarks))
# converting the (x,y) points into distances
thumb_distance = distance(thumb)
index_distance = distance(index)
middle_distance = distance(middle)
ring_distance = distance(ring)
pinky_distance = distance(pinky)
# our video 80 - 75 - 87 - 73 - 50
# these numbers are chosen to fit with this specific hands movement in this specific video
pressed_thumb = pressed_keys(thumb_distance, 80)
pressed_index = pressed_keys(index_distance, 75)
pressed_mid = pressed_keys(middle_distance, 87)
pressed_ring = pressed_keys(ring_distance, 73)
pressed_pinky = pressed_keys(pinky_distance, 50)
# we have to change the list to numpy array in order to use clear_frames
pressed_thumb = clear_frames(pressed_thumb)
pressed_index = clear_frames(pressed_index)
pressed_mid = clear_frames(pressed_mid)
pressed_ring = clear_frames(pressed_ring)
pressed_pinky = clear_frames(pressed_pinky)
notes = [[100, '@']]
# notes contains names of all pressed keys
notes = np.concatenate((notes, find_note(pressed_thumb)))
notes = np.concatenate((notes, find_note(pressed_index)))
notes = np.concatenate((notes, find_note(pressed_mid)))
notes = np.concatenate((notes, find_note(pressed_ring)))
notes = np.concatenate((notes, find_note(pressed_pinky)))
notes = delete_extra_notes(notes)
list_notes = notes.tolist()
# sort the notes in order by frame
list_notes.sort(key=lambda x: x[0])
cap = cv2.VideoCapture(filename)
to_be_colored = color_key(list_notes)
t = 0
while True:
    success, img = cap.read()
    if success != 1:
        break
    t += 1
    imgResize = cv2.resize(img, (resizeWidth, resizeHeight))
    imgOutPut = cv2.warpPerspective(imgResize, matrix2, (desiredWidth, desiredHeight))
    for frame_ in to_be_colored:
        if int(frame_[0]) < t:
            continue
        if int(frame_[0]) > t:
            break
        cv2.rectangle(imgOutPut, (int(frame_[1]), 0), (int(frame_[2]), int(frame_[3])), (110, 70, 45), cv2.FILLED)
    cv2.imshow("colored keys", imgOutPut)
    cv2.waitKey(1)
# print notes to file
with open('OutPutNote.txt', 'w') as file:
    for note in list_notes:
        file.write(note[1])
        file.write(' ')
