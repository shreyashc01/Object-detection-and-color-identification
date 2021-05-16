from tkinter import *
from tkinter import filedialog
import os
import cv2
import numpy as np
import pandas as pd

window = Tk()
window.title("Intershaala")
window.geometry("650x350")
window.configure(background='white')
window.resizable(0, 0)


def file_upload():
    answer = filedialog.askopenfile(parent=window,
                                    initialdir=os.getcwd(),
                                    title="Please select image/Video:")
    Label(window, text=answer.name, font=('Courier', 10), bg="white").place(x=50, y=130)
    return answer.name


def object_detection():
    cap = cv2.VideoCapture(file_upload())
    thres = 0.45
    nms_threshold = 0.2
    classFile = 'coco.names'
    with open(classFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)

    while True:
        success, img = cap.read()
        classIds, confs, bbox = net.detect(img, confThreshold=thres)
        bbox = list(bbox)
        confs = list(np.array(confs).reshape(1, -1)[0])
        confs = list(map(float, confs))

        indices = cv2.dnn.NMSBoxes(bbox, confs, thres, nms_threshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            cv2.rectangle(img, (x, y), (x + w, h + y), color=(255, 0, 0), thickness=2)
            cv2.putText(img, classNames[classIds[i][0] - 1].upper(), (box[0] + 10, box[1] + 30),
                        cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 255, 0), 2)
        img = cv2.resize(img, (500, 500))
        cv2.imshow('img', img)

        if cv2.waitKey(25) & 0xFF == 32:
            break
    cv2.destroyAllWindows()


def color_detection():
    img_path = file_upload()
    csv_path = 'colors.csv'

    index = ['color', 'color_name', 'hex', 'R', 'G', 'B']
    df = pd.read_csv(csv_path, names=index, header=None)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (800, 600))

    r = g = b = xpos = ypos = 0

    def get_color_name(R, G, B):
        minimum = 1000
        for i in range(len(df)):
            d = abs(R - int(df.loc[i, 'R'])) + abs(G - int(df.loc[i, 'G'])) + abs(B - int(df.loc[i, 'B']))
            if d <= minimum:
                minimum = d
                cname = df.loc[i, 'color_name']

        return cname

    clicked = False

    while True:
        def draw_function(event, x, y, flags, params):
            if event == cv2.EVENT_LBUTTONDBLCLK:
                global b, g, r, xpos, ypos, clicked
                clicked = True
                xpos = x
                ypos = y
                b, g, r = img[y, x]
                b = int(b)
                g = int(g)
                r = int(r)
                cv2.rectangle(img, (20, 20), (600, 60), (b, g, r), -1)
                text = get_color_name(r, g, b) + ' R=' + str(r) + ' G=' + str(g) + ' B=' + str(b)
                cv2.putText(img, text, (50, 50), 2, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
                if r + g + b >= 600:
                    cv2.putText(img, text, (50, 50), 2, 0.8, (0, 0, 0), 2, cv2.LINE_AA)

        cv2.namedWindow('image')
        cv2.setMouseCallback('image', draw_function)
        cv2.imshow('image', img)

        if cv2.waitKey(20) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


def help_detection():
    Label(window, text='Upload video or image for object detection', bg="white", font=('Courier', 11)).place(x=80,
                                                                                                             y=280)
    Label(window, text='Upload only image for color detection', bg="white", font=('Courier', 11)).place(x=80, y=300)
    Label(window, text='Press Esc to exit from color detection output window', bg="white", font=('Courier', 11)).place(
        x=80, y=320)


Label(window, text='Graduate Rotational Internship Program', bg="white", font=('Algerian', 20)).place(x=30, y=0)
Label(window, text='The Sparks Foundation', bg="white", font=('Algerian', 20)).place(x=150, y=40)
Label(window, text='ComputerVision & Internet of Things', bg="white", font=('Algerian', 20)).place(x=50, y=80)

Label(window, text='Path:-', font=('Courier', 10), bg="white").place(x=0, y=130)

Button(window, text="Object Detection           ", command=object_detection, font=('Courier', 15)).place(x=150, y=170)
Button(window, text="Color Identification       ", command=color_detection, font=('Courier', 15)).place(x=150, y=210)
Button(window, text="Help", command=help_detection, font=('Courier', 15)).place(x=0, y=300)

window.mainloop()
