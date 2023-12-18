import os, sys
import numpy as np
import math
import torch
from threading import Thread
from time import sleep
import asyncio
import multiprocessing as mp
from collections import deque
import pickle
from dataQuery import SqlQueries
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"

from cvzone.FaceDetectionModule import FaceDetector
import face_recognition
import cvzone
import cv2

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)

rtspurl = 'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
Localurl = 'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl = 'http://192.168.10.226:80/video'
darourl = 'rtsp://admin:admin1234@192.168.16.252:554'
resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]

class camCapture:
    def __init__(self, camID, buffer_size):
        torch.set_default_device("cuda:0")
        # self.Frame = deque(maxlen=buffer_size)
        self.Frame = deque(maxlen=buffer_size)
        self.status = False
        self.isstop = False
        self.capture = cv2.VideoCapture(camID)
        self.capture.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.capture.set(cv2.CAP_GSTREAMER,1)
        self.capture.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('m','j','p','g'))
        self.capture.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('M','J','P','G'))
        self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[1][0])
        self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[1][1])
        self.capture.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        self.capture.set(cv2.CAP_PROP_FPS, 5.0)
        self.capture.set(cv2.CAP_PROP_FRAME_COUNT,1)
        self.capture.set(cv2.CAP_PROP_POS_FRAMES,1)
        
        t1 = Thread(target=self.queryframe, daemon=True, args=())
        t1.start()
        
        
    def stop(self):
        self.isstop = True

    def getframe(self):
        return self.Frame.popleft()

    def queryframe(self):
        while (not self.isstop):
            self.status, tmp = self.capture.read()
            if not self.status:
                break
            self.Frame.append(tmp)

        self.capture.release()

def EncodeFiles():
    print('start to decode file')
    file = open('EncodeFile.p', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    return encodeListKnownWithIds




def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    r = 0

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

def EncodeImg(source):
    img = cv2.resize(source , (0, 0),None, 0.15, 0.15)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        mtchs = face_recognition.compare_faces(encodeListKnown, encodeFace)
        fceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        mtchIndx = np.argmin(fceDis)
        if fceDis[mtchIndx] < 0.4 and mtchs[mtchIndx]:
            return EmployeeIds[mtchIndx]

cam = camCapture(Localurl, buffer_size=10000)
# sleep(0.030)
encodeListKnown, EmployeeIds = EncodeFiles()

class Detector:
    def __init__(self):
        print('start to get FaceDetector')
        Thread(target=self.getDetector, args=()).start()

    def getDetector(self):
        return FaceDetector()

imgBackground = cv2.imread('assets/background.png')
folderModePath = 'assets/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

def rescale_frame(frame):
    dim = (640, 480)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

def main():
    global x, y, w, h , x2
    
    # detector = Thread.threading(target=FaceDetector, args=()) 
    detect = Detector()
    # detector = FaceDetector()
    detector = detect.getDetector()
    while True:
        frame = cam.getframe()
        frame, bboxs = detector.findFaces(frame)
        for box in bboxs:
            emplyee = EncodeImg(frame)
            if emplyee is not None and emplyee is not int(0):
                
                ####################
                query = SqlQueries(emplyee)
                query.CheckAttandance(emplyee)


                ####################

                x, y, w, h = box['bbox']
                x2 = x + (int(w) / 2)
                cvzone.putTextRect(frame, f'{emplyee}', (int(x2+45), y-10),2, 1, (0, 255, 25),(10, 10, 10, 0.1), cv2.BORDER_TRANSPARENT,1, 1)
                cvzone.cornerRect(frame, (x, y, w, h))
            else:
                x1, y1, w1, h1 = box['bbox']
                x3 = x1 + (int(w1) / 2)
                txt ="Unknown"
                cvzone.putTextRect(frame, f'{txt}', (int(x3+15), y1-15),2, 1, (0, 0, 255),(255, 255, 255, 0.9), cv2.BORDER_TRANSPARENT,1, 1)
                cvzone.cornerRect(frame, (x1, y1, w1, h1))
        
        # img = ResizeWithAspectRatio(frame, width=640)
        img = rescale_frame(frame)
        imgBackground[162:162+480, 55:55+640] = img
        imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]
        cv2.imshow("Real-time Detection", imgBackground)
        # cv2.waitKey(0)
        k = cv2.waitKey(30) & 0Xff
        if k == 27: # Press 'ESC' to quit
            cam.stop()
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    print('start process')
    p = mp.Process(target=EncodeFiles, args=())    
    p.start()
    p.join()

    print('start Thread')    
    t3 = Thread(target=main, args=()) 
    t3.start()
    
    
    # t = Thread(target=fr.encode_faces, daemon=True, args=())
    # t.start()