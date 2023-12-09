import os, sys
import numpy as np
import math
import torch
from threading import Thread
from time import sleep
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import face_recognition
import cv2
import asyncio
import multiprocessing as mp
from collections import deque
import pickle
from cvzone.FaceDetectionModule import FaceDetector
import cvzone

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
        
        # p = mp.Process(target=self.queryframe, args=())
        # p.start()
        
    def start(self):
        t1 = Thread(target=self.queryframe,daemon=True ,args=())
        t1.start()
        
    def stop(self):
        self.isstop = True

    def getframe(self):
        return self.Frame.pop()

    def queryframe(self):
        while (not self.isstop):
            self.status, tmp = self.capture.read()
            self.Frame.append(tmp)

        self.capture.release()


def EncodeFiles():
    file = open('EncodeFile.p', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    return encodeListKnownWithIds

encodeListKnown, EmployeeIds = EncodeFiles()

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
    img = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (0, 0), 0.15, 0.15)
    faceCurFrame = face_recognition.face_locations(img)
    encodeCurFrame = face_recognition.face_encodings(img, faceCurFrame)
    for encodeFace, faceLoc in zip(encodeCurFrame, faceCurFrame):
        mtchs = face_recognition.compare_faces(encodeListKnown, encodeFace)
        fceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
        mtchIndx = np.argmin(fceDis)
        if fceDis[mtchIndx] < 0.4 and mtchs[mtchIndx]:
            return EmployeeIds[mtchIndx]


def main():
    global x, y, w, h , x2
    
    cam = camCapture(0, buffer_size=2)
    cam.start()
    
    sleep(1)
    detector = FaceDetector()
            
    while True:
        frame = cam.getframe()
        frame, bboxs = detector.findFaces(frame)
        for box in bboxs:
            emplyee = EncodeImg(frame)
            if emplyee is not None and emplyee is not int(0):
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
        
        frm = ResizeWithAspectRatio(frame, width=1024)
        cv2.imshow("Real-time Detection", frm)
        k = cv2.waitKey(30) & 0Xff
        if k == 27: # Press 'ESC' to quit
            cam.stop()
            break
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
    
    # t = Thread(target=fr.encode_faces, daemon=True, args=())
    # t.start()