import os
import numpy as np
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import pickle
import torch
from threading import Thread
from dataQuery import SqlQueries
# import dataQuery

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
import face_recognition
import cvzone
import cv2



device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)
torch.cuda.init()

rtspurl = 'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
Localurl = 'rtsp://admin:admin4763@192.168.5.190:554/sub'
# Localurl = 'rtsp://admin:admin4763@192.168.5.190/554/sub'
httpurl = 'http://192.168.10.226:80/video'
darourl = 'rtsp://admin:admin1234@192.168.16.252:554'
resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]

folderPath = 'faces'
pathList = os.listdir(folderPath)

class Vision:
    def __init__(self, camID):
        self.camId = camID
        self.imgBackground = cv2.imread('assets/background.png')
        self.imgEmployee = []
        self.modeType = 0
        self.counter = 0
        self.id = -1
        self.result = -1
        self.num = 0
        self.empQueu = set() 
    
    def back_box(self):
        # imgBackground = cv2.imread('assets/background.png')
        folderModePath = 'assets/Modes'
        modeList = os.listdir(folderModePath)
        imgModeList = []
        for path in modeList:
            imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
        return imgModeList

    def EncodeFiles(self):
        file = open('EncodeFile.p', 'rb')
        encodeListKnownWithIds = pickle.load(file)
        file.close()
        return encodeListKnownWithIds
    
    def imgsEncoded(self):
        tEncode = Thread(target=self.EncodeFiles, daemon= True, args=())
        tEncode.start()

    def rescale_frame(self, frame):
        dim = (640, 480)
        return cv2.resize(frame, dim, interpolation= cv2.INTER_AREA)
    
    def findEncoding(images):
        encodeList = []
        for img in images:
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            location = face_recognition.face_locations(img)[0]
            encode = face_recognition.face_encodings(location)
            encodeList.append(encode)
            return encodeList

    def EncodeFrames(self, frame):
        img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_location = face_location = face_recognition.face_locations(rgb_img)
        face_encoding = face_encoding = face_recognition.face_encodings(rgb_img, face_location)
        for face_encoding in self.face_encodings:
            matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
            name = 'Unknown'
            confidence = 'Unknown'

    # @lru_cache(maxsize=128)
    def processInputGate(self):
        known_face_encodings, EmployeeIds = self.EncodeFiles()
        # self.capture = cv2.VideoCapture(self.camId, cv2.CAP_FFMPEG)
        cam = cv2.VideoCapture(0)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cam.set(cv2.CAP_GSTREAMER,1)
        cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('m','j','p','g'))
        cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('M','J','P','G'))
        cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[0][0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[0][1])
        cam.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        cam.set(cv2.CAP_PROP_FPS, 5.0)
        cam.set(cv2.CAP_PROP_EXPOSURE, -5)
        
        while True:
            self.counter = 0
            ret, frame = cam.read()
            img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            # rgb_img = img
            
            face_location = face_location = face_recognition.face_locations(rgb_img)
            face_encoding = face_encoding = face_recognition.face_encodings(rgb_img, face_location)

            
            self.imgBackground[162:162 + 480, 55:55 + 640] = frame
            # self.imgBackground[44:44 + 633, 808:808 + 414] = self.back_box()[self.modeType]

            for face_encode, faceLoc in zip(face_encoding, face_location):
                matches = face_recognition.compare_faces(known_face_encodings, face_encode)
                face_distance = face_recognition.face_distance(known_face_encodings, face_encode)
                best_match = np.argmin(face_distance)
                if matches[best_match]:
                    self.empQueu.add(EmployeeIds[best_match])                  
                    y1, x2 ,y2, x1 = faceLoc
                    y1, x2 ,y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 *4
                    bbox = 55 + x1, 162 +y1, x2 - x1, y2 - y1
                    self.imgBackground = cvzone.cornerRect(self.imgBackground, bbox, rt=0)
                    self.id = EmployeeIds[best_match]
                    if self.counter == 0:
                        self.counter = 1
                        self.modeType = 1
                        print('first cond',self.counter)
                
            
            if self.counter != 0:
                if self.counter == 1:
                    query = SqlQueries(self.id)
                    self.result = query.CheckAttandance()
                    if self.result == 1:                    
                        self.imgEmployee = cv2.imread(f'{folderPath}/{self.id}.jpg')
                        self.modeType = 1
                        self.imgBackground[44:44 + 633, 808:808 + 414] = self.back_box()[self.modeType]
                    elif self.result == 3:
                        self.num +=1
                        self.imgEmployee = cv2.imread(f'{folderPath}/{self.id}.jpg')
                        if self.num >= 5:
                            self.modeType = 2
                            self.imgBackground[44:44 + 633, 808:808 + 414] = self.back_box()[self.modeType]

                    #-------------------------------------update data of attandance#-------------------------------------
                if self.result == 1:
                    cv2.putText(self.imgBackground, str(self.counter), (861, 125), cv2.FONT_HERSHEY_COMPLEX, 1,(10,10,10),1)
                    cv2.putText(self.imgBackground, str('Role'), (1006, 550), cv2.FONT_HERSHEY_COMPLEX, 0.7(10,10,15),1)
                    cv2.putText(self.imgBackground, str(self.id), (1006, 493), cv2.FONT_HERSHEY_COMPLEX, 0.7,(10,10,10),1)

                    # cv2.putText(self.imgBackground, str(self.id), (910, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,(100,100,100),1)
                    # cv2.putText(self.imgBackground, str(self.id), (1025, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,(100,100,100),1)
                    # cv2.putText(self.imgBackground, str(self.id), (1125, 625), cv2.FONT_HERSHEY_COMPLEX, 0.6,(100,100,100),1)

                    (w, h), _ = cv2.getTextSize(str('salar ghahremani'), cv2.FONT_HERSHEY_COMPLEX, 1, 1)
                    offset = (414-w)//2
                    cv2.putText(self.imgBackground, str('salar ghahremani'), (858+offset, 445), cv2.FONT_HERSHEY_COMPLEX, 0.5,(50,50,50),1)

                self.imgEmployee = cv2.resize(self.imgEmployee, (216, 216), interpolation= cv2.INTER_AREA)
                self.imgBackground[175:175 + 216, 909:909 + 216] = self.imgEmployee

                # self.counter+1
            elif self.counter == 0:
                self.empQueu.clear()
                # self.imgBackground[175:175 + 216, 909:909 + 216] = self.imgEmployee
                self.modeType = 4
                self.imgBackground[44:44 + 633, 808:808 + 414] = self.back_box()[self.modeType]
                
                # self.imgBackground = cv2.imread('assets/background.png')
                # self.imgBackground[175:175 + 216, 909:909 + 216] = None


            print(self.empQueu)
            cv2.imshow("Real-Time Detection", self.imgBackground)
            # cv2.imshow("Real-Time Detection", frame)

            k = cv2.waitKey(30) & 0Xff
            if k == 27:
                cam.release()
                break
        cv2.destroyAllWindows()



if __name__ == '__main__':
    print('app is running')
    vis = Vision(camID=0)
    # vis.start()
    vis.imgsEncoded()
    Thread(target=vis.back_box, daemon=True, args=()).start()

    vis.processInputGate()