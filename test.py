import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import pickle
import face_recognition
import cv2
import torch
from cvzone.FaceDetectionModule import FaceDetector
import numpy as np
import cvzone

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)

imgBackground = cv2.imread('assets/background.png')
folderModePath = 'assets/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))

def EncodeFiles():
    file = open('EncodeFile.p', 'rb')
    encodeListKnownWithIds = pickle.load(file)
    file.close()
    return encodeListKnownWithIds
known_face_encodings, known_face_names = EncodeFiles()

# def encode_faces():
# for image in os.listdir('faces'):
#     face_image = face_recognition.load_image_file(f'faces/{image}')
#     face_encoding = face_recognition.face_encodings(face_image)[0]        
#     known_face_encodings.append(face_encoding)
#     known_face_names.append(image)

def rescale_frame(frame):
    dim = (640, 480)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

# detect = FaceDetector()



Localurl = 'rtsp://admin:admin4763@192.168.5.190:554/'
resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
cap = cv2.VideoCapture(Localurl, cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(Localurl)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_GSTREAMER,1)
cap.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('M','J','P','G'))
# cap.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[0][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[0][1])
cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
cap.set(cv2.CAP_PROP_FPS, 10.0)
cap.set(cv2.CAP_PROP_FRAME_COUNT,1)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

while True:
    ret, frame= cap.read()
    # if  not ret:
    #     break
    # frame, boxs = detect.findFaces(frame)
    s_img = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_img = cv2.cvtColor(s_img, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_img)
    face_encodings = face_recognition.face_encodings(rgb_img, face_locations)

    img = rescale_frame(frame)
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]

    for encodeFace, faceLoc in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_face_encodings, encodeFace)
        distance = face_recognition.face_distance(known_face_encodings, encodeFace)
        mtchIndx = np.argmin(distance)
        if matches[mtchIndx]:
                # name= known_face_names[mtchIndx]
            y1, x2, y2, x1 = faceLoc
            # y1, x2, y2, x1 = y1 *4, x2 *4, y2 *4, x1 *4
            bbox = 55 + x1, 162 + y1, x2 - x1, y2 - y1
            print(known_face_names[mtchIndx])
            # imgBackground = cvzone.cornerRect(frame, bbox, 24,5,3,(200,10,10), (5,5,255))
            cvzone.cornerRect(img, bbox, rt=0)

        
    #     face_names = []
    #     for face_encoding in face_encodings:
    #         name = 'unknown'
    #         matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    #         distance = face_recognition.face_distance(known_face_encodings, face_encoding)
    #         matchIndex = np.argmin(distance)
    #         if matches[matchIndex] and distance[matchIndex] < 0.4:
    #             name= known_face_names[matchIndex]
    #         face_names.append(f'{name}')
            
    #         x, y , w, h = box['bbox']
    #         x2 = x + ((int(w) / 2) + 15)
    #         cvzone.putTextRect(frame, name, (x +20 , y + h + 25), 2, 2, (255, 255, 255), (10, 10, 10, 0.1),cv2.FONT_HERSHEY_PLAIN)
    #         cvzone.cornerRect(frame, (x, y, w, h), 24,5,3,(200,10,10), (5,5,255))
    
    # cv2.imshow('Face Recognition', frame)


    # img = rescale_frame(frame)
    # imgBackground[162:162+480, 55:55+640] = img
    # imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]
    cv2.imshow("fAce TRacker", imgBackground)
    # cv2.waitKey(1)
    k = cv2.waitKey(1) & 0Xff
    if k == 27: # Press 'ESC' to quit
        break
cv2.destroyAllWindows()