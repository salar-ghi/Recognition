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
# from facenet_pytorch import InceptionResnetV1, MTCNN

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)

# mtcnn = MTCNN(keep_all=True, device=device)

# # Define Inception Resnet V1 module
# model = InceptionResnetV1(pretrained='vggface2').eval().to(device)


def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val  +((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    
rtspurl = 'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
Localurl = 'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl = 'http://192.168.10.226:80/video'
darourl = 'rtsp://admin:admin1234@192.168.16.252:554'
resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]

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

global known_face_encodings, known_face_names
class FaceRecognition():
    face_locations = []
    face_encodings = []
    face_name = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True

    
    def __init__(self):
        torch.set_default_device("cuda:0")
        
        # self.encode_faces()
        
        # self.cam = cv2.VideoCapture(Localurl ,cv2.CAP_DSHOW,(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY))
        self.cam = cv2.VideoCapture(Localurl)
        # self.cam.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_NONE)
        self.cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.cam.set(cv2.CAP_GSTREAMER,1)
        self.cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('m','j','p','g'))
        self.cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('M','J','P','G'))
        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[0][0])
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[0][1])
        self.cam.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        self.cam.set(cv2.CAP_PROP_FPS, 5.0)
        self.cam.set(cv2.CAP_PROP_FRAME_COUNT,1)
        self.cam.set(cv2.CAP_PROP_POS_FRAMES,1)
        
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
     
    async def run_recognition(self):
        if not self.cam.isOpened():
            sys.exit('video source not found...')
            
        print('start steaming')
        while True:
            ret, frame = self.cam.read()

            # boxes, probs = mtcnn.detect(frame)

            # if boxes is not None:
            #     for box in boxes:
            # boxes
            # await asyncio.sleep(0.03)
            if self.process_current_frame:
                
                s_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb_s_frame = cv2.resize(s_frame, (0, 0), fx=0.25, fy=0.25)
                # rgb_s_frame = s_frame[:, :, ::-1]
                

                self.face_locations = face_recognition.face_locations(rgb_s_frame)
                self.face_encodings = face_recognition.face_encodings(rgb_s_frame, self.face_locations)
                
                self.face_names = []
                for face_encoding in self.face_encodings:
                    matches = face_recognition.compare_faces(self.known_face_encodings, face_encoding)
                    name = 'Unknown'
                    confidence = 'Unknown'
                    
                    face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    
                    if matches[best_match_index]:
                        name = self.known_face_names[best_match_index]
                        confidence = face_confidence(face_distances[best_match_index])
                        
                    self.face_names.append(f'{name} ({confidence})')
            
            self.process_current_frame = not self.process_current_frame
            
            # Display annotations
            for (top, right, bottom, left), name in zip(self.face_locations, self.face_names):
                top *= 4
                right *= 4
                bottom *= 4
                left *= 4
                
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), -1)
                cv2.putText(frame, name, (left + 6, bottom -6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)
                
            frm = ResizeWithAspectRatio(frame, width=1024)
            cv2.imshow('Face Recognition', frm)
            # cv2.waitKey(1)
            if cv2.waitKey(1) == ord('q'):
                break
            
        self.cam.release()
        cv2.destroyAllWindows()
                    
      

if __name__ == '__main__':
    p1 = mp.Process(target=FaceRecognition, args=())
    p1.start()
    fr = FaceRecognition()
    t = Thread(target=fr.encode_faces, daemon=True, args=())
    t.start()
    t.join()
    t1 = Thread(target=asyncio.run(fr.run_recognition()), daemon=True, args=())
    t1.start()
    # asyncio.run(fr.run_recognition())