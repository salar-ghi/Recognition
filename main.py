import face_recognition
import os, sys
import cv2
import numpy as np
import math
import torch

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.set_default_device(device)
# torch.device(device)



def face_confidence(face_distance, face_match_threshold=0.6):
    range = (1.0 - face_match_threshold)
    linear_val = (1.0 - face_distance) / (range * 2.0)
    
    if face_distance > face_match_threshold:
        return str(round(linear_val * 100, 2)) + '%'
    else:
        value = (linear_val  +((1.0 - linear_val) * math.pow((linear_val - 0.5) * 2, 0.2))) * 100
        return str(round(value, 2)) + '%'
    
rtspurl =  'rtsp://admin:ndcndc@192.168.10.226:554/channel1'
Localurl =  'rtsp://admin:admin4763@192.168.5.190:554/'
httpurl =  'http://192.168.10.226:80/video'
darourl =  'rtsp://admin:admin1234@192.168.16.252:554'
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

class FaceRecognition():
    face_locations = []
    face_encodings = []
    face_name = []
    known_face_encodings = []
    known_face_names = []
    process_current_frame = True
    
    def __init__(self):
        # torch.set_default_device("cuda:0")
        self.encode_faces()
        #encode faces
        
    def encode_faces(self):
        for image in os.listdir('faces'):
            face_image = face_recognition.load_image_file(f'faces/{image}')
            face_encoding = face_recognition.face_encodings(face_image)[0]
            
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(image)
            
        print(self.known_face_names)
      
    def run_recognition(self):
        # torch.set_default_device("cuda:0")
        # cam = cv2.VideoCapture(0, cv2.CAP_GSTREAMER)
        cam = cv2.VideoCapture(0, cv2.CAP_PROP_BUFFERSIZE)
        cam.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        cam.set(cv2.CAP_GSTREAMER,1)
        cam.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[1][0])
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[1][1])
        cam.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
        cam.set(cv2.CAP_PROP_FPS, 10)
        cam.set(cv2.CAP_PROP_FRAME_COUNT,1)
        cam.set(cv2.CAP_PROP_POS_FRAMES,1)
        
        if not cam.isOpened():
            sys.exit('video source not found...')
            
            
        while True:
            ret, frame = cam.read()
            
            if self.process_current_frame:
                s_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
                # rgb_s_frame = s_frame[:, :, ::-1]
                rgb_s_frame = cv2.cvtColor(s_frame, cv2.COLOR_BGR2RGB)
                
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
                
            # frm = ResizeWithAspectRatio(frame, width=1280)
            cv2.imshow('Face REcognition', frame)
            cv2.waitKey(1)
            if cv2.waitKey(1) == ord('q'):
                break
            
        cam.release()
        cv2.destroyAllWindows()
                    
      
      
        
if __name__ == '__main__':
    fr = FaceRecognition()
    fr.run_recognition()