from cvzone.FaceDetectionModule import FaceDetector
from threading import Thread

class Detector:
    def __init__(self):
        Thread(target=self.getDetector(), daemon=True, args=())

    def getDetector(self):
        return FaceDetector()

    
        