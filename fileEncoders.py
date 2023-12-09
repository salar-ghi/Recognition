import pickle
import os
import face_recognition
import cv2
from numba import jit, cuda, njit, numba
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.device('cuda')

folderPath = 'faces'
pathList = os.listdir(folderPath)
# print(pathList)

imgList = []
EmployeeIds = []


for item in pathList:
    imgList.append(cv2.imread(os.path.join(folderPath, item)))
    EmployeeIds.append(os.path.splitext(item)[0])


# @njit(nopython=False, target_backend='cuda', parallel = True, fastmath = True)
def findEncodings(imagesList):
    encodeList = []
    for img in imagesList:
        img = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
    return encodeList

print("Encoding Started ... ")
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, EmployeeIds]
print("Encoding Complete")

file = open("EncodeFile.p", 'wb')
pickle.dump(encodeListKnownWithIds, file)
file.close()