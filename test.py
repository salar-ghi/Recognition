import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import torch

os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;0"
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.set_default_device(device)
torch.device(device)

Localurl = 'rtsp://admin:admin4763@192.168.5.190:554/'
resolutions = [[640, 480],[1024, 768],[1280, 704],[1920, 1088],[3840, 2144], [4032, 3040]]
cap = cv2.VideoCapture(Localurl, cv2.CAP_FFMPEG)
cap = cv2.VideoCapture(Localurl)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_GSTREAMER,1)
cap.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter.fourcc('M','J','P','G'))
cap.set(cv2.CAP_PROP_FOURCC ,cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolutions[0][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolutions[0][1])
cap.set(cv2.CAP_PROP_POS_AVI_RATIO, 1)
cap.set(cv2.CAP_PROP_FPS, 5.0)
cap.set(cv2.CAP_PROP_FRAME_COUNT,1)
cap.set(cv2.CAP_PROP_EXPOSURE, -5)

imgBackground = cv2.imread('assets/background.png')
folderModePath = 'assets/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(os.path.join(folderModePath, path)))
    pass



def rescale_frame(frame):
    # width = int(frame.shape[1] * percent/ 100)
    # height = int(frame.shape[0] * percent/ 100)
    # dim = (width, height)
    dim = (640, 480)
    return cv2.resize(frame, dim, interpolation =cv2.INTER_AREA)

while True:
    ret, frame= cap.read()
    if  not ret:
        break



    img = rescale_frame(frame)
    imgBackground[162:162+480, 55:55+640] = img
    imgBackground[44:44 + 633, 808:808 + 414] = imgModeList[3]
    cv2.imshow("face Attaendance", imgBackground)
    # cv2.waitKey(1)
    k = cv2.waitKey(1) & 0Xff
    if k == 27: # Press 'ESC' to quit
        break
cv2.destroyAllWindows()