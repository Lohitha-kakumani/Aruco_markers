import cv2
import cv2.aruco as aruco

VideoCap = False

cap = cv2.VideoCapture(0)

def findAruco(img,marker_size = 6,total_markers = 250,draw = True):
    gray = cv2.cutColor(img,cv2.COLOR_BGR2GRAY)
    arucoDict = aruco.Dictionary_get(aruco.DICT_6X6_250 )

while True:
    #if VideoCap: 
    _,img = cap.read()
    #else: 
    #    img = cv2.imread("images.jpg") #picture detection
    #    img = cv2.resize(img,(0,0),fx=0.5,fy=0.5)
    if cv2.waitKey(1) ==113:
        break
    cv2.imshow("img",img)  #live camera press Q to exit

