import cv2 as cv
import numpy as np
from scipy.ndimage.filters import gaussian_filter

from kalman.kalmanfilter import KalmanFilter

import time 
REDU = 8
def rgbh(xs,mask):
    def normhist(x): return x / np.sum(x)
    def h(rgb):
        return cv.calcHist([rgb], [0, 1, 2],mask, [256//REDU, 256//REDU, 256//REDU] , [0, 256] + [0, 256] + [0, 256])
    return normhist(sum(map(h, xs)))

def smooth(s,x):
    return gaussian_filter(x,s,mode='constant')

bgsub = cv.createBackgroundSubtractorMOG2(500, 60, True) 
cap = cv.VideoCapture("src/Static/Dataset/ball_2.mp4")
key = 0

kernel = np.ones((3,3),np.uint8)
crop = False
camshift = False
termination = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 1)
font = cv.FONT_HERSHEY_SIMPLEX
pause= False

pause= False

degree = np.pi/180

fps = 150
dt = 1/fps

noise = 3

A = np.array(
    [1, 0, dt, 0,
    0, 1, 0, dt,
    0, 0, 1, 0,
    0, 0, 0, 1 ]).reshape(4,4)


u = np.array([0, 5])
B = np.array(
    [dt**2/2, 0,
    0, dt**2/2,
    dt, 0,
    0, dt ]).reshape(4,2)

H = np.array(
    [1,0,0,0,
    0,1,0,0]).reshape(2,4)

mu = np.array([0,0,0,0])

P = np.diag([10,10,10,10])**2
res=[]
N = 15
sigmaM = 0.0001
sigmaZ = 3*noise

Q = sigmaM**2 * np.eye(4)
R = sigmaZ**2 * np.eye(2)
listCenterX=[]
listCenterY=[]
kf = KalmanFilter()
add_count = 0
mm=False

while(True):
    key = cv.waitKey(30) & 0xFF
    if key== ord("c"): crop = True
    if key== ord("p"): P = np.diag([100,100,100,100])**2
    if key==27: break
    if key==ord(" "): pause =not pause
    if(pause): continue
    
    ret, frame = cap.read()
    if ret == False:
        break
    
    frame=cv.resize(frame,(1366,768))
    bgs = bgsub.apply(frame)
    bgs = cv.erode(bgs,kernel,iterations = 1)
    bgs = cv.medianBlur(bgs,3)
    bgs = cv.dilate(bgs,kernel,iterations=2)
    bgs = (bgs > 200).astype(np.uint8)*255
    colorMask = cv.bitwise_and(frame,frame,mask = bgs)

    if(crop):
        fromCenter= False
        img = colorMask
        r = cv.selectROI(img, fromCenter)
        imCrop = img[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        crop = False
        camshift = True
        imCropMask = cv.cvtColor(imCrop, cv.COLOR_BGR2GRAY)
        ret,imCropMask = cv.threshold(imCropMask,30,255,cv.THRESH_BINARY)
        his = smooth(1,rgbh([imCrop],imCropMask))
        roiBox = (int(r[0]), int(r[1]),int(r[2]), int(r[3]))

        cv.destroyWindow("ROI selector")

    if(camshift):
        cv.putText(frame,'Center roiBox',(0,10), font, 0.5,(0,255,0),2,cv.LINE_AA)
        cv.putText(frame,'Estimated position',(0,30), font,0.5,(255,255,0),2,cv.LINE_AA)
        cv.putText(frame,'Prediction',(0,50), font, 0.5,(0,0,255),2,cv.LINE_AA)
        add_count += 1
        rgbr = np.floor_divide( colorMask , REDU)
        r,g,b = rgbr.transpose(2,0,1)
        l = his[r,g,b]
        maxl = l.max()

        aa=np.clip((1*l/maxl*255),0,255).astype(np.uint8)
        (rb, roiBox) = cv.CamShift(l, roiBox, termination)

        cv.ellipse(frame, rb, (0, 255, 0), 2)
        xo=int(roiBox[0]+roiBox[2]/2)
        yo=int(roiBox[1]+roiBox[3]/2)

        error=(roiBox[3])
        if(yo<error or bgs.sum()<50 ):
            predicted, mu, statePost, errorCovPre = kf.predict(int(xo), int(yo))
            mu,P = kf.kal(mu,P,B,u,z=None)

            m="None"
            mm=False
        else:

            predicted, mu, statePost, errorCovPre = kf.predict(int(xo), int(yo))
            mu,P = kf.kal(mu,P,B,u,z=np.array([xo,yo]))
            m="normal"
        mm=True
        if(mm):
            listCenterX.append(xo)
            listCenterY.append(yo)

        if len(listCenterX) > 2:
            res += [(mu,P)]
            cv.circle(frame, (predicted[0], predicted[1]), 10, (255, 0, 255),3)

            mu2 = mu
            P2 = P
            res2 = []

            for _ in range(fps*2):
                mu2,P2 = kf.kal(mu2,P2,B,u,z=None)
                res2 += [(mu2,P2)]

            
            xe = [mu[0] for mu,_ in res]
            xu = [2*np.sqrt(P[0,0]) for _,P in res]
            ye = [mu[1] for mu,_ in res]
            yu = [2*np.sqrt(P[1,1]) for _,P in res]
            
            xp=[mu2[0] for mu2,_ in res2]          
            yp=[mu2[1] for mu2,_ in res2]

            xpu = [np.sqrt(P[0,0]) for _,P in res2]
            ypu = [np.sqrt(P[1,1]) for _,P in res2]


            for n in range(len(listCenterX)): 
                cv.circle(frame,(int(listCenterX[n]),int(listCenterY[n])),3,(0, 255, 0),-1)

            for n in [-1]:
                incertidumbre=(xu[n]+yu[n])/2 
                cv.circle(frame,(int(xe[n]),int(ye[n])),5,(255, 255, 0),3)


            for n in range(len(xp)):
                incertidumbreP=(xpu[n]+ypu[n])/2
                cv.circle(frame,(int(xp[n]),int(yp[n])),int(incertidumbreP),(0, 0, 255))            

            if(len(listCenterY)>40):

                print("REBOTE")
                listCenterY=[]
                listCenterX=[]
                res=[]

                mu = np.array([0,0,0,0])
                P = np.diag([100,100,100,100])**2


    cv.imshow('Frame', frame)
    cv.imwrite('./saved/saved_{:04d}.png'.format(add_count),frame)

cv.destroyAllWindows()