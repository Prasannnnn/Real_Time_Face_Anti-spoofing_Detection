from cvzone.FaceDetectionModule import FaceDetector
import cv2
import cvzone
from time import time

classID = 0 # 0 is fake 1 is Real
outputfolderpath = 'Dataset/Datacollect'
confidence = 0.8
offsetpercentagew = 10
offsetpercentageh = 20
camwidth , camheight = 640,480
blurthershold = 35 # larger is more focus
floatingpoint = 6
save = True
debug = False


cap = cv2.VideoCapture(0)
detector = FaceDetector(minDetectionCon=0.5, modelSelection=0)
cap.set(3,camwidth)
cap.set(4,camheight)


while True:
        
        success, img = cap.read()
        imgout = img.copy()
        img, bboxs = detector.findFaces(img, draw=False)

        listblur = [] #True False values indicating if the faces are blur or not
        listinfo = [] #The Normalized values and the class name for the label txt file
        if bboxs:
            for bbox in bboxs:
                center = bbox["center"]
                score = bbox["score"][0]
                # print(score)
                x, y, w, h = bbox['bbox']
                # print(x,y,w,h)

                #### check the score ############################
                if score > confidence:

                    ########offset creation############################
                    offsetw = (offsetpercentagew / 100)*w

                    x = int(x - offsetw)
                    w = int(w + offsetw * 2)

                    offseth = (offsetpercentageh / 100)*h

                    y = int(y - offseth*2.5)
                    h = int(h + offseth * 3.5)

                    ######### To avoid value 0 ############################
                    if x < 0 : x = 0
                    if y < 0 : y = 0
                    if w < 0 : w = 0
                    if h < 0 : h = 0




                    ###Find the blurring images ##############################

                    imgFace = img[y:y+h,x:x+w]
                    cv2.imshow("Face",imgFace)
                    blurvalue = int(cv2.Laplacian(imgFace,cv2.CV_64F).var())
                    if blurvalue > blurthershold:
                         listblur.append(True)
                    else:
                         listblur.append(False)

                    ### Normalize values ##############################
                    ih, iw, _ = img.shape
                    xc,yc = x+w/2,y+h/2
                    # print(xc,yc)
                    xcn,ycn = round(xc/iw,floatingpoint),round(yc/ih,floatingpoint) 
                    wn,hn = round(w / iw,floatingpoint), round(h/ih,floatingpoint)
                    # print(xcn,ycn,wn,hn)

                    ######### To avoid value 0 ############################
                    if xcn > 1 : xcn = 1
                    if ycn > 1 : ycn = 1
                    if wn > 1 : wn = 1
                    if hn > 1 : hn = 1


                    listinfo.append(f"{classID} {xcn} {ycn} {wn} {hn}\n")
                    ##Drawing ##############################

                    cv2.rectangle(imgout,(x,y,w,h),(255,0,0),3)
                    cvzone.putTextRect(imgout,f'Score :{int(score*100)} % Blur: {blurvalue}',(x,y-20),
                                       scale=1,thickness=2)
                    
                    if debug:
                         cv2.rectangle(img,(x,y,w,h),(255,0,0),3)
                         cvzone.putTextRect(img,f'Score :{int(score*100)} % Blur: {blurvalue}',(x,y-20),
                                       scale=1,thickness=2)


            ##To save        
            if save :
                 if all(listblur) and listblur!=[] :
            ##save Image
                    timenow = time()
                    timenow = str(timenow).split('.')
                    timenow = timenow[0] + timenow[1]
                    cv2.imwrite(f"{outputfolderpath}/{timenow}.jpeg",img)

                    ##save Label text file
                    for info in listinfo:
                         f = open(f"{outputfolderpath}/{timenow}.txt","a")
                         f.write(info)
                         f.close()


                      
                      
            
        cv2.imshow("Image", imgout)
        if cv2.waitKey(1) & 0xFF == "q":  # 27 is the ESC key
            break

cap.release()
cv2.destroyAllWindows()