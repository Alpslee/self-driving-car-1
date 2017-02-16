import numpy as np
import cv2
import argparse
import imutils
import os
import time

procWidth = 720 #640   # processing width (x resolution) of frame
#procHeight = 480   # processing width (x resolution) of frame

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-s", "--skipNr", help="skip frames nr")

args = vars(ap.parse_args())
FORWARD='gimx --event "abs_axis_9(255)" --dst 127.0.0.1:51914 && sleep 0.2 && gimx --event "abs_axis_9(0)" --dst 127.0.0.1:51914'
REVERSE='gimx --event "triangle(255)" --dst 127.0.0.1:51914 && sleep 0.2 && gimx --event "triangle(0)" --dst 127.0.0.1:51914'
LEFT='gimx --event "abs_axis_6(255)" --dst 127.0.0.1:51914 && sleep 0.2 && gimx --event "abs_axis_6(0)" --dst 127.0.0.1:51914'
RIGHT='gimx --event "abs_axis_4(255)" --dst 127.0.0.1:51914 && sleep 0.2 && gimx --event "abs_axis_4(0)" --dst 127.0.0.1:51914'
PS='gimx --event "abs_axis_2(255)" --dst 127.0.0.1:51914 && sleep 0.2 && gimx --event "abs_axis_2(0)" --dst 127.0.0.1:51914'

def send_gimx_cmd(cmd):
    print("cmd: "+cmd)
    os.system(cmd)
    
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False

# if the video argument is None, then we are reading from webcam
if args.get("video", None) is None:
    camera = cv2.VideoCapture(0)
    time.sleep(2)

skipNr=0
if args.get("skipNr", None) is not None:
    skipNr = int(args.get("skipNr"))

print("skipNr", skipNr)

if args.get("video", None) is None:
    portNr = args.get("video")
# otherwise, we are reading from a video file
else:
    portNr = args.get("video")
    if is_number(portNr):
        camera = cv2.VideoCapture(int(portNr))
    else:
        camera = cv2.VideoCapture(portNr)


#fourcc = cv2.VideoWriter_fourcc(*'MJPG')
#fourcc = cv2.VideoWriter_fourcc(*'mp4v')
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
# Define the codec and create VideoWriter object
#fourcc = cv2.VideoWriter_fourcc(*'XVID')
fourcc = cv2.VideoWriter_fourcc(*'MPEG')
out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480)) 

blank_image = np.zeros((480,640,3), np.uint8)
out = None
record = False

mode=1 # 1 - The Crew, 2 - Drive Club
#The Crew
x1=65
y1=560 #585
x2=180 #160
y2=640
maxHeight=37
maxWidth=20
#Drive Club
if(mode==2):
    x1=1280-220
    y1=540 #585
    x2=1280-130 #160
    y2=610
    maxHeight=27
    maxWidth=14

digitsImages = []
#digitsImages = np.zeros((37*10,20,1), np.uint8)
for i in range (0,10):
    img = cv2.imread('digit-'+str(i)+'-scaled.png',1)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #img = imutils.resize(img, height=37) #The Crew
    if(mode==2):
        img = imutils.resize(img, height=27) #Drive Club
    digitsImages.append(img)
    print("img.shape",img.shape)
    #map = [2,0,3,4,5,6,1,7,-1,-1] #for diff < 50
#map = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] #for diff < 50

while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()
    
    if not ret:
        frame = blank_image
        #print ("blank_image")
    else:
        #frame = imutils.resize(frame, width=procWidth)
        blank_image = frame
    
    if skipNr>0:
        skipNr = skipNr - 1
        continue

    #Mat im;
    #im = frame[y1:y2, x1:x2]

    speed_section = frame[y1:y2, x1:x2]
    
    mask = cv2.cvtColor(speed_section,cv2.COLOR_BGR2GRAY)
    #thresh = cv2.adaptiveThreshold(gray,255,1,1,11,2)
    #cv2.imshow('digitImage',speed_section)

    try:

        #image_final = cv2.bitwise_and(gray , gray , mask =  mask)
        #thresh = cv2.adaptiveThreshold(blur,255,1,1,11,2)
        #mask = cv2.GaussianBlur(mask,(5,5),0)
        ret, mask = cv2.threshold(mask, 240, 255, cv2.THRESH_BINARY)
        #ret, mask = cv2.dilate(mask, 250, 255, cv2.THRESH_BINARY)
        #mask = cv2.dilate(mask, np.ones((3, 3)), iterations=1)
        #mask = cv2.erode(mask, np.ones((1, 1)), iterations=1)


        mask = cv2.dilate(mask, np.ones((2, 2)), iterations=1)

        image_final = mask

        #ret, new_img = cv2.threshold(image_final, 180 , 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3 , 3)) # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more 
        #dilated = cv2.dilate(new_img,kernel,iterations = 9) # dilate , more the iteration more the dilation

        #contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
        #index = 0 
        
        contours = []
        #image, contours, hierarchy = cv2.findContours(image_final.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        image, contours, hierarchy = cv2.findContours(image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #,(x1,y2))

        #image, contours, hierarchy = cv2.findContours(image_final, cv2.RETR_EXTERNAL, cv2.CV_CHAIN_APPROX_TC89_KCOS)
        
        #image, contours, hierarchy = cv2.findContours(image_final.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #contours = []
        #colorImage = image_final

        digitsCount=0
        #print("digit: ")
        
        speed = list("000")

        for cnt in contours:
            #print("cnt.len",len(cnt))
            #if cv2.contourArea(cnt)>0:
            [x,y,w,h] = cv2.boundingRect(cnt)
            #if w >= 15 and (h >= 30 and h <= 40):
            if  h>(maxHeight-3):
                #print("h w",h,w) #37:20
                #colorImage = cv2.cvtColor(colorImage,cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame,(x1+x-2,y1+y-2),(x1+x+w,y1+y+h),(0,255,127),1)
                digitsCount=digitsCount+1
                
                #currentDigitImage = image_final[y:y+37,x:x+20]
                if mode==1:
                    currentDigitImage = image_final[y:y+maxHeight,x:x+maxWidth]
                elif mode==2:
                    currentDigitImage = image_final[y:y+maxHeight,x:x+w]

                blackPixelsCurrentDigitImage = cv2.countNonZero(currentDigitImage)
                index = -1
                digit = -1
                maxDiff=0 #37*20
                for digitImage in digitsImages:
                    index=index+1
                    if index >9:
                        break

                    ''''
                    key = cv2.waitKey(1)
                    if key == 32:
                        print("maxDiff",maxDiff)
                        time.sleep(3)

                    if key & 0xFF == ord('q'):
                        cap.release()
                        cv2.destroyAllWindows()
                    '''
                    # Take only region of logo from logo image.
                    #digitImageDiff = cv2.bitwise_and(digitImage.copy(),currentDigitImage.copy())
                    digitImageDiff = cv2.bitwise_xor(currentDigitImage,digitImage)
                    #digitImageDiff = cv2.bitwise_and(currentDigitImage,digitImage)
                    
                    #cv2.imshow('currentDigitImage',currentDigitImage)
                    #cv2.imshow('digitImage',digitImage)
                    #cv2.imshow('digitImageDiff',digitImageDiff)

                    diffPixels = blackPixelsCurrentDigitImage-cv2.countNonZero(digitImageDiff);
                    
                    #if(abs(diffPixels)>580): #adjust value based on the size of the digit
                    if abs(diffPixels) > maxDiff:
                        digit = index
                        #time.sleep(3)
                        maxDiff = abs(diffPixels)
                        #break
                        #found match
                
                #print("index",index)
                #print("maxDiff",maxDiff)

                if index<10:
                    if digit==-1:
                        #digitsImages.append(currentDigitImage)
                        speed[3-digitsCount] = str(index)
                        #print(index, end='', flush=True)
                    else:
                        speed[3-digitsCount] = str(digit)
                        #print(digit, end='', flush=True)
                else:
                    speed[3-digitsCount] = 'u' 
                    #print('u', end='', flush=True)

                #cv2.rectangle(frame,(x1+x-2,y1+y-2),(x1+x+w,y1+y+h),(0,255,127),1)
                cv2.putText(frame,speed[3-digitsCount],(x1+x-2,y1+y-7),0,1.0,(0,255,0),2)
                '''
                if digit==8:
                    print("index",index)
                    print("maxDiff",maxDiff)
                    print("speed","".join(speed))
                    time.sleep(3)
                '''
                #cv2.putText(out,string,(x+200,y+h),0,1,(0,255,0))
                #frame[y1:y2, x1:x2] = im

        print("speed","".join(speed))
        cv2.imshow('original',frame)
        cv2.imshow('digitImage',image_final)

    except:
        #exit(0)
        print("speed","0")
        cv2.imshow('original',frame)

    #print("digitsCount",digitsCount)
    #cv2.imshow('speed',image_final)

    # Our operations on the frame come here
    #gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Display the resulting frame
    
    key = cv2.waitKey(1)
    if(key!=-1):
        print ("key: "+str(key))
    if key & 0xFF == ord('q'):
        break
    '''    
    if key == 65361:
        send_gimx_cmd(LEFT)
    if key == 65362:
        send_gimx_cmd(FORWARD)
    if key == 65363:
        send_gimx_cmd(RIGHT)
    if key == 65364:
        send_gimx_cmd(REVERSE)
    if key == 32:
        send_gimx_cmd(PS)
    '''    
    if key == 32:
        record = not record
        print("record: ",record)

    if(record):
        if(out == None):
            out = cv2.VideoWriter('output.mp4',fourcc, 30.0, (int(procWidth),int(procHeight)))
        out.write(frame)
    

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()
