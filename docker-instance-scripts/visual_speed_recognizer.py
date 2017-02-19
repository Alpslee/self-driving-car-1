import numpy as np
import cv2
import argparse
import imutils
import os
import time

procWidth = 1280 #640   # processing width (x resolution) of frame
procHeight = 720   # processing width (x resolution) of frame

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-v", "--video", help="path to the video file")
ap.add_argument("-s", "--skipNr", help="skip frames nr")
ap.add_argument("-m", "--mode", help="mode - game")

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

mode=1 # 1 - The Crew PS4, 2 - Drive Club 3 - GT5 PS3

if args.get("mode", None) is not None:
    mode = int(args.get("mode"))

print("mode", mode)

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
#out = cv2.VideoWriter('output.mp4',fourcc, 20.0, (640,480)) 
out = cv2.VideoWriter('output.mp4',fourcc, 60.0, (int(procWidth),int(procHeight)))

blank_image = np.zeros((480,640,3), np.uint8)
out = None
record = False

#The Crew
x1= 75 #65
y1= 580 #560 #585
x2= 160 #180 #160
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
if(mode==3):
    x1=int(1280/2-36)
    y1=621 #585
    x2=int(1280/2+36) #160
    y2=650
    maxHeight=28
    maxWidth=24

digitsImages = []
#digitsImages = np.zeros((37*10,20,1), np.uint8)
for i in range (0,10):
    try:
        img = cv2.imread('digit-'+str(i)+'-scaled-'+str(mode)+'.png',1)
        #if mode == 1:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        #print("img.shape",img.shape)
        #img = imutils.resize(img, height=37) #The Crew
        if(mode==2):
            img = imutils.resize(img, width=maxWidth, height=maxHeight) #Drive Club
        #elif(mode==3):
        #    img = cv2.resize(img, (maxWidth, maxHeight)) #GT5
            #img = cv2.resize(img, fx=maxWidth/img.shape[0], fy=maxHeight/img.shape[1]) #GT5
            #img = np.reshape(img,  (maxHeight, maxWidth)) #GT5
        digitsImages.append(img)
        print("i img.shape",str(i),img.shape)
    except:
        print("not loaded i",str(i))
    #map = [2,0,3,4,5,6,1,7,-1,-1] #for diff < 50
#map = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1] #for diff < 50
diffImage=[]
digitsImagesIndex=3

speed = [0,0,0]
frames=0
while(True):
    # Capture frame-by-frame
    ret, frame = camera.read()
    frames=frames+1
    
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
        
        if mode==1:
            #ret, mask = cv2.threshold(mask, 150, 200, cv2.THRESH_BINARY)
            ret, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        elif mode==2:
            #ret, mask = cv2.threshold(mask, 150, 200, cv2.THRESH_BINARY)
            ret, mask = cv2.threshold(mask, 150, 255, cv2.THRESH_BINARY)
        elif mode==3:
            ret, mask = cv2.threshold(mask, 150, 200, cv2.THRESH_BINARY)
            #ret, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
        
        #ret, mask = cv2.dilate(mask, 250, 255, cv2.THRESH_BINARY)
        #mask = cv2.dilate(mask, np.ones((3, 3)), iterations=1)
        #mask = cv2.erode(mask, np.ones((1, 1)), iterations=1)


        if mode==1:
            mask = cv2.dilate(mask, np.ones((2, 2)), iterations=1)

        image_final = mask

        #ret, new_img = cv2.threshold(image_final, 180 , 255, cv2.THRESH_BINARY)  # for black text , cv.THRESH_BINARY_INV
        #kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(3 , 3)) # to manipulate the orientation of dilution , large x means horizonatally dilating  more, large y means vertically dilating more 
        #dilated = cv2.dilate(new_img,kernel,iterations = 9) # dilate , more the iteration more the dilation

        #contours, hierarchy = cv2.findContours(dilated,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE) # get contours
        #index = 0 
        
        contours = []
        #image, contours, hierarchy = cv2.findContours(image_final.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
        if mode == 1:
            image, contours, hierarchy = cv2.findContours(image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #,(x1,y2))
        elif mode == 2:
            image, contours, hierarchy = cv2.findContours(image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #,(x1,y2))
        elif mode == 3:
            #image, contours, hierarchy = cv2.findContours(image_final.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #,(x1,y2))
            shift=1
            contours.append(np.array([(maxWidth*(shift-1), 0),(maxWidth*(shift-1),maxHeight),(maxWidth*shift, maxHeight),(maxWidth*shift, 0),(maxWidth*(shift-1),0)], dtype=np.int))
            shift=2
            contours.append(np.array([(maxWidth*(shift-1), 0),(maxWidth*(shift-1),maxHeight),(maxWidth*shift, maxHeight),(maxWidth*shift, 0),(maxWidth*(shift-1),0)], dtype=np.int))
            shift=3
            contours.append(np.array([(maxWidth*(shift-1), 0),(maxWidth*(shift-1),maxHeight),(maxWidth*shift, maxHeight),(maxWidth*shift, 0),(maxWidth*(shift-1),0)], dtype=np.int))


        #image, contours, hierarchy = cv2.findContours(image_final, cv2.RETR_EXTERNAL, cv2.CV_CHAIN_APPROX_TC89_KCOS)
        
        #image, contours, hierarchy = cv2.findContours(image_final.copy(),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        #contours = []
        #colorImage = image_final

        digitsCount=-1

        #print("digit: ")
        
        cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),1)

        for cnt in contours:
            #print("cnt.len",len(cnt))
            #if cv2.contourArea(cnt)>0:
            [x,y,w,h] = cv2.boundingRect(cnt)
            #if w >= 15 and (h >= 30 and h <= 40):
            #print("x y h w",x,y,h,w) #37:20
            if  w > (maxWidth-2):
                #colorImage = cv2.cvtColor(colorImage,cv2.COLOR_GRAY2BGR)
                cv2.rectangle(frame,(x1+x,y1+y),(x1+x+w,y1+y+h),(0,255,127),1)

                digitsCount=digitsCount+1
                
                #currentDigitImage = image_final[y:y+37,x:x+20]
                if mode==1:
                    currentDigitImage = image_final[y:y+maxHeight,x:x+maxWidth]
                elif mode==2:
                    currentDigitImage = image_final[y:y+maxHeight,x:x+w]
                elif mode==3:
                    currentDigitImage = image_final[y:y+maxHeight,x:x+maxWidth]

                blackPixelsCurrentDigitImage = cv2.countNonZero(currentDigitImage)
                #print("blackPixelsCurrentDigitImage",blackPixelsCurrentDigitImage)

                index = -1
                digit = -1
                maxDiff=0
                '''
                if mode == 1:
                    maxDiff=0 #37*20
                elif mode==3:
                    maxDiff=0
                '''

                #if blackPixelsCurrentDigitImage > 0:
                for digitImage in digitsImages:
                    index=index+1
                    if blackPixelsCurrentDigitImage < 1 or index>9:
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

                    #print("currentDigitImage.shape",currentDigitImage.shape)
                    #print("img.shape",digitImage.shape)
                    if mode==1:
                        digitImageDiff = cv2.bitwise_xor(currentDigitImage,digitImage)
                        #digitImageDiff = cv2.bitwise_and(currentDigitImage,digitImage)
                    elif mode==3:
                        digitImageDiff = cv2.bitwise_and(currentDigitImage,digitImage)
                    
                    #cv2.imshow('currentDigitImage',currentDigitImage)
                    #cv2.imshow('digitImage',digitImage)
                    #cv2.imshow('digitImageDiff',digitImageDiff)

                    if mode==1:
                        diffPixels = blackPixelsCurrentDigitImage-cv2.countNonZero(digitImageDiff)
                        #diffPixels = cv2.countNonZero(digitImageDiff)
                    elif mode==3:
                        diffPixels = cv2.countNonZero(digitImageDiff)
                    
                    #if(abs(diffPixels)>580): #adjust value based on the size of the digit
                    if abs(diffPixels) > maxDiff:
                        digit = index
                        #time.sleep(3)
                        maxDiff = abs(diffPixels)
                        #break
                        #found match
                

                '''
                print("index",index)
                print("maxDiff",maxDiff)
                print("digit",digit)
                print("digitsCount",digitsCount)
                print("len(digitsImages)",len(digitsImages))
                '''

                '''
                if blackPixelsCurrentDigitImage > 0 and digit<digitsImagesIndex:
                    cv2.imwrite("digit-"+str(digitsImagesIndex)+"-scaled-"+str(mode)+".png",currentDigitImage)
                    digitsImages.append(currentDigitImage)
                    print("newDigit index",index)
                    digitsImagesIndex =  digitsImagesIndex+1
                '''

                if index<10:
                    if digit==-1:
                        if index > -1:
                            speed[digitsCount] = index
                        else:
                            speed[digitsCount] = 0
                    else:
                        speed[digitsCount] = digit
                        #print(digit, end='', flush=True)
                else:
                    speed[digitsCount] = 'u' 
                    #print('u', end='', flush=True)

                #cv2.rectangle(frame,(x1+x-2,y1+y-2),(x1+x+w,y1+y+h),(0,255,127),1)
                cv2.putText(frame,str(speed[digitsCount]),(x1+x-2,y1+y-7),0,1.0,(0,255,0),2)
                '''
                if digit==8:
                    print("index",index)
                    print("maxDiff",maxDiff)
                    print("speed","".join(speed))
                    time.sleep(3)
                '''
                #cv2.putText(out,string,(x+200,y+h),0,1,(0,255,0))
                #frame[y1:y2, x1:x2] = im

        if mode==1:
            speed = speed[::-1]
            print("speed ",speed,str(frames))#,sep='')
        else:
            print("speed ",speed,str(frames))#,sep='')
        cv2.imshow('original',frame)
        #cv2.imshow('digitImage',image_final)
        #cv2.imshow('digitImageDiff',digitImageDiff)

    except:
        #exit(0)
        print("speed","0")
        cv2.imshow('original',frame)
        #cv2.imshow('digitImage',image_final)

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
