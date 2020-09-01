from PIL import Image
import cv2
import imutils
import numpy as np
from keras.models import load_model


bg = None
model = load_model('mudras_model_2.h5') #mudras_model_2.h5 - Model loaded which was created using Project.py file 

def resizeImage(imageName):
    basewidth = 100
    img = Image.open(imageName)
    wpercent = (basewidth/float(img.size[0]))
    hsize = int((float(img.size[1])*float(wpercent)))
    img = img.resize((basewidth,hsize), Image.ANTIALIAS)
    img.save(imageName)

def run_avg(image, aWeight):
    global bg
    # initialize the background
    if bg is None:
        bg = image.copy().astype("float")
        return

    # compute weighted average, accumulate it and update the background
    cv2.accumulateWeighted(image, bg, aWeight)

def segment(image, threshold=25):
    global bg
    # find the absolute difference between background and current frame
    diff = cv2.absdiff(bg.astype("uint8"), image)

    # threshold the diff image so that we get the foreground
    foreground = cv2.threshold(diff,
                                threshold,
                                255,
                                cv2.THRESH_BINARY)[1]

    # get the contours in the thresholded image
    (cnts, hierarchy) = cv2.findContours(foreground.copy(),
                                    cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)

    # return None, if no contours detected
    if len(cnts) == 0:
        return
    else:
        # based on contour area, get the maximum contour which is the hand
        segmented = max(cnts, key=cv2.contourArea)
        return (foreground, segmented)

def prediction():
    # prediction of the image given 
    image = cv2.imread('img.JPG')
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_image = cv2.resize(gray_image,(120,120))
    prediction = model.predict([gray_image.reshape(-1,120, 120, 1)])
    return np.argmax(prediction)

# initialize weight for running average
aWeight = 0.5

# get the reference to start the webcam
camera = cv2.VideoCapture(0)

# region of interest (ROI) coordinates - creating a box to capture the hand 
top, right, bottom, left = 10, 350, 325, 590

# keeping count of the frames
frame_numbers = 0
start_recording = False

# Keep capturing frames until user interrupts it 
while(True):
    # get the current frame
    (grabbed, frame) = camera.read()

    # resize the frame
    frame = imutils.resize(frame, width = 700)

    # flip the frame so that it is not the mirror view
    frame = cv2.flip(frame, 1)

    # clone the frame
    copyFrame = frame.copy()

    # get the height and width of the frame
    (height, width) = frame.shape[:2]

    # get the ROI
    roi = frame[top:bottom, right:left]

    # convert the roi to grayscale and blur it
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)

    # to get the background, keep looking till a threshold is reached
    # so that our running average model gets calibrated
    if frame_numbers < 30:
        run_avg(gray, aWeight)
    else:
        # segmention of hand by background elimination 
        hand = segment(gray)

        # check whether hand region is segmented
        if hand is not None:
            
            (foreground, segmented) = hand

            # draw the region of interest and display the frame
            cv2.drawContours(copyFrame, [segmented + (right, top)], -1, (0, 0, 255))
            if start_recording:
                cv2.imwrite('img.JPG', foreground)
                resizeImage('img.JPG')
                predictedClass = prediction()
                # associating each number/label with its corresponding mudra name
                mudras = { 0: 'Pataka-Flag',
                1: 'Tripataka-Tree',
                2: 'Ardhapataka-HalfFlag',
                3: 'Kartarimukha-ScissorsFace',
                4: 'Mayura-Peacock',
                5: 'Ardhachandra-HalfMoon',
                6: 'Arala-Bent',
                7: 'Shukatunda-BeakOfAParrot',
                8: 'Mushthi-Fist',
                9: 'Shikhara-Peak',
                10: 'Kapitta-ElephantApple',
                11: 'Katakamukha-OpeningOfABracelet',
                12: 'Katakamukha-OpeningOfABracelet',
                13: 'Katakamukha-OpeningOfABracelet',
                14: 'Suchi-Needle',
                15: 'Chandrakala-FaceOfTheMoon',
                16: 'Padmakosha-LotusBud',
                17: 'Sarpashirsha-SnakeHead',
                18: 'Mrigashirsha-HeadOfADeer',
                19: 'Simhamukha-FaceOfLion',
                20: 'Kangula-Lily',
                21: 'Alapadma-lotus',
                22: 'Chatura-Four',
                23: 'Bhramara-Bee',
                24: 'Hamsasya-SwanHead',
                25: 'Hamsapaksha-SwanWings',
                26: 'Sandamsha-Pincers1',
                27: 'Sandamsha-Pincers2',
                28: 'Mukula-FlowerBud',
                29: 'Tamrachuda-Rooster',
                30: 'Trishula-Trident',
                31: 'Anjali-Offering',
                32: 'Kapotam-Dove',
                33: 'Karkatam-Crab',
                34: 'Swastikam-AuspiciousSign',
                35: 'Pushpaputam-BagOfFlowers',
                36: 'Shivalingam-SignOfLordShiva',
                37: 'Katakavardhanam-chain',
                38: 'Kartariswastikam-StemAndBranchesOfTheTree',
                39: 'Shakatam-Carriage',
                40: 'Shankha-ConchShell',
                41: 'Chakram-RotatingDisc',
                42: 'Samputa-RoundShapedCasket',
                43: 'Pasha-Ropes',
                44: 'Kilaka-Bolt',
                45: 'Matsya-Fish',
                46: 'Kurma-Tortoise',
                47: 'Varaha-Boar',
                48: 'Garuda-HalfEagleHalfHumanMountOfLordVishnuABirdFlying',
                49: 'Nagabandham-SnakesEntwined',
                50: 'Khattva-Cot',
                51: 'Bherunda-APairOfBirds'
                }
                result = np.zeros((300,1000,3), np.uint8)
                cv2.putText(result,"Predicted Class : " + str(mudras[predictedClass]), (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,(255, 255, 255),2)
                cv2.imshow("Thresholded", foreground)
                cv2.imshow("Result", result)

    # draw the segmented hand
    cv2.rectangle(copyFrame, (left, top), (right, bottom), (0,255,0), 2)

    # increment the number of frames
    frame_numbers += 1

    # display the frame with segmented hand
    cv2.imshow("Video Feed", copyFrame)

    # observe the keypress by the user
    key = cv2.waitKey(1) & 0xFF

    # if the user pressed "e", then stop looping
    if key == ord("e"):
        break
    
    if key == ord("s"):
        start_recording = True
