import cv2
import numpy as np

#A. Apply the geometric transformations over an image
def A():
    img = cv2.imread(r'img_219.jpg')
    height, width = img.shape[:2]

    cv2.imshow('Titlu', img)

    # Scale
    scaled_image = cv2.resize(img, None, interpolation=cv2.INTER_LINEAR, fx=2, fy=2)
    cv2.imshow('Scaled', scaled_image)

    # Rotate
    center = (height // 2, width // 2)
    angle = 90
    scale = 1
    rot_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotate_image = cv2.warpAffine(img, rot_matrix, (height, width))
    cv2.imshow('Rotated', rotate_image)

    # Translate
    translate_matrix = np.float32([[1, 0, 100], [0, 1, 50]])
    translate_image = cv2.warpAffine(img, translate_matrix, (height, width))
    cv2.imshow('Translated', translate_image)

    # Flip
    flip_image = cv2.flip(img, 1)
    cv2.imshow('Flipped', flip_image)

# B. Apply morphological operations over an image (dilation, erosion)
def B():
    img = cv2.imread(r'img_219.jpg')
    cv2.imshow("Titlu", img)

    height, width = img.shape[:2]

    kernel = np.ones((3,3), np.uint8)

    # erosion
    eroded_image = cv2.erode(img, kernel, iterations=2)
    cv2.imshow('Erosion', eroded_image)

    # dilation
    dilated_image = cv2.dilate(img, kernel, iterations=4)
    cv2.imshow('Dilatation', dilated_image)

    # img2 = cv2.imread(r'face_benchmarks\2002\07\19\big\img_389.jpg')
    # img2 = cv2.resize(img2, (height, width), interpolation=cv2.INTER_LINEAR)
    # img2 = img2[:, :, 1]
    # eroded_image2 = cv2.erode(img, img2, iterations=2)
    # cv2.imshow('ErosionManu', eroded_image2)

# C. Transform an image to grayscale
def C():
    img = cv2.imread(r'img_219.jpg')
    cv2.imshow('C', img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow('Grayscale', gray)

# D. Count the number of frames in a video
def D():
    cap = cv2.VideoCapture('drop.avi')
    num = 1

    while (cap.isOpened()):
        _, frame = cap.read()

        if frame is not None:
            cv2.imshow('frame', frame)
            num += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print(num)

def E():
    img = cv2.imread('tema.jpg') #citire imagine
    cv2.imshow('Tema', img) #afisare imagine
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #conversie la grayscale
    cv2.imshow('Grayscale', gray) #afisare grayscale
    _, albNegru = cv2.threshold(gray,70,255,cv2.THRESH_BINARY) #binarizare <70 -> negru, >70->alb, max = 255
    cv2.imshow('Binarizat', albNegru) #afisare binarizata
    font = cv2.FONT_HERSHEY_COMPLEX #font
    nrct, _ = cv2.findContours(albNegru,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE) #ierarhie contururi

    for cnt in nrct:
     M = cv2.moments(cnt)
     if M['m00'] != 0:
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
     else:
        cx,cy = 0, 0
     center = (cx, cy)
     radius = 5
     cv2.circle(img, (cx, cy), radius, (0, 0, 255), -1)
     cv2.putText(img,(str(cx)+","+str(cy)),(cx-15,cy-10),font,0.3,(0,100,255),1,cv2.LINE_AA)

    cv2.imshow('Final', img)

if __name__ == "__main__":
     E()
     cv2.waitKey(0)
     cv2.destroyAllWindows()

