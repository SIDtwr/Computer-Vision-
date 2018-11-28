import numpy as np
import cv2


def noisy(noise_typ,image):

   if noise_typ == "s&p":
      row,col,ch = image.shape
      s_vs_p = 0.5
      amount = 0.054
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt))
              for i in image.shape]
      out[coords] = 1

      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper))
              for i in image.shape]
      out[coords] = 0
      return out
   else :
       print("passed wrong argument to the function")


cap = cv2.VideoCapture('SampleVideo.mp4')

while(cap.isOpened()):
    ret, frame = cap.read()

    # changing the brightness and the contrast of the video
    brightness = 50
    contrast = 10
    frame = np.int16(frame)
    frame = frame * (contrast / 127 + 1) - contrast + brightness
    frame = np.clip(frame, 0, 255)
    frame = np.uint8(frame)

    # adding random gaussian noise to the video
    frame = noisy("s&p",frame)

    # converting image to gray-scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # reducing noise in the image.
    equ = cv2.fastNlMeansDenoising(gray,None,10,7,21)

    # Histogram Equalization (comment out to use standard equalization technique)
    # equ2 = cv2.equalizeHist(equ)

    # adaptive histogram equalization filter
    # create a CLAHE object (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equ2 = clahe.apply(equ)


    ## different types of smoothing filters
    # equ2 = cv2.GaussianBlur(equ2,(5,5),0)
    # equ2 = cv2.bilateralFilter(equ2, 9, 75, 75)
    equ2 = cv2.medianBlur(equ2, 3) # medium-blur filter reduces salt & pepper noise from the image.

    # Create our shapening kernel, it must equal to one eventually
    kernel_sharpening = np.array([[-1, -1, -1],
                                  [-1, 9, -1],
                                  [-1, -1, -1]])

    # applying the sharpening kernel to the input image & displaying it.
    sharpened = cv2.filter2D(equ2, -1, kernel_sharpening)

    grayR = cv2.resize(gray, (680, 480))  # Resize image
    equR = cv2.resize(equ2, (680, 480))  # Resize image

    #res = np.hstack((frame, equ))  #stacking images side-by-side
    cv2.imshow('Original Video', grayR)
    cv2.imshow('Histogram Equalization & Contrast Enhancement', equR)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


