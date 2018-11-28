import numpy as np
import cv2

kernelSize = 21  # Kernel Bluring size

# Edge Detection Parameter
parameter1 = 20
parameter2 = 60
intApertureSize = 1

cap = cv2.VideoCapture(0)
while (True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Our operations on the frame come here
    img = cv2.flip(frame, 1)  # flip left-right

    # Our operations on the frame come here
    # img = cv2.GaussianBlur(img, (kernelSize, kernelSize), 0, 0)
    # img = cv2.medianBlur(frame, kernelSize)  # Median Blur smoothing filter
    # img = frame = cv2.blur(frame, (kernelSize, kernelSize))  # Average Blur smoothing filter
    # img = frame = cv2.bilateralFilter(frame, 9, 75, 75)  # Bilateral Filter for smoothing filter

    img = cv2.Canny(frame, parameter1, parameter2, intApertureSize)  # Canny edge detection
    # img = cv2.Laplacian(frame,cv2.CV_64F) # Laplacian edge detection
    # img = cv2.Sobel(frame,cv2.CV_64F,1,0,ksize=kernelSize) # X-direction Sobel edge detection
    # img = cv2.Sobel(frame,cv2.CV_64F,0,1,ksize=kernelSize) # Y-direction Sobel edge detection

    # convert to Gray_Scale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    imgS = cv2.resize(img, (680, 480))   # Resize image
    grayS = cv2.resize(gray, (680,480))  # Resize image

    # Display the resulting frame
    cv2.imshow('Webcam_View', grayS)
    cv2.imshow('Edge Detection', imgS)


    if cv2.waitKey(1) & 0xFF == ord('q'):  # press q to quit
        break
# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()