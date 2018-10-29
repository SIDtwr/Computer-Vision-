import argparse
import cv2


# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="path to the input image")
args = vars(ap.parse_args())

im = cv2.imread(args["image"])
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)

thresh = cv2.threshold(imgray, 60, 255, cv2.THRESH_BINARY)[1]

# find contours in the threshold image and initialize the shape detector
contours = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
contours = contours[1]

cnt = contours[0] # if the program returns a compilation error set to 1

for cnt in contours:

    # calculating the perimeter and the aprox vertices of the contour
    shape = "unidentified"
    perimeter = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)

    # if the shape is a triangle, it will have 3 vertices
    if len(approx) == 3:
        shape = "triangle"

    # if the shape has 4 vertices, it is either a square or
    # a rectangle
    elif len(approx) == 4:
        # compute the bounding box of the contour and use the
        # bounding box to compute the aspect ratio
        (x, y, w, h) = cv2.boundingRect(approx)
        ar = w / float(h)

        # a square will have an aspect ratio that is approximately
        # equal to one, otherwise, the shape is a rectangle
        shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"

    # if it's a polygon, it will have more than 4 vertices
    elif len(approx) > 4:
        shape = "polygon"

    # compute the center of the contour
    M = cv2.moments(cnt)
    cX = int((M["m10"] / M["m00"]))
    cY = int((M["m01"] / M["m00"]))

    # then draw the contours and specifying the shape
    cv2.drawContours(im, cnt, -1, (0, 255, 0), 2)
    cv2.putText(im, shape, (cX, cY), cv2.FONT_HERSHEY_SIMPLEX,0.5, (255, 255, 255), 2)

cv2.imshow("ShapeDetetction", im)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()



