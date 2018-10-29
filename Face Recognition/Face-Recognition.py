# Face Recognition with OpenCV
import cv2
import os
import numpy as np



# there is no label 0 in our training data so subject name for index/label 0 is empty
subjects = [ "ElonMusk", "BillGates", "SteveJobs"]



#function to detect face using OpenCV
def extract_face(img):
    # convert the test image to gray image as opencv face detector expects gray images
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # load OpenCV face detector, I am using LBP which is fast
    # there is also a more accurate but slow Haar classifier
    face_cascade = cv2.CascadeClassifier('opencv-files/lbpcascade_frontalface.xml')

    # let's detect multiscale (some images may be closer to camera than others) images
    # result is a list of faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.2, minNeighbors=5);

    # if no faces are detected then return original img
    if (len(faces) == 0):
        return None, None

    #under the assumption that there will be only one face,
    #extract the face area
    (x, y, w, h) = faces[0]

    #return only the face part of the image
    return gray[y:y+w, x:x+h], faces[0]


def training_data(training_folder_path):


    # get the directories (one directory for each subject) in data folder
    dirs = os.listdir(training_folder_path)

    # list to hold all subject faces
    faces = []
    # list to hold labels for all subjects
    labels = []

    # let's go through each directory and read images within it
    for dir_name in dirs:


        if not dir_name.startswith("s"):
            continue;

        # each directory starts with "sample" suffixed with the integer sequence. eg - sample1, sample2 .
        label = int(dir_name.replace("sample", "")) # removing sample from the name
        subject_dir_path = training_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path) #adding the integer suffix to the list.

        #detect face and add face to list of faces
        for image_name in subject_images_names:

            # ignore system files like .DS_Store
            if image_name.startswith("."):
                continue;

            # build image path
            # sample image path = training-data/s1/1.pgm
            image_path = subject_dir_path + "/" + image_name

            # read image
            image = cv2.imread(image_path)

            #detect face
            face, rect = extract_face(image)

           # ignoring faces that are not detected
            if face is not None:
                # add face and labels to list
                faces.append(face)
                labels.append(label)

    cv2.destroyAllWindows()
    cv2.waitKey(1)
    cv2.destroyAllWindows()

    return faces, labels



print(" Preparing Data: This might thake time")
# preparing training data
faces, labels = training_data("training-data")
print("Data prepared")


# create our LBPH face recognizer
face_recognizer = cv2.face.LBPHFaceRecognizer_create()

# or use EigenFaceRecognizer by replacing above line with
# face_recognizer = cv2.face.EigenFaceRecognizer_create()

# or use FisherFaceRecognizer by replacing above line with
# face_recognizer = cv2.face.FisherFaceRecognizer_create()


# training the reconizer
face_recognizer.train(faces, np.array(labels))



def prediction(test_img):
    # make a copy of the image as we don't want to chang original image
    img = test_img.copy()
    # detect face from the image
    face, rect = extract_face(img)

    # predict the image using our face recognizer
    label, confidence = face_recognizer.predict(face)
    # get name of respective label returned by face recognizer
    label_text = subjects[label-1]
    
    # draw a rectangle around face detected
    (x, y, w, h) = rect
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # draw name of predicted person
    cv2.putText(img, label_text, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    
    return img


print("Predicting images...")

# load test images
test_img1 = cv2.imread("test-data/test1.jpg")
test_img2 = cv2.imread("test-data/test2.jpg")
test_img3 = cv2.imread("test-data/test3.jpg")

# making a prediction
predicted_img1 = prediction(test_img1)
# predicted_img2 = predict(test_img2)
predicted_img3 = prediction(test_img3)
print("Prediction complete")

# display both images
cv2.imshow(subjects[0], cv2.resize(predicted_img1, (400, 500)))
# cv2.imshow(subjects[1], cv2.resize(predicted_img2, (400, 500)))
cv2.imshow(subjects[2], cv2.resize(predicted_img3, (400, 500)))

cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.waitKey(1)
cv2.destroyAllWindows()





