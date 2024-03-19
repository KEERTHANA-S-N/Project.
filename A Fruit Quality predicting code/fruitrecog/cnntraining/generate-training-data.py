import os
import cv2
import argparse

TRAINING_DATA_DIR = "training_data"

def capture_images(label, label_path, sample_size):
    # open camera (device : 0) in video capture mode
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error opening video")

    capture = False
    count = 0

    while True:
        
        ret, frame = cap.read()
        # checking if the image was retrived
        if not ret:
            print("Error getting image")
            continue

        # if required number of images have been captured
        if count == sample_size:
            break
        
        # to start capturing images for training once the key is hit
        if capture:
            img_path = os.path.join(label_path, '{}.jpg'.format(count + 1))
            # saving the image
            cv2.imwrite(img_path, frame)
            count += 1
        
        
        cv2.rectangle(frame, (75, 75), (300, 300), (0, 0, 255), 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        # for displaying the text
        cv2.putText(frame, "Images Collected: {}".format(count), (70, 325), font, 0.7, (0, 0, 255), 2, cv2.LINE_AA)

        # Displaying the Images which are to be captured
        cv2.imshow("Collecting Images", frame)

        k = cv2.waitKey(10)
        if k == ord('c'):
            capture = not capture
        if k == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    msglbl=input('enter the msg to train')
    imgcount=int(input('enter no of images to train'))

    # training_data folder creation
    if not os.path.exists(TRAINING_DATA_DIR):
        print("Creating output directory " + TRAINING_DATA_DIR)
        os.makedirs(TRAINING_DATA_DIR)

    label_path = TRAINING_DATA_DIR + "//" + msglbl
    # label folder creattion
    if not os.path.exists(label_path):
        print("Creating output directory " + label_path)
        os.makedirs(label_path)
    else:
        print("Label directory exists: generated images will be appended to the existing data")

    capture_images(msglbl, label_path, imgcount)


if __name__ == "__main__":
    main()
