import cv2
import os
dataPath="./examples"


for root, dirs, files in os.walk(dataPath):

    for filename in files:
        cap = cv2.VideoCapture(os.path.join(root,filename))

        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        output=os.path.join(dataPath,filename).replace('.avi','')
        os.makedirs(output, exist_ok=True)
        i=0
        while (cap.isOpened()):

            ret, frame = cap.read()
            if (not ret):
                break

            cv2.imwrite(os.path.join(output,i.__str__()+".jpeg"),frame)
            i+=1

        cap.release()


