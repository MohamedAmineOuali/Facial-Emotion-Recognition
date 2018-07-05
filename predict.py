import datetime

import cv2

from models.simple_cnn_model import SimpleCnnModel
from models.vgg_models import Vgg16Model

model=SimpleCnnModel(False,sequence=True,nbSequence=1,stateful=True)

model.model.summary()

model.load_weights("weights/simple_cnn_based/sequence/statefull/86-86.hdf5")

# model=Vgg16Model(False)
#
# model.model.summary()
#
# model.load_weights("weights/vgg_based/static/wild_validation/87-64.hdf5")
#


face_cascade = cv2.CascadeClassifier('pre_processing/shared/haarcascades_cuda/haarcascade_frontalface_default.xml')


def predict(img):
    img=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    faces = face_cascade.detectMultiScale(img, 1.3, 5)
    if(len(faces)==0):
        return None
    x, y, w, h = faces[0]

    img = img[y:y + h, x:x + w]

    return model.predict(img)


EMOTIONS={"Angry":0,"Disgusted":1,"fear":2,"happy":3, "Sad":4 ,"surprised":5,"neutral":6}

def show_webcam(mirror=False):
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    out = cv2.VideoWriter('examples/{}.avi'.format(datetime.datetime.now().__str__()), fourcc, 20.0, (640, 480))
    cam = cv2.VideoCapture(0)
    i=1
    while True:
        ret_val, img = cam.read()

        if(img is not None):
            result=predict(img)
            if result is not None:
                for emotion, index in EMOTIONS.items():
                    cv2.putText(img, emotion, (10, index * 20 + 20), cv2.FONT_HERSHEY_PLAIN, 0.5, (0, 255, 0), 1)
                    cv2.rectangle(img, (130, index * 20 + 10),
                                  (130 + int(result[0][index] * 100), (index + 1) * 20 + 4), (255, 0, 0), -1)


        i+=1
        if mirror:
            img = cv2.flip(img, 1)

        out.write(img)
        cv2.imshow('my webcam', img)
        if cv2.waitKey(50) == 27:
            break  # esc to quit

    out.release()
    cam.release()
    cv2.destroyAllWindows()



show_webcam()

