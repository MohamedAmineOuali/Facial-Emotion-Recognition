import cv2


def cropping_face(sequence):
    face_cascade = cv2.CascadeClassifier('../shared/haarcascades_cuda/haarcascade_frontalface_default.xml')
    i = 0
    faces = []
    for image in sequence:
        face = face_cascade.detectMultiScale(image, 1.1, 5)
        if len(face) > 0:
            cur = face[0]
            while i < len(face) and face[i][2] > face[i - 1][2]:
                cur = face[i]
                i += 1
            if cur[2] > 50:
                faces.append(cur)
    w = 0
    h = 0
    x = 10000
    y = 10000
    for face in faces:
        w = max(w, face[2])
        h = max(h, face[3])
        x = min(x, face[0])
        y = min(y, face[1])

    return x, y, w, h


def process_video(filename, nbSequence,size=None):
    cap = cv2.VideoCapture(filename)
    if(cap is None):
        return None

    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if nbSequence==1:
        step=None
    else:
        step=int((length-1)/(nbSequence-1))-1

    if step is None:
        offset=length-1
    else:
        offset=(length-1)%(nbSequence-1)

    sequence=[]

    while (cap.isOpened()):

        ret, frame = cap.read()
        if(not ret):
            break

        if( offset):
            offset-=1
            continue

        offset=step
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if (size is not None):
            image=cv2.resize(image, size)

        sequence.append(image)


    cap.release()
    if(len(sequence)==0):
        return None

    return sequence


