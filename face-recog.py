import cv2
import os
import numpy as np

DATASET_DIR = "dataset"
MODEL_PATH = "face_model.yml"
LABELS_PATH = "labels.txt"

DNN_PROTO = "deploy.prototxt"
DNN_MODEL = "res10_300x300_ssd_iter_140000.caffemodel"


def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def load_labels(path):
    labels = {}
    if not os.path.exists(path):
        return labels
    with open(path, "r") as f:
        for line in f:
            id, name = line.strip().split(",")
            labels[int(id)] = name
    return labels


def save_labels(label_map):
    with open(LABELS_PATH, "w") as f:
        for id, name in label_map.items():
            f.write(f"{id},{name}\n")


def detect_faces_dnn(frame, net):
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300),
                                 (104.0,177.0,123.0))
    net.setInput(blob)
    detections = net.forward()

    faces = []
    for i in range(detections.shape[2]):
        conf = detections[0,0,i,2]
        if conf > 0.4:     # more sensitive
            box = detections[0,0,i,3:7] * [w,h,w,h]
            x1,y1,x2,y2 = box.astype(int)
            faces.append((x1,y1,x2-x1,y2-y1))
    return faces


# capturing images
def capture_faces():
    name = input("Person name: ").strip()
    if not name:
        print("Invalid name")
        return

    net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)

    normal_dir = os.path.join(DATASET_DIR, name, "normal")
    ensure_dir(normal_dir)

    cap = cv2.VideoCapture(0)
    count = 0
    target = 60

    print("Capturing NORMAL… press q to stop")

    while True:
        ret, frame = cap.read()
        faces = detect_faces_dnn(frame, net)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            face = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(200,200))

            count += 1
            cv2.imwrite(os.path.join(normal_dir, f"{count}.jpg"), gray)

            cv2.putText(frame,f"Normal {count}/{target}",(20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            cv2.waitKey(200)

            if count >= target:
                cap.release()
                cv2.destroyAllWindows()
                print("Normal done!")
                break

        cv2.imshow("Capture", frame)
        if count >= target: break
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


    input("\nPut mask and press ENTER to continue…")

    masked_dir = os.path.join(DATASET_DIR, name, "masked")
    ensure_dir(masked_dir)

    cap = cv2.VideoCapture(0)
    count2 = 0
    target2 = 60

    print("Capturing MASKED… press q to stop")

    while True:
        ret, frame = cap.read()
        faces = detect_faces_dnn(frame, net)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            face = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(200,200))

            count2 += 1
            cv2.imwrite(os.path.join(masked_dir, f"{count2}.jpg"), gray)

            cv2.putText(frame,f"Masked {count2}/{target2}",(20,30),
                        cv2.FONT_HERSHEY_SIMPLEX,0.7,(0,255,0),2)

            cv2.waitKey(200)

            if count2 >= target2:
                cap.release()
                cv2.destroyAllWindows()
                print("Masked done!")
                return

        cv2.imshow("Capture", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

    cap.release()
    cv2.destroyAllWindows()


# train

def train_model():
    faces=[]
    labels=[]
    label_map={}
    current=0

    for person in os.listdir(DATASET_DIR):
        person_path = os.path.join(DATASET_DIR, person)
        if not os.path.isdir(person_path): continue

        for typ in ["normal","masked"]:
            folder = os.path.join(person_path, typ)
            if not os.path.exists(folder): continue

            label_map[current] = f"{person}-{typ}"

            for imgfile in os.listdir(folder):
                p = os.path.join(folder,imgfile)
                img = cv2.imread(p,cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                img = cv2.resize(img,(200,200))
                faces.append(img)
                labels.append(current)

            current += 1

    if not faces:
        print("No images!")
        return

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.train(faces, np.array(labels))
    rec.save(MODEL_PATH)
    save_labels(label_map)
    print("Training done!")



# recognize
def recognize():
    if not os.path.exists(MODEL_PATH):
        print("Train first!")
        return

    rec = cv2.face.LBPHFaceRecognizer_create()
    rec.read(MODEL_PATH)

    labels = load_labels(LABELS_PATH)

    net = cv2.dnn.readNetFromCaffe(DNN_PROTO, DNN_MODEL)
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        faces = detect_faces_dnn(frame, net)

        for (x,y,w,h) in faces:
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)

            face = frame[y:y+h, x:x+w]
            gray = cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
            gray = cv2.resize(gray,(200,200))

            id,conf = rec.predict(gray)
            label = labels.get(id,"Unknown")

            if conf > 80:
                display="Unknown"
            else:
                if "-" in label:
                    person,mask=label.split("-")
                    display=f"{person} ({mask})"
                else:
                    display=label

            cv2.putText(frame,display,(x,y-10),
                        cv2.FONT_HERSHEY_SIMPLEX,0.8,(0,255,0),2)

        cv2.imshow("Recognition",frame)
        if cv2.waitKey(1)&0xFF==ord('q'): break

    cap.release()
    cv2.destroyAllWindows()



def main():
    ensure_dir(DATASET_DIR)

    while True:
        print("\n1. Capture")
        print("2. Train")
        print("3. Recognize")
        print("0. Exit")

        c=input("Choice: ")

        if c=="1": capture_faces()
        elif c=="2": train_model()
        elif c=="3": recognize()
        elif c=="0": break


if __name__=="__main__":
    main()
