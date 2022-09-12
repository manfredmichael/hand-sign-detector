from HandSignModel.HandSignModel import HandSignModel 
import cv2
from utils import draw_bounding_box

import time

model = HandSignModel(onnx_path="checkpoints/yolov4_tiny_1_3_416_416_static.onnx")

def test():
    img = cv2.imread("test_images/B19_jpg.rf.69527cc1f34d694cc04e55db80ed9b1a.jpg")
    img = cv2.imread("test_images/A1_jpg.rf.c4ccc21338f79e0f68d89dfc817ddd1f.jpg")
    img = cv2.imread("test_images/D9_jpg.rf.3250856285c1a522ba86b3135b5dd6bc.jpg")
    img = cv2.imread("test_images/E0_jpg.rf.926e842cd69d98b54aec8c371d61bf8d.jpg")

# # Test Predict
    model.predict(img)

def webcam_inference():
    cap= cv2.VideoCapture(0)

    # used to record the time when we processed last frame
    prev_frame_time = 0
     
    # used to record the time at which we processed current frame
    new_frame_time = 0
 
    while True:
        _, frame = cap.read()
        # prep_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prep_frame = cv2.GaussianBlur(frame,(3,3),0)

        new_frame_time = time.time()
        fps = int(1/(new_frame_time-prev_frame_time))
        prev_frame_time = new_frame_time
        frame = cv2.rectangle(frame, (0, 0), (30, 30), (0, 102, 255), -1)
        frame = cv2.putText(frame, str(fps), (3,20),cv2.FONT_HERSHEY_SIMPLEX,0.5,(255,255,255), 1)

        t0 = time.time()
        boxes = model.predict(prep_frame)
        
        print("latency: ", time.time() - t0)
        if not boxes:
            print(boxes)
            continue

        frame = draw_bounding_box(frame, boxes)
        cv2.imshow('hand sign detector', frame)


     
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    webcam_inference()
