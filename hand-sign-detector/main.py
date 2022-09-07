from HandSignModel.HandSignModel import HandSignModel 
import cv2

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
 
    while True:
        _, frame = cap.read()
        t0 = time.time()
        boxes = model.predict(frame)
        print("latency: ", time.time() - t0)
        if not boxes:
            print(boxes)
            continue
        for box in boxes[0]:
            width = frame.shape[1]
            height = frame.shape[0]
            x = int(box[0] * width)
            y = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            w = x2 - x
            h = y2 - y
            confidence = box[4]
            label = box[5]
            cv2.rectangle(frame,(x,y,w,h),(0,255,0),2)
            cv2.putText(frame, "{}: {:.3f}".format(label, confidence), (w+10,y+h),0,1,(255,0,0))
        cv2.imshow('hand sign detector', frame)


     
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            cap.release()
            cv2.destroyAllWindows()
            break


if __name__ == "__main__":
    webcam_inference()
