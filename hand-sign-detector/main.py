from TomatoModel.TomatoModel import TomatoModel
import cv2

model = TomatoModel(onnx_path="checkpoints/yolov4_1_3_416_416_static.onnx")

img = cv2.imread("test_images/B19_jpg.rf.69527cc1f34d694cc04e55db80ed9b1a.jpg")
img = cv2.imread("test_images/A1_jpg.rf.c4ccc21338f79e0f68d89dfc817ddd1f.jpg")
img = cv2.imread("test_images/D9_jpg.rf.3250856285c1a522ba86b3135b5dd6bc.jpg")
img = cv2.imread("test_images/E0_jpg.rf.926e842cd69d98b54aec8c371d61bf8d.jpg")

# # Test Predict
model.predict(img)
