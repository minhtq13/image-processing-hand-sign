import cv2
from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import numpy as np
import math

# Khởi tạo camera
cap = cv2.VideoCapture(0)

# Khởi tạo bộ nhận dạng tay
detector = HandDetector(maxHands=1)

# Khởi tạo bộ phân loại
classifier = Classifier("Model/keras_model.h5", "Model/labels.txt")

# Thiết lập các thông số
offset = 20
imgSize = 300
folder = "Data/C"
counter = 0

# Các nhãn cho từng cử chỉ tay
labels = ["Nam dam", "4 ngon", "Chu C", "5 ngon", "2 ngon", "1 ngon", "3 ngon"]

while True:
    # Đọc từng frame từ camera
    success, img = cap.read()
    imgOutput = img.copy()

    # Tìm tay trong hình ảnh
    hands, img = detector.findHands(img)

    if hands:
        # Lấy tay đầu tiên trong danh sách
        hand = hands[0]
        x, y, w, h = hand['bbox']

        # Tạo ảnh trắng để thực hiện xử lý
        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8) * 255

        # Cắt và điều chỉnh kích thước ảnh của vùng tay
        imgCrop = img[y - offset:y + h + offset, x - offset:x + w + offset]
        imgCropShape = imgCrop.shape
        aspectRatio = h / w

        if aspectRatio > 1:
            # Điều chỉnh kích thước theo tỷ lệ dọc
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        else:
            # Điều chỉnh kích thước theo tỷ lệ ngang
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize
            prediction, index = classifier.getPrediction(imgWhite, draw=False)

        # cv2.rectangle(imgOutput, (x - offset, y - offset-50),
        #               (x - offset+90, y - offset-50+50), (255, 0, 255), cv2.FILLED)

        # Hiển thị nhãn dự đoán lên hình ảnh và vẽ hình chữ nhật xung quanh vùng tay
        cv2.putText(imgOutput, labels[index], (x, y - 26),
                    cv2.FONT_HERSHEY_COMPLEX, 1.7, (255, 255, 255), 2)
        cv2.rectangle(imgOutput, (x - offset, y - offset),
                      (x + w + offset, y + h + offset), (255, 0, 255), 4)

        # Hiển thị ảnh cắt và ảnh trắng
        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    # Hiển thị ảnh gốc với nhãn dự đoán và hình chữ nhật xung quanh vùng tay
    cv2.imshow("Image", imgOutput)
    cv2.waitKey(1)
