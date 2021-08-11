import numpy as np
import imutils
import time
import cv2
import os
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream


def detect_and_predict_mask(frame, faceNet, maskNet):


	(h, w) = frame.shape[:2] #높이,너비
	blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
		(104.0, 177.0, 123.0))


	faceNet.setInput(blob)
	detections = faceNet.forward()
	print(detections.shape)

	
	faces = [] 
	locs = [] # 객체 인식 위치
	preds = []


	for i in range(0, detections.shape[2]):
		confidence = detections[0, 0, i, 2] # Detection된 확률

		# 인식한 값 중에서 confidence가 낮은 값들을 필터링하기
		if confidence > 0.5:
			
			box = detections[0, 0, i, 3:7] * np.array([w, h, w, h]) # box로 인삭한 객체의 좌표 값 계산
			(startX, startY, endX, endY) = box.astype("int")

			# 프레임 내에 box가 있는지 확인
			(startX, startY) = (max(0, startX), max(0, startY))
			(endX, endY) = (min(w - 1, endX), min(h - 1, endY))

			# 인식한 객체(얼굴)의 ROI 추출
			face = frame[startY:endY, startX:endX]
			face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)  # BGR을 RGB 채널로 변환
			face = cv2.resize(face, (224, 224)) # 사이즈 수정
			face = img_to_array(face) 
			face = preprocess_input(face)

			
			faces.append(face) # 객체 인식했으면 face 값에 추가
			locs.append((startX, startY, endX, endY)) # 객체 인식했으면 위치값에도 추가 

	# 얼굴이 인식될 때만!!! 
	if len(faces) > 0:
		
		faces = np.array(faces, dtype="float32") # 원핫 인코딩 한 값을 정수로 변환한다.
		preds = maskNet.predict(faces, batch_size=32)


	return (locs, preds) #인식했으면 객체의 위치, 예측값 반환


prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel" # 학습된 모델의 가중치
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# face mask detector model 불러오기
maskNet = load_model("mask_detector.model")

# cam 영상 띄우기
print("starting video ...")
vs = VideoStream(src=0).start()


while True:
	
	frame = vs.read()
	frame = imutils.resize(frame, width=400)

	# 마스크 착용유무 판별
	(locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)


	# 영상에 인식되는 사물의 위치별 box 표시
	for (box, pred) in zip(locs, preds):

		(startX, startY, endX, endY) = box
		(mask, withoutMask) = pred


		label = "Mask" if mask > withoutMask else "No Mask"
		color = (0, 255, 0) if label == "Mask" else (0, 0, 255) #색상은 RBG 순으로 'Mask'는 Green, 'Nomask'는 Red 색상으로 출력 


		label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100) #출력되는 확률 값의 label은 소수점 2째자리 값 까지만 나오도록 



		cv2.putText(frame, label, (startX, startY - 10),
			cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
		cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

	
	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF

	# 'q' 버튼 누르면 cam 종료
	if key == ord("q"):
		break

# cam 종료와 함께 초기화 
cv2.destroyAllWindows()
vs.stop()