# import the necessary packages
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D #딥러닝 학습
from tensorflow.keras.layers import Dropout #딥러닝 학습
from tensorflow.keras.layers import Flatten #딥러닝 학습
from tensorflow.keras.layers import Dense #딥러닝 학습
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model #딥러닝 학습
from tensorflow.keras.optimizers import Adam #딥러닝 학습
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer #마스크 착용 유무를 분류하기
from sklearn.model_selection import train_test_split 
from sklearn.metrics import classification_report
from imutils import paths




INIT_LR = 1e-4 #initial learning rate
EPOCHS = 20 #더 이상 올려도 성능에는 영향이 없음.
BS = 32 #batch_size 줄임말

#colab으로 실행했음.
#colab에서 실행 시에 구글드라이브에 폴더를 올려서 경로 확인 잘하고 실행할 것!
DIRECTORY = r"C:\ai_01\Project_1\Face-Mask-Detection\dataset" #실행 시 경로 수정해야함! 
CATEGORIES = ["with_mask", "without_mask"] 

# 데이터셋에 있는 이미지 불러와서 라벨링 해주기.
print("[INFO] loading images...")
#각각의 이미지 불러오면 data,labels에 지정해두기
data = []
labels = []

for category in CATEGORIES:
    path = os.path.join(DIRECTORY, category)
    for img in os.listdir(path):
    	img_path = os.path.join(path, img)
    	image = load_img(img_path, target_size=(224, 224)) 
    	image = img_to_array(image)
    	image = preprocess_input(image)

    	data.append(image) #각각의 이미지가 들어오면 data에 append로 추가해주기.
    	labels.append(category) #각각의 이미지가 들어오면 labels에 append로 추가해주기


# 마스크 착용 유무를 labels에 one-hot 인코딩 해주기
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)

data = np.array(data, dtype="float32") # data에 저장되는 one-hot 인코딩 된 값을 정수로 변환
labels = np.array(labels)

(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.20, stratify=labels, random_state=42) # 모델링하기 위한 데이터 셋 분리하기

# construct the training image generator for data augmentation
aug = ImageDataGenerator(
	rotation_range=20,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")

# MobileNetV2 network (경량화 네트워크할 때 좋다고 함)
# Fully Connectec Layer (이미지 분류/설명하는 데 가장 적합하게 예측한다고 함)
baseModel = MobileNetV2(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))


# base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel) # 결과값은 2개만 나오면 됨  'Mask' or 'Nomask'



model = Model(inputs=baseModel.input, outputs=headModel)


for layer in baseModel.layers:
	layer.trainable = False

# 모델 컴파일
print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt,
	metrics=["accuracy"])

# network 학습
print("[INFO] training head...")
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

# test 셋 predict 만들기
print("[INFO] evaluating network...")
predIdxs = model.predict(testX, batch_size=BS)

# test set에서 각 이미지에 해당하는 가장 큰 predict 확률 가진 레이블 찾기
predIdxs = np.argmax(predIdxs, axis=1)

# 잘 분류한 모델 출력
print(classification_report(testY.argmax(axis=1), predIdxs,
	target_names=lb.classes_))


print("[INFO] saving mask detector model...")
model.save("mask_detector.model", save_format="h5")

# training loss and accuracy 출력
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")