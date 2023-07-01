# 인공 신경망

# 실행마다 동일한 결과를 얻기 위해 케라스에 랜덤 시드를 사용할고 텐서플로 연산을 결정적으로 만든다.
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# 패션 MNIST: 딥러닝에서 기본이 되는 이미지 데이터셋
from tensorflow import keras
(train_input, train_target), (test_input, test_target) = keras.datasets.fashion_mnist.load_data()
# load_data()는 훈련세트와 테스트세트에 각각 데이터를 넣어준다.

# print(train_input.shape, train_target.shape)    # (60000, 28, 28) (60000,)
# print(test_input.shape, test_target.shape)      # (10000, 28, 28) (10000,)

import matplotlib.pyplot as plt
fig, axs = plt.subplots(1, 10, figsize=(10, 10))
for i in range(10):
    axs[i].imshow(train_input[i], cmap='gray_r')    # 그림을 반전시킨다
    axs[i].axis('off')
plt.show()

print([train_target[i] for i in range(10)]) # [9, 0, 0, 3, 0, 2, 7, 2, 5, 5]


# unique()로 클래스 분포를 확인. return_counts=True 클래스 정수값이 얼마가 있고 얼마만큼 있는가를 확인시켜줌
import numpy as np
print(np.unique(train_target, return_counts=True))  
# (array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=uint8), 
# array([6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000, 6000],
# dtype=int64))

# 로지스틱 회귀로 패션 아이템 분류
# 경사하강법을 이용한 로지스틱 회귀. 10개의 이진분류 (OvR 또는 OvA)
# max_iter=epoch
train_scaled = train_input / 255.0  # 이미지의 픽셀값이 255까지이기 때문에 이로 나누어 0~1로 만든다.
train_scaled = train_scaled.reshape(-1, 28*28)  # 3차원 데이터를 1차원으로 펼치기 위해 reshape()
print(train_scaled.shape)   # (60000, 784)

from sklearn.model_selection import cross_validate
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss', max_iter=5, random_state=42)    
scores = cross_validate(sc, train_scaled, train_target, n_jobs=-1)  # 교차검증

print(np.mean(scores['test_score']))    # 0.8192833333333333


# 인공신경망으로 모델 만들기
import tensorflow as tf
from tensorflow import keras

from sklearn.model_selection import train_test_split
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42) # 20% 떼어냄

print(train_scaled.shape, train_target.shape)   # (48000, 784) (48000,)
print(val_scaled.shape, val_target.shape)       # (12000, 784) (12000,)

dense = keras.layers.Dense(10, activation='softmax', input_shape=(784,))
# 밀집층, 완전 연결층(fully connected layer) 
# 10개 뉴런(유닛), softmax(다중함수), 
# 입력층이 없고 출력층 하나(첫번째 모델에 추가되는 층엔 input_shape 지정해줘야 한다)
# input_shape의 크기는 샘플의 크기

model = keras.Sequential(dense) # 모델을 만들어 dense를 집어넣는다.


# 인공신경망으로 패션 아이템 분류하기
# 모델 설정 compile() (손실함수 설정, 손실값 기록, 정확도 기록 등)
model.compile(loss='sparse_categorical_crossentropy', metrics='accuracy')

print(train_target[:10])
model.fit(train_scaled, train_target, epochs=5)

model.evaluate(val_scaled, val_target)