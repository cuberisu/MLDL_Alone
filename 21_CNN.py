# Convolutional Neural Network
# 실행마다 동일한 결과를 얻기 위해 사용하고 텐서플로 연산을 결정적으로 만든다.
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


# 패션 MNIST 데이터 불러오기
from tensorflow import keras
from sklearn.model_selection import train_test_split
(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()

train_scaled = train_input.reshape(-1, 28, 28, 1) / 255.0   # 3차원으로 만든다.
train_scaled, val_scaled, train_target, val_target = train_test_split(
    train_scaled, train_target, test_size=0.2, random_state=42)


# 합성곱 신경망 만들기
model = keras.Sequential()  # 신경망 모델 만들기

# 첫 번째 합성곱 층 Conv2D: 필터 32개, 커널 크기 3*3, 렐루 활성화 함수, 세임패딩, 3차원 입력층
model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                              padding='same', input_shape=(28, 28, 1)))

model.add(keras.layers.MaxPooling2D(2)) # 2*2 풀링층

# 합성곱 층 Conv2D: 필터 64개, 커널 크기 3*3, 렐루 활성화 함수, 세임패딩
model.add(keras.layers.Conv2D(64, kernel_size=(3, 3), activation='relu',
                              padding='same'))

model.add(keras.layers.MaxPooling2D(2)) # 2*2 풀링층

model.add(keras.layers.Flatten())   # 출력층에서의 계산을 위해 일렬로 펼침
model.add(keras.layers.Dense(100, activation='relu'))   # 밀집 은닉층
model.add(keras.layers.Dropout(0.4))    # 드롭아웃 층
model.add(keras.layers.Dense(10, activation='softmax')) # 밀집 출력층

model.summary() # 모델 구조 출력

keras.utils.plot_model(model)   # 층의 구조를 그림으로 출력

keras.utils.plot_model(model, show_shapes=True) # 입력과 출력의 크기까지 출력


# 모델 컴파일과 훈련
# adam 옵티마이저, sparse_categorical_crossentropy 손실함수, 정확도 출력
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics='accuracy')
# 가장 낮은 손실함수를 저장하는 콜백
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-cnn-model.h5',
                                                save_best_only=True)
# 조기 종료 콜백(2번까지는 손실값이 높아져도 진행함)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=2, 
                                                  restore_best_weights=True)

history = model.fit(train_scaled, train_target, epochs=20, 
                    validation_data=(val_scaled, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

import matplotlib.pyplot as plt
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()

# 모델 평가
model.evaluate(val_scaled, val_target)

# 첫 번째 샘플 이미지 확인
plt.imshow(val_scaled[0].reshape(28, 28), cmap='gray_r')
plt.show()

# 예측 확인
preds = model.predict(val_scaled[0:1])
print(preds)

# 이 예측을 막대그래프로 그리기
plt.bar(range(1, 11), preds[0])
plt.xlabel('class')
plt.ylabel('prob.')
plt.show()

# 레이블을 다루기 위해 리스트로 저장
classes = ['티셔츠', '바지', '스웨터', '드레스', '코트',
           '샌달', '셔츠', '스니커즈', '가방', '앵클 부츠']

# preds 배열에서 가장 큰 인덱스를 찾아 classes 리스트의 인덱스로 사용
import numpy as np
print(classes[np.argmax(preds)])

# 맨 처음에 떼어 놓았던 테스트 세트로 합성곱 신경망의 일반화 성능을 가늠
test_scaled = test_input.reshape(-1, 28, 28, 1) / 255.0

# 성능 측정
model.evaluate(test_scaled, test_target)

