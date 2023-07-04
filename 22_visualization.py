# 가중치 시각화

# 실행마다 동일한 결과를 얻기 위해 사용하고 텐서플로 연산을 결정적으로 만듭니다. 
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()


# 층의 가중치 분포
from tensorflow import keras
model = keras.models.load_model('best-cnn-model.h5')    # 이전 checkpoint 파일을 불러온다.
print(model.layers) # 층들을 확인할 수 있다.

# 첫 번째 층의 가중치 조사
conv = model.layers[0]
print(conv.weights[0].shape, conv.weights[1].shape) # (3, 3, 1, 32) (32,)
# weight[]의 첫 번째 원소가 가중치, 두 번째 원소가 절편

conv_weights = conv.weights[0].numpy()  # numpy 배열로 변환
print(conv_weights.mean(), conv_weights.std())  # 평균, 표준 편차 확인

# 가중치 시각화
# 가중치 히스토그램
import matplotlib.pyplot as plt
plt.hist(conv_weights.reshape(-1, 1))   # 히스토그램을 그리려면 1차원 배열로 전달해야 함
plt.xlabel('weight')
plt.ylabel('count')
plt.show()


# subplots()로 가중치 시각화
# vmin과 vmax로 컬러맵으로 표현할 범위를 지정
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(conv_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# 훈련하지 않은 빈 합성곱 신경망
no_training_model = keras.Sequential()
no_training_model.add(keras.layers.Conv2D(32, kernel_size=3, activation='relu',
                                          padding='same', input_shape=(28,28,1)))

# 훈련하지 않은 가중치
no_training_conv = no_training_model.layers[0]
print(no_training_conv.weights[0].shape)    # (3, 3, 1, 32)

# 훈련하지 않은 평균과 표준 편차
no_training_weights = no_training_conv.weights[0].numpy()
print(no_training_weights.mean(), no_training_weights.std())

# 훈련하지 않은 히스토그램
plt.hist(no_training_weights.reshape(-1,1))
plt.xlabel('weight')
plt.ylabel('count')
plt.show()

# 훈련하지 않은 가중치 시각화
fig, axs = plt.subplots(2, 16, figsize=(15,2))
for i in range(2):
    for j in range(16):
        axs[i, j].imshow(no_training_weights[:,:,0,i*16 + j], vmin=-0.5, vmax=0.5)
        axs[i, j].axis('off')
plt.show()

# 함수형 API 층들을 함수처럼 쓸 수 있다.
# model = keras.Model(입력, 출력)
print(model.input)  # input은 Sequential() 객체를 만들 때 자동으로 생성된다.
conv_acti = keras.Model(model.input, model.layers[0].output)    # 1번째 층인 Conv2D 층을 출력층으로 삼는다.

(train_input, train_target), (test_input, test_target) = \
    keras.datasets.fashion_mnist.load_data()    # 데이터 로드

plt.imshow(train_input[0], cmap='gray_r')   # 첫 번째 그림 그리기
plt.show()  # 신발 그림이 나온다.

# 특성 맵 시각화
inputs = train_input[0:1].reshape(-1, 28, 28, 1) / 255.0
feature_maps = conv_acti.predict(inputs)    # input을 입력해서 Conv2D 층만 통과한 특성맵
print(feature_maps.shape)   # (1, 28, 28, 32)

fig, axs = plt.subplots(4, 8, figsize=(15,8))
for i in range(4):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8+j]) # imshow 시각화 함수
        axs[i, j].axis('off')
plt.show()
# 32개의 필터로 인해 입력 이미지에서 강하게 활성화된 부분을 보여준다.

# 두 번째 합성곱 층 특성 맵 시각화
conv2_acti = keras.Model(model.input, model.layers[2].output)
feature_maps = conv2_acti.predict(train_input[0:1].reshape(-1, 28, 28, 1)/255.0)

print(feature_maps.shape)   # (1, 14, 14, 64)

fig, axs = plt.subplots(8, 8, figsize=(12,12))
for i in range(8):
    for j in range(8):
        axs[i, j].imshow(feature_maps[0,:,:,i*8+j])
        axs[i, j].axis('off')
plt.show()