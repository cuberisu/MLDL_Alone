# 합성곱 신경망 convolutional neural network

# 케라스 합성곱 층
from tensorflow import keras
keras.layers.Conv2D(10, kernal_size=(3, 3), activation='relu')
# 필터 개수, 커널 크기, 활성화 함수

# 케라스의 패딩 설정
keras.layers.Conv2D(10, kernal_size=(3, 3), activation='relu', padding='same')

# 케라스의 스트라이드 설정 (기본값은 1)
keras.layers.Conv2D(10, kernal_size=(3, 3), activation='relu', padding='same', strides=1)

# 케라스의 풀링 층
keras.layers.MaxPooling2D(2)
keras.layers.MaxPooling2D(2, strides=2, padding='valid')
# strides, padding은 입력하지 않아도 자동으로 설정된다.