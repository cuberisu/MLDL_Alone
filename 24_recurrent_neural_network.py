# 순환 신경망으로 IMDB 리뷰 분류하기

# 실행마다 동일한 결과를 얻기 위해 사용하고 텐서플로 연산을 결정적으로 만듭니다. 
import tensorflow as tf
tf.keras.utils.set_random_seed(42)
tf.config.experimental.enable_op_determinism()

# IMDB 리뷰 데이터셋에서 가장 자주 등장하는 단어 300개 추리기
# 실제 IMDB 리뷰 데이터셋은 영어 문장이지만 텐서플로에는 이미 정수로 바꾼 데이터가 포함되어 있다.
from tensorflow.keras.datasets import imdb
(train_input, train_target), (test_input, test_target) = imdb.load_data(num_words=300)

print(train_input.shape, test_input.shape)  # (25000,) (25000,) 샘플 개수

print(len(train_input[0]))  # 218

print(len(train_input[1]))  # 189

print(train_input[0])   # 첫 번째 샘플 출력
# [1, 14, 22, 16, 43, 2, 2, 2, 2, 65, ...

print(train_target[:20])    # 이진 분류
# [1 0 0 1 0 0 1 0 1 0 1 0 0 0 0 0 1 1 0 1]

# 훈련 세트 준비
from sklearn.model_selection import train_test_split
train_input, val_input, train_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)  
# 20% 검증 세트로 떼어내 에포크마다 검증 점수 확인하여 조기종료에 활용

import numpy as np
lengths = np.array([len(x) for x in train_input])   # 샘플들을 모두 다 순환하면서 길이를 계산

print(np.mean(lengths), np.median(lengths)) # 평균값, 중간값

# 히스토그램
import matplotlib.pyplot as plt
plt.hist(lengths)
plt.xlabel('length')
plt.ylabel('frequency')
plt.show()

# 시퀀스 패딩
from tensorflow.keras.preprocessing.sequence import pad_sequences
train_seq = pad_sequences(train_input, maxlen=100)  # 문장을 100에 맞춰 패딩

print(train_seq.shape)  # (20000, 100)

print(train_seq[0])
# [ 10   4  20   9   2   2   2   5  45   6   2   2  33 269   8   2 142   2
# ...
#   6   2  46   7  14  20  10  10   2 158
print(train_input[0][-10:])  # 앞에가 잘렸다는 것을 확인할 수 있다.
# [6, 2, 46, 7, 14, 20, 10, 10, 2, 158]
print(train_seq[5]) # 앞에 패딩됨
# [  0   0   0   0   1   2 195  19  49   2   2 190   4   2   2   2 183  10 ...

val_seq = pad_sequences(val_input, maxlen=100)


# 순환 신경망 만들기
from tensorflow import keras
model = keras.Sequential()

model.add(keras.layers.SimpleRNN(8, input_shape=(100, 300)))    # 순환신경망 층
model.add(keras.layers.Dense(1, activation='sigmoid'))

train_oh = keras.utils.to_categorical(train_seq)    # 원-핫 인코딩

print(train_oh.shape)   # (20000, 100, 300)

print(train_oh[0][0][:12])  # [0. 0. 0. 0. 0. 0. 0. 0. 0. 0. 1. 0.]

print(np.sum(train_oh[0][0]))   # 1.0 원-핫 인코딩이기 때문에

val_oh = keras.utils.to_categorical(val_seq)

model.summary()


# 순환 신경망 훈련하기
rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])
checkpoint_cb = keras.callbacks.ModelCheckpoint('best-simplernn-model.h5', save_best_only=True)
early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model.fit(train_oh, train_target, epochs=100, batch_size=64, 
                    validation_data=(val_oh, val_target),
                    callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()


# 단어 임베딩을 사용하기 (원-핫 인코딩이 아니라 실수 벡터로 변환)
model2 = keras.Sequential()
model2.add(keras.layers.Embedding(300, 16, input_length=100))
model2.add(keras.layers.SimpleRNN(8))
model2.add(keras.layers.Dense(1, activation='sigmoid'))

model2.summary()

rmsprop = keras.optimizers.RMSprop(learning_rate=1e-4)
model2.compile(optimizer=rmsprop, loss='binary_crossentropy', metrics=['accuracy'])

checkpoint_cb = keras.callbacks.ModelCheckpoint('best-embedding-model.h5', save_best_only=True)

early_stopping_cb = keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)

history = model2.fit(train_seq, train_target, epochs=100, batch_size=64, 
                     validation_data=(val_seq, val_target), 
                     callbacks=[checkpoint_cb, early_stopping_cb])

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.xlabel('epoch')
plt.ylabel('loss')
plt.legend(['train', 'val'])
plt.show()