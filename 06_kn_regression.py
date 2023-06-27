# 농어의 길이로 무게를 예측하는 k-최근접 이웃 회귀 알고리즘

# 데이터 준비
# 이번엔 처음부터 numpy 배열로 준비해보자.
import numpy as np
# 농어 길이 데이터 56개
perch_length = np.array([8.4, 13.7, 15.0, 16.2, 17.4, 18.0, 18.7, 19.0, 19.6, 20.0, 21.0,
       21.0, 21.0, 21.3, 22.0, 22.0, 22.0, 22.0, 22.0, 22.5, 22.5, 22.7,
       23.0, 23.5, 24.0, 24.0, 24.6, 25.0, 25.6, 26.5, 27.3, 27.5, 27.5,
       27.5, 28.0, 28.7, 30.0, 32.8, 34.5, 35.0, 36.5, 36.0, 37.0, 37.0,
       39.0, 39.0, 39.0, 40.0, 40.0, 40.0, 40.0, 42.0, 43.0, 43.0, 43.5,
       44.0])
# 농어 무게 데이터 56개
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 110.0,
       115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 130.0,
       150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 197.0,
       218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 514.0,
       556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 820.0,
       850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 1000.0,
       1000.0])

# 산점도
import matplotlib.pyplot as plt
# plt.scatter(perch_length, perch_weight)
# plt.xlabel('length')
# plt.ylabel('weight')
# plt.show()


# 데이터 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

print(train_input.shape, test_input.shape)  # (42,) (14,)

# shape
test_array = np.array([1,2,3,4])
print(test_array.shape) # (4,)

# reshape
test_array = test_array.reshape(2, 2)
print(test_array.shape) # (2, 2)

# test_array = test_array.reshape(2, 3) 이거는 에러가 발생한다.

# 열 차원을 1로 두고 행 차원을 -1로 둔다는 것은 
# 나머지 차원이 다 결정되고 남은 차원을 사용하겠다는 뜻
# 특성이 열 방향으로 있어야 하기 때문에 이렇게 한다.
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)
print(train_input.shape, test_input.shape)  # (42, 1) (14, 1)
# target 데이터는 1차원 함수로 두어도 된다.

# regression에서의 score는 결정 계수 R^2
# 1에 가까울수록 좋다
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor()
knr.fit(train_input, train_target) # k-최근접 이웃 회귀 모델을 훈련
print(knr.score(test_input, test_target)) # 0.992809406101064

# 다른 지표: 평균절댓값오차
from sklearn.metrics import mean_absolute_error
test_prediction = knr.predict(test_input) # 테스트 세트에 대한 예측
mae = mean_absolute_error(test_target, test_prediction) # 테스트 세트에 대한 평균 절댓값 오차 계산
print(mae)    # 19.157142857142862 19g 정도 오차난다는 뜻


# 과대 적합 underfitting vs 과소 적합 overfitting
# 일반적으로 훈련 세트의 점수가 테스트세트 점수가 높아야 한다.
# 아래는 과소적합
print(knr.score(train_input, train_target))      # 0.9698823289099254
print(knr.score(train_input, train_target))      # 0.9928094061010639

# 이웃의 개수가 늘어나면 과소적합, 줄어들면 과대적합
knr.n_neighbors = 3  # 이웃의 개수를 3으로 설정
# 우리가 지정해야 하는 매개변수: 하이퍼파라미터
knr.fit(train_input, train_target)        # 모델을 재훈련
print(knr.score(train_input, train_target))      # 0.9804899950518966
print(knr.score(test_input, test_target))        # 0.9746459963987609
