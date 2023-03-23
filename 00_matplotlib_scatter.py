# matplotlib은 그래프 관련 표준 라이브러리
import matplotlib.pyplot as plt

# plt.xlabel('x축 이름')
# plt.ylabel('y축 이름')
# plt.show()    # 그래프 보이기

# plot(x, y) 함수로 그래프를 그려보자 

plt.plot([1, 2, 3, 4, 5], [1, 4, 9, 16, 25])    # x축, y축의 값을 파이썬 리스트로 전달
plt.show()

# scatter(x, y, marker='') 함수로 산점도를 그려보자
plt.scatter([1, 2, 3, 4, 5], [1, 4, 9, 16, 25]) # x축, y축의 값을 파이썬 리스트로 전달
plt.show()



# numpy 배열 형태의 데이터 준비
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

# 데이터 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)
# 훈련 세트와 테스트 세트를 2차원 배열로 바꿉니다
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

# k-최근접 이웃 회귀 모델에서 이웃 표시하기
from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor() # 괄호에 n_neighbors=3 하면 이웃 기본값 5에서 지정값 3이 된다.
knr.fit(train_input, train_target) # k-최근접 이웃 회귀 모델을 훈련
# data와 target을 각각 나누는 게 아니라 한꺼번에 전달해서 train과 test로 나눌 수 있음
# 분류 문제를 다루는 경우 stratify를 사용하는 것이 좋다.
# random_state 랜덤시드 설정. 결과를 재현하기 위해 씀. 실전에선 중요하지 않다.
# train_test_split은 배열을 받아서 각각 2개로 나누어준다. 3개를 전달하면 6개가 나온다.

# stratify=fish_target
# 샘플링 편향이 일어나지 않도록 target class가 골고루 섞여서 train과 test로 나눠야 함
# train data가 아주 크거나 class가 비교적 균등하게 있으면(?) 그냥 적당히 랜덤으로 섞어도
# 꽤 괜찮게 나눌 수 있지만 이 경우는 그렇지 않기 때문에 
# stratify 매개변수에 target 배열을 전달하여 target 값이 골고루 섞이도록
# train과 test를 나눠준다.

# 가장 가까운 이웃(기본값 5개)을 찾아주는 kneighbors()
# 이웃까지의 거리와 이웃 샘플의 인덱스를 반환
distances, indexes = knr.kneighbors([[50]]) # 50cm 농어의 이웃 구하기

# 훈련 세트의 산점도 그리기
plt.scatter(train_input, train_target)
# 훈련 세트 중에서 이웃 샘플만 다시 그립니다
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
# 50cm 농어 데이터
plt.scatter(50, 1033, marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()