# 넘파이로 데이터 준비
import numpy as np

# 도미 데이터 35개
bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 
                31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 
                34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 
                38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 
                450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 
                700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 
                700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 
                925.0, 975.0, 950.0]
# 빙어 데이터 14개
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 
                12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 
                12.2, 19.7, 19.9]

# 리스트 합치기. 연산자 오버로딩?
fish_length = bream_length + smelt_length
fish_weight = bream_weight + smelt_weight

# 샘플을 행에 두고 특성을 열에 두는 2차원 배열을 만들기
# for 대신 numpy의 column_stack()을 이용
# 주어진 두 배열을 나란히 세운 다음 열로 붙여줌
fish_data = np.column_stack((fish_length, fish_weight))

# 그냥 입력되는 두 배열을 길게 붙여주는 역할 concatenate()
# 연산자 오버로딩 [1]*35 대신 ones(35) [0]*14 대신 zeros(14)
fish_target = np.concatenate((np.ones(35), np.zeros(14)))
# np.ones((2,3))도 가능. np.full((2, 3), 9): 9로 채워진 2by3 배열
print(fish_target)

# 사이킷런으로 데이터 섞고 나누기 train_test_split()
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_data, fish_target, stratify=fish_target, random_state=42)
# data와 target을 각각 나누는 게 아니라 한꺼번에 전달해서 train과 test로 나눌 수 있음
# stratify는 샘플링 편향이 일어나지 않도록 target class가 골고루 섞여서 train과 test로 나눠야 함
# train data가 아주 크거나 class가 비교적 균등하게 있으면(?) 그냥 적당히 랜덤으로 섞어도
# 꽤 괜찮게 나눌 수 있지만 이 경우는 그렇지 않기 때문에 
# stratify 매개변수에 target 배열을 전달하여 target 값이 골고루 섞이도록
# train과 test를 나눠준다.
# 분류 문제를 다루는 경우 stratify를 사용하는 것이 좋다.
# random_state 랜덤시드 설정. 결과를 재현하기 위해 씀. 실전에선 중요하지 않다.
# train_test_split은 배열을 받아서 각각 2개로 나누어준다. 3개를 전달하면 6개가 나온다.

from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
kn = KNeighborsClassifier()
kn.fit(train_input, train_target)
kn.score(test_input, test_target)   # 1.0
print(kn.predict([[25, 150]]))  # [0. ]

# 가장 가까운 이웃(기본값 5개)을 찾아주는 kneighbors()
distance, indexes = kn.kneighbors([[25, 150]])  # 이웃까지의 거리와 이웃 샘플의 인덱스를 반환

plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')    # 삼각형 마커
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')   # 마름모 마커
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# scale 맞추기
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(25, 150, marker='^')
plt.scatter(train_input[indexes, 0], train_input[indexes, 1], marker='D')
plt.xlim((0, 1000)) # x축의 scale을 0~1000으로 맞춰서 y축과 같게 함
plt.xlabel('length')
plt.ylabel('weight')
plt.show()  # y축에 의존한다는 것을 알 수 있다.

# 표준점수(z점수)로 바꾸기: (특성-평균)/표준편차
mean = np.mean(train_input, axis=0)
std = np.std(train_input, axis=0)
train_scaled = (train_input - mean) / std   # numpy 브로드캐스팅

# 수상한 도미 다시 표시하기
new = ([25, 150] - mean) / std  # 훈련데이터와 똑같은 방식으로 변환해야 쓸 수 있다.
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 전처리 데이터에서 모델 훈련
# 지금까지 한 내용이 전처리이다. 표준점수로 바꾸고, scale을 맞추는 등.
kn.fit(train_scaled, train_target)
test_scaled = (test_input - mean) / std
kn.score(test_scaled, test_target)
print(kn.predict([new]))
distances, indexes = kn.kneighbors([new])
plt.scatter(train_scaled[:, 0], train_scaled[:, 1])
plt.scatter(new[0], new[1], marker='^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()