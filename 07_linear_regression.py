# 농어의 길이로 무게를 예측하는 알고리즘
# 선형 회귀는 k-최근접 이웃 회귀 알고리즘의 한계를 보완할 수 있다.

# 데이터를 numpy 배열로 준비
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

from sklearn.model_selection import train_test_split

# train set과 test set로 나누기
train_input, test_input, train_target, test_target = train_test_split(
    perch_length, perch_weight, random_state=42)

# 2차원 배열로 바꾸기
train_input = train_input.reshape(-1, 1)
test_input = test_input.reshape(-1, 1)

from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=3)   # 이웃의 수를 3으로
knr.fit(train_input, train_target)  # k-최근접 이웃 회귀 모델을 훈련

print(knr.predict([[50]]))  # [1000.] 예상보다 작은 값이다!!!!

# 산점도로 이유 알아보기
import matplotlib.pyplot as plt

# 50 cm 농어의 이웃 표시하기
distances, indexes = knr.kneighbors([[50]])
plt.scatter(train_input, train_target)  # train set의 산점도 그리기
# train set 중 이웃 샘플만 다이아몬드 모양으로 다시 그리기
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(50, 1033, marker='^')   # 50 cm 농어 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(np.mean(train_target[indexes]))   # 1033.3333333333333

# 100 cm 농어의 이웃 표시하기
distances, indexes = knr.kneighbors([[100]])
plt.scatter(train_input, train_target)  # train set의 산점도 그리기
# train set 중 이웃 샘플만 다이아몬드 모양으로 다시 그리기
plt.scatter(train_input[indexes], train_target[indexes], marker='D')
plt.scatter(100, 1033, marker='^')  # 100 cm 농어 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()


# 선형 회귀
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_input, train_target)   # 선형 회귀 모델 훈련
print(lr.predict([[50]]))   # 50 cm 농어에 대한 예측 [1241.83860323]

# 선형 회귀 y = ax + b 에서
# coef_는 기울기 a. 배열이다. 
# 여기서는 길이 특성 하나만 사용했기 때문에 원소가 딱 하나이다.
# intercept_는 절편 b. 절편은 하나이다.
# 끝에 _가 붙어 있는 속성은?
# 사이킷런 모델들은 지정한 값이 아닌 "학습된" 값들을 저장할 때 다른 속성과 구분되라고 _문자를 추가한다.
print(lr.coef_, lr.intercept_)  # [39.01714496] -709.0186449535474

# train set의 산점도 그리기
plt.scatter(train_input, train_target)
# 15에서 50까지 1차 방정식 그래프 그리기
plt.plot([15, 50], [15*lr.coef_+lr.intercept_, 50*lr.coef_+lr.intercept_])
plt.scatter(50, 1241.8, marker='^') # 50 cm 농어 데이터
plt.show()    
# 선형회귀 단점: 음수값이 나올 수 있다.

# 과소적합
print(lr.score(train_input, train_target))  # 0.9398463339976041
print(lr.score(test_input, test_target))    # 0.824750312331356


# 다항 회귀 polynomial regression
# 이차 함수 형태로 그래프를 그려보자.

# 일차원 배열 여러 개를 열 방향으로 나란히 붙이는 건 column_stack() 함수
train_poly = np.column_stack((train_input ** 2, train_input))
test_poly = np.column_stack((test_input ** 2, test_input))
print(train_poly.shape, test_poly.shape)    # (42, 2) (14, 2)

# 다항 회귀도 LinearRegression() 클래스에 집어넣으면 된다.
lr = LinearRegression()
lr.fit(train_poly, train_target)
print(lr.predict([[50**2, 50]]))    # [1573.98423528]
print(lr.coef_, lr.intercept_)  # [  1.01433211 -21.55792498] 116.05021078278264

# 구간별 직선을 그리기 위해 15에서 49까지 정수 배열을 만듭니다.
# 직선을 잘게 쪼개서 이어붙여 곡선 그래프로 그린다.: point 배열
point = np.arange(15, 50)
plt.scatter(train_input, train_target)  # train set의 산점도 그리기
# 15에서 49까지 2차 방정식 그래프 그리기
plt.plot(point, 1.01*point**2 - 21.6*point + 116.05)
plt.scatter([50], [1574], marker='^') # 50 cm 농어 데이터
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

print(lr.score(train_poly, train_target))   # 0.9706807451768623
print(lr.score(test_poly, test_target)) # 0.9775935108325122