# 특성 공학(feature engineering)과 규제(regularization)
# 특성 공학: 여러 개의 특성을 조합해서 새로운 특성을 만든다.
# 규제: 너무 과대적합된 모델의 가중치(기울기)를 작게 만드는 방법. 릿지 회귀 / 라쏘 회귀


# 농어 데이터 준비
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')   # csv 파일을 다운받아 pandas df를 만듦
perch_full = df.to_numpy()  # numpy 배열로 바꿈
# print(perch_full)     # 특성은 3개, 샘플은 42개가 있는 numpy 배열이 출력

# target data (농어 무게 데이터)
import numpy as np
perch_weight = np.array([5.9, 32.0, 40.0, 51.5, 70.0, 100.0, 78.0, 80.0, 85.0, 85.0, 
     110.0, 115.0, 125.0, 130.0, 120.0, 120.0, 130.0, 135.0, 110.0, 
     130.0, 150.0, 145.0, 150.0, 170.0, 225.0, 145.0, 188.0, 180.0, 
     197.0, 218.0, 300.0, 260.0, 265.0, 250.0, 250.0, 300.0, 320.0, 
     514.0, 556.0, 840.0, 685.0, 700.0, 700.0, 690.0, 900.0, 650.0, 
     820.0, 850.0, 900.0, 1015.0, 820.0, 1100.0, 1000.0, 1100.0, 
     1000.0, 1000.0])

# 두 배열 데이터를 train set과 test set으로 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)


# 사이킷런의 변환기(Transformer) PolynomialFeatures 클래스
# 특성을 변환하주거나 전처리하거나 바꿔주는 클래스들
# fit, transform 메서드가 있다. predict나 score 메서드는 존재하지 않음.
# 이 클래스에서는 특성이 몇 개 있고, 어떤 조합을 만들어줄지 파악하는 정도가 fit 메서드
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(include_bias=False)   # include_bias=False는 1을 빼줌
                                                # degree=2(제곱수), include_bias=True(1 포함)가 기본값
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape)     # (42, 9) 42개의 샘플 9개의 특성
print(poly.get_feature_names_out())    # 각각의 특성이 어떻게 만들어졌는지 확인하려면 이 메서드를 사용
# ['x0' 'x1' 'x2' 'x0^2' 'x0 x1' 'x0 x2' 'x1^2' 'x1 x2' 'x2^2']

# 다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression() 
lr.fit(train_poly, train_target)

# 특성이 많아져서 복잡한 모델이 되었고 점수가 높아졌다.
print(lr.score(train_poly, train_target))   # 0.9903183436982126
print(lr.score(test_poly, test_target)) # 0.9714559911594125 


# 더 많은 특성 만들기
poly = PolynomialFeatures(degree=5, include_bias=False) # 5제곱, 1 제외
poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)
# print(train_poly.shape)   # (42, 55) 특성 개수가 55개라는 뜻

lr.fit(train_poly, train_target)
print(lr.score(train_poly, train_target))   # 0.9999999999996433
print(lr.score(test_poly, test_target))     # -144.40579436844948   너무 과대적합 -> 규제로 해결


# 규제 (regularization) (정규화라고도 함)
# 가중치(기울기)를 작게 만드는 방법. 릿지 회귀 / 라쏘 회귀
# 가중치가 높을수록 벌칙을 내림

# 규제 전에 표준화 (z 점수 혹은 표준 점수)
# 규제를 하기 위해서는 특성의 스케일을 조정해주어야 함
# 이전에는 mean, std로 수동으로 만들었는데 이번에는 자동으로 만들어보자.
from sklearn.preprocessing import StandardScaler    # 변환기
ss = StandardScaler()
ss.fit(train_poly)

train_scaled = ss.transform(train_poly)
test_scaled = ss.transform(test_poly)

# 릿지 회귀
# (가중치)^2을 벌칙으로 내림 (L2 규제라고도 부름)
from sklearn.linear_model import Ridge
ridge = Ridge()
ridge.fit(train_scaled, train_target)
print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 적절한 규제 강도 찾기
import matplotlib.pyplot as plt
train_score = []
test_score = []

# Ridge에는 alpha=1 매개변수가 있다.
# 여러 값을 넣어보면서 어떤 값이 최적값인지 살펴본다.

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100] # 10의 배수로 하는 것이 관례이다.
for alpha in alpha_list:
    ridge = Ridge(alpha=alpha)
    ridge.fit(train_scaled, train_target)
    # score 저장
    train_score.append(ridge.score(train_scaled, train_target))
    test_score.append(ridge.score(test_scaled, test_target))

# 그대로 그래프를 그리면 소수점부분이 잘 안보이기 때문에 로그를 취해준다.
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()

ridge = Ridge(alpha=0.1)    # 최적값
ridge.fit(train_scaled, train_target)

print(ridge.score(train_scaled, train_target))
print(ridge.score(test_scaled, test_target))

# 라쏘 회귀
# 가중치의 절댓값에 벌칙을 준다.
# L1 규제
from sklearn.linear_model import Lasso
lasso = Lasso()
lasso.fit(train_scaled, train_target)
print(lasso.score(train_scaled, train_target))  # 0.989789897208096
print(lasso.score(test_scaled, test_target))    # 0.9800593698421883

train_score = []
test_score = []

alpha_list = [0.001, 0.01, 0.1, 1, 10, 100]
for alpha in alpha_list:
    lasso = Lasso(alpha=alpha, max_iter=10000)
    lasso.fit(train_scaled, train_target)
    # score 저장
    train_score.append(lasso.score(train_scaled, train_target))
    test_score.append(lasso.score(test_scaled, test_target))
    
plt.plot(np.log10(alpha_list), train_score)
plt.plot(np.log10(alpha_list), test_score)
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.show()    

lasso = Lasso(alpha=10)
lasso.fit(train_scaled, train_target)

print(lasso.score(train_scaled, train_target))  # 0.9888067471131867
print(lasso.score(test_scaled, test_target))    # 0.9824470598706695

print(np.sum(lasso.coef_ == 0)) # 40. 55개중 40개가 0이라는 뜻