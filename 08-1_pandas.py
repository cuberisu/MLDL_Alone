# 특성 공학과 규제
# 특성 공학: 여러 개의 특성을 조합해서 새로운 특성을 만든다.
# 새로운 특성을 추가하거나 특성을 변경하거나 특성끼리 조합을 하거나 ...
# feature engineering

# 다중 회귀 multiple regression (multinomial regression)
# 여러 개의 항(특성)을 쓰는 회귀


# 데이터 준비
# 판다스: 파이썬 과학 라이브러리의 핵심 패키지 중 하나
# 데이터프레임 DataFrame이라는 객체 (다차원 배열.)
# numpy와는 다르게 다른 종류의 데이터 타입을 가질 수 있다.
# 파일을 불러와서 넘파이 배열로 만드는 과정을 따라가보자.
import pandas as pd
df = pd.read_csv('https://bit.ly/perch_csv_data')   # csv 파일을 사용하였다.
""" read_csv() 함수를 사용하면 다운로드 받아서 파일을 읽고
자동으로 판다스의 데이터프레임으로 만들어준다
read csv는 자동으로 첫 번째 행이 각열의 제목
두 번째부터 실제 데이터로 처리해준다. """
# 이렇게 pandas 데이터프레임을 만든 다음 numpy 배열로 바꿀 때는 
# to_numpy() 메소드를 호출해주면 된다.
# 이렇게 호출해주면 perch_full 객체가 만들어짐
# 이 객체는 컬럼 스택으로 만들었던 numpy 배열과 같음
perch_full = df.to_numpy()
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

# numpy 배열과 target 데이터를 가지고 train set과 test set으로 나누는 과정
# train_test_split()을 이용
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    perch_full, perch_weight, random_state=42)


# 다항 특성만들기: 사이킷런에서 자동으로 해준다.

# degree
# PolynomialFeatures()의 기본 매개변수 degree=2 (제곱항을 만들어줌)
# degree=3 이면 3제곱 항도 만들어줌

# 사이킷런의 "변환기(Transformer)" PolynomialFeatures 클래스
# 특성을 변환하주거나 전처리하거나 바꿔주는 클래스들
# fit, transform 메서드가 있다. predict나 score 메서드는 존재하지 않음.
# 이걸 합친 fit-transform() 메서드도 제공함
# 그 외에 모델링을 하는 것들은...
# LinearRegression, KNeighborsClassifier 는 "추정기(Estimator)" 라고 한다. 
# 추정기들은 fit, predict, score 메서드들이 있음
# PolynomialFeatures 클래스는 실제로 뭔가 학습하는 건 아님
# 특성이 몇 개 있고, 어떤 조합을 만들어줄지 파악하는 정도가 fit 메서드
from sklearn.preprocessing import PolynomialFeatures

# 예시
# poly = PolynomialFeatures()   
# poly.fit([[2, 3]])
# print(poly.transform([[2, 3]])) # [[1. 2. 3. 4. 6. 9.]]

# 설명
# 1(bias), 2(원래 항), 3(원래 항), 4(2**2), 6(2*3), 9(3**2)
# 1?: 절편을 위한 특성.
# y = ax + b에서 [a, b] * [x, 1] 을 할 수 있다.
# LinearRegression 클래스는 1 특성을 무시한다.

# 1을 빼는 예시: include_bias 매개변수
# poly = PolynomialFeatures(include_bias=False) # include_bias=False는 1을 빼줌 기본값은 True
# poly.fit([[2, 3]])
# print(poly.transform([[2, 3]])) # [[2. 3. 4. 6. 9.]]

poly = PolynomialFeatures(include_bias=False)
poly.fit(train_input)
train_poly = poly.transform(train_input)

print(train_poly.shape)     # (42, 9) 42개의 샘플 9개의 특성

print(poly.get_feature_names_out())    # 각각의 특성이 어떻게 만들어졌는지 확인하려면 이 메서드를 사용
# array(['x0', 'x1', 'x2', 'x0^2', 'x0 x1', 'x0 x2', 'x1^2', 'x1 x2',
#      'x2^2'], dtype=object)   # x0, x1, x2는 원래 특성들을 나타냄.

test_poly = poly.transform(test_input)  # test도 train에서 훈련(?)한 것을 사용한다.


# 다중 회귀 모델 훈련하기
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(train_poly, train_target)

# 특성이 많아져서 복잡한 모델이 되었고 점수가 높아졌다.
print(lr.score(train_poly, train_target))   # 0.9903183436982126
print(lr.score(test_poly, test_target)) # 0.9714559911594125 


# 더 많은 특성 만들기
poly = PolynomialFeatures(degree=5, include_bias=False)

poly.fit(train_input)
train_poly = poly.transform(train_input)
test_poly = poly.transform(test_input)

print(train_poly.shape) # (42, 55) 특성의 개수가 무려 55개!

lr.fit(train_poly, train_target)

print(lr.score(train_poly, train_target))   # 0.9999999999997232
print(lr.score(test_poly, test_target)) # -144.40564483377855   너무 과대적합.... -> 규제로 해결


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
# alpha값이 커지면 강도가 세지고, 작아지면 약해진다.
# 여러 값을 넣어보면서 어떤 값이 최적값인지 살펴본다.
# alpha값은 훈련값이 아닌 우리가 정해주는 값이다. -> 하이퍼파라미터 라고 부른다.
# 학습하는 매개변수는 모델 파라미터라고 부른다.
alpha_list = [0.001, 0.01, 0.1, 1, 10, 100] # 10의 배수로 하는 것이 관례이다.
for alpha in alpha_list:
    # 릿지 모델을 만듭니다
    ridge = Ridge(alpha=alpha)
    # 릿지 모델을 훈련합니다
    ridge.fit(train_scaled, train_target)
    # train score과 test score을 저장합니다
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
    # 라쏘 모델을 만듭니다
    lasso = Lasso(alpha=alpha, max_iter=10000)
    # 라쏘 모델을 훈련합니다
    lasso.fit(train_scaled, train_target)
    # 훈련 점수와 테스트 점수를 저장합니다
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

print(np.sum(lasso.coef_ == 0)) # 40

# 라쏘보단 릿지를 선호. 이게 좀더 효과적임
# 라쏘는 일부 특성을 완전히 사용 안 할 수 있음
# 가중치를 0으로 만들어버릴 수 있음 -> 특성을 사용 안 하게 됨
# 40개가 0으로 되어있음
# 55개중 40개가 0이라는 뜻.
