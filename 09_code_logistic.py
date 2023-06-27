# 로지스틱 회귀 (Logistic Regression)
# 2. 로지스틱 회귀

# 럭키백의 확률

# 데이터 준비
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(pd.unique(fish['Species']))   
# ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

# input, target 설정
# list에서 특성1, 특성2만 뽑아서 numpy 배열로 바꿔 넣기: list['특성1', '특성2'].to_numpy()
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 데이터 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 표준화 (전처리)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# 로지스틱 회귀: 이진분류
# 불리언 인덱싱으로 도미와 빙어만 추출
bream_smelt_indexes = (train_target == 'Bream') | (train_target == 'Smelt')
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

# 로지스틱 회귀 돌리기
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

# 예측하기
print(lr.predict(train_bream_smelt[:5]))    # z값 (-무한대 ~ +무한대. 여기선 사이킷런이 문자열로 바꿔줌)
print(lr.predict_proba(train_bream_smelt[:5]))  # phi값 (0~1) [[음성(0), 양성(1)]] 순으로 출력
# 알파벳 순서로 인덱스 번호가 매겨져서 Bream이 0이 되고 Smelt가 1이 되었다. 이전과 반대가 되었다.

# 가중치, 절편
print(lr.coef_, lr.intercept_)
# [[-0.4037798  -0.57620209 -0.66280298 -1.01290277 -0.73168947]] [-2.16155132]

# 또다른 방법으로 예측하기: decision_function() (사이킷런의 분류 모델에서 또 제공하는 방법. z값을 숫자로 출력)
decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)    # [-6.02927744  3.57123907 -5.26568906 -4.24321775 -6.0607117 ]

# 이 z값을 시그모이드 함수에 넣기
from scipy.special import expit  # 계산을 자동으로 해주는 클래스
print(expit(decisions))  # [0.00240145 0.97264817 0.00513928 0.01415798 0.00232731] 확률값

# 로지스틱 회귀가 이진분류일 경우 양성클래스의 z값만 계산한다. (줄56)
# 근데 확률을 출력할 땐 음성클래스의 확률도 출력할 수 있다. (줄52)


# 로지스틱 회귀: 다중분류

# 로지스틱 회귀는 L2 norm을 사용하는 규제를 기본적으로 적용. 규제가 기본적으로 적용되어 있는 것.
# 규제의 강도를 정하는 것이 필요. 선형회귀에선 alpha, 
# 사이킷런의 분류 모델에서는 많은 경우에 C 매개변수. default=1
# 올라가면 규제가 약해진다... 내려가면 강해진다. 규제의 역수라고 표현하기도 함.

lr = LogisticRegression(C=20, max_iter=1000)    
# max_iter: 반복횟수를 조절. default=100 근데 그러면 반복횟수가 모자라다고 경고 메시지가 뜬다.
lr.fit(train_scaled, train_target)
print(lr.score(train_scaled, train_target)) # 0.9327731092436975
print(lr.score(test_scaled, test_target))   # 0.925

# 확률 예측하기
proba = lr.predict_proba(test_scaled[:5])
import numpy as np
print(np.round(proba, decimals=3))
# [0.    0.014 0.841 0.    0.136 0.007 0.003] 이런 식으로 출력

# 계수(가중치)와 절편의 shape (값은 너무 많으니까...)
print(lr.coef_.shape, lr.intercept_.shape)  # (7, 5) (7,)


# softmax 함수 (시그모이드 함수를 대신하는, 확률을 다 더했을 때 1이 나오게 하는 함수)
decision = lr.decision_function(test_scaled[:5])
print(np.round(decision, decimals=2))
# [ -6.5    1.03   5.16  -2.73   3.34   0.33  -0.63] 이상한 출력
# 이 z값을 softmax에 넣으면 S0(시그모이드1) = e^z0/sum, S1 = e^z1/sum, ... + S6 = e^6/sum

from scipy.special import softmax
proba = softmax(decision, axis=1)
print(np.round(proba, decimals=3))  # round 함수의 decimals를 3으로 하면 소수점 3째자리까지 출력함
# 줄 78과 동일한 출력

# 시그모이드 함수: 이진분류일 경우 확률을 표현하기 위한 수학적 트리
# softmax(): 다중분류일 경우 확률처럼 표현하기 위한 수학적 트리