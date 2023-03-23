# 로지스틱 회귀 (Logistic Regression)
# 이름은 회귀지만 분류 모델이다.

# 럭키백의 확률  
# 도미일 확률이 몇%일까? (분류하면서 얼마나 확신하는지를 확률로 표현)


# 데이터 준비하기
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
# print(fish.head())  # 표로 나온다.
# print(pd.unique(fish['Species'])) # 물고기 종들을 출력
# ['Bream' 'Roach' 'Whitefish' 'Parkki' 'Perch' 'Pike' 'Smelt']

# fish_input (특성 데이터는 5가지 무게, 길이, 대각선, 높이, 두께)
fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
# print(fish_input[:5])
'''
[[242.      25.4     30.      11.52     4.02  ]
 [290.      26.3     31.2     12.48     4.3056]
 [340.      26.5     31.1     12.3778   4.6961]
 [363.      29.      33.5     12.73     4.4555]
 [430.      29.      34.      12.444    5.134 ]]
'''

# fish_target
fish_target = fish['Species'].to_numpy()

# 데이터 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 표준화 (z 점수 혹은 표준 점수) (scale 맞추기. 분류 모델에서는 scale이 중요하다.)
# 이전에는 mean, std로 수동으로 만들었는데 이번에는 자동으로 만들어보자.
from sklearn.preprocessing import StandardScaler    # 변환기
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# k-최근접 이웃 분류기의 확률 예측
from sklearn.neighbors import KNeighborsClassifier
kn = KNeighborsClassifier(n_neighbors=3)    # default n_neighbors=5
kn.fit(train_scaled, train_target)
print(kn.score(train_scaled, train_target)) # 0.8907563025210085
print(kn.score(test_scaled, test_target))   # 0.85

# species 출력
# print(kn.classes_)
# ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

# 예측 출력
# print(kn.predict(test_scaled[:5]))
# ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']

# predict_proba로 확률 출력
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
# print(np.round(proba, decimals=4))
'''
[[0.     0.     1.     0.     0.     0.     0.    ]
 [0.     0.     0.     0.     0.     1.     0.    ]
 [0.     0.     0.     1.     0.     0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
[[0.     0.     1.     0.     0.     0.     0.    ]
 [0.     0.     0.     0.     0.     1.     0.    ]
 [0.     0.     0.     1.     0.     0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]
 [0.     0.     0.6667 0.     0.3333 0.     0.    ]]
'''

distances, indexes = kn.kneighbors(test_scaled[3:4])
# print(train_target[indexes])    # [['Roach' 'Perch' 'Perch']]


# 로지스틱 회귀
# 대표적인 분류 모델
# 선형회귀와 비슷
# z = a*무게 + b*길이 + c*대각선 + d*높이 + e*두께 + f
# 이걸 그대로 사용하면 회귀모델이 됨
# -무한대 ~ +무한대 사이 값을 0~1 확률값으로 바꾸는 과정이 필요함
# 시그모이드 함수 혹은 로지스틱 함수 라고 함
# 시그모이드 함수의 0.5를 기준으로 위쪽이면 양성클래스, 아래쪽이면 음성클래스

import numpy as np
import matplotlib.pyplot as plt

z = np.arange(-5, 5, 0.1)
phi = 1 / (1 + np.exp(-z))

plt.plot(z, phi)
plt.xlabel('z')
plt.ylabel('phi')
plt.show()


# 로지스틱 회귀로 이진분류 수행하기
# 불리언 인덱싱: True 원소만 뽑아내는 방식

# 예시
char_arr = np.array(['A', 'B', 'C', 'D', 'E'])
print(char_arr[[True, False, True, False, False]])  # ['A', 'C']

# 실전: 도미 | 빙어인 것을 True로 둔 것. | 는 비교 연산자 'or'
bream_smelt_indexes = (train_target == 'Bream') | (train_target =='Smelt')  
train_bream_smelt = train_scaled[bream_smelt_indexes]
target_bream_smelt = train_target[bream_smelt_indexes]

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(train_bream_smelt, target_bream_smelt)

print(lr.predict(train_bream_smelt[:5]))
# ['Bream' 'Smelt' 'Bream' 'Bream' 'Bream']

print(lr.predict_proba(train_bream_smelt[:5]))
'''
[[0.99759855 0.00240145]
 [0.02735183 0.97264817]
 [0.99486072 0.00513928]
 [0.98584202 0.01415798]
 [0.99767269 0.00232731]]
'''

print(lr.classes_)

print(lr.coef_, lr.intercept_)

decisions = lr.decision_function(train_bream_smelt[:5])
print(decisions)

from scipy.special import expit
print(expit(decisions))