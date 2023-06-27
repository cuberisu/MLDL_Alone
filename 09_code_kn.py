# 로지스틱 회귀 (Logistic Regression)
# 1. KNeighborsClassifier 예측 한계

# 럭키백의 확률

# 데이터 준비하기
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')
print(fish.head())  # 표로 나온다.
print(pd.unique(fish['Species'])) # 물고기 종들을 출력
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

# target의 class들 출력. abc 순
print(kn.classes_)  # ['Bream' 'Parkki' 'Perch' 'Pike' 'Roach' 'Smelt' 'Whitefish']

# 예측해보기
print(kn.predict(test_scaled[:5]))  # ['Perch' 'Smelt' 'Pike' 'Perch' 'Perch']

# 확률을 출력하기
import numpy as np
proba = kn.predict_proba(test_scaled[:5])
print(np.round(proba, decimals=4))  
# [0.     0.     0.6667 0.     0.3333 0.     0.    ] 이런 식으로 출력

distances, indexes = kn.kneighbors(test_scaled[3:4])
print(train_target[indexes])    # [['Roach' 'Perch' 'Perch']]
