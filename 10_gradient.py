# 확률적 경사 하강법 (Stochastic Gradient Descent)
# SGDClassifier

# 데이터 준비 (전처리, 각 특성마다 스케일을 같게 해야 함)
import pandas as pd
fish = pd.read_csv('https://bit.ly/fish_csv_data')

fish_input = fish[['Weight', 'Length', 'Diagonal', 'Height', 'Width']].to_numpy()
fish_target = fish['Species'].to_numpy()

# 데이터 나누기
from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    fish_input, fish_target, random_state=42)

# 표준화 (표준점수)
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
ss.fit(train_input)
train_scaled = ss.transform(train_input)
test_scaled = ss.transform(test_input)


# SGDClassifier
from sklearn.linear_model import SGDClassifier
sc = SGDClassifier(loss='log_loss', max_iter=10, random_state=42)
# loss='log_loss': 로지스틱 손실 함수를 나타냄
# max_iter: 에포크. 훈련 세트를 10번 쓴다는 뜻
sc.fit(train_scaled, train_target)  
# print(sc.score(train_scaled, train_target)) # 0.773109243697479
# print(sc.score(test_scaled, test_target))   # 0.775

sc.partial_fit(train_scaled, train_target)
# 기존에 학습한 w와 b를 그대로 유지하면서 다시 또 훈련하는 것
print(sc.score(train_scaled, train_target)) # 0.8151260504201681
print(sc.score(test_scaled, test_target))   # 0.85


# 에포크와 과대/과소적합
import numpy as np
sc = SGDClassifier(loss='log_loss', random_state=42)
train_score = []
test_score = []

classes = np.unique(train_target)

for _ in range(0, 300):
    sc.partial_fit(train_scaled, train_target, classes=classes)
    
    train_score.append(sc.score(train_scaled, train_target))
    test_score.append(sc.score(test_scaled, test_target))
    

import matplotlib.pyplot as plt
plt.plot(train_score)
plt.plot(test_score)
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.show()


sc = SGDClassifier(loss='log_loss', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) # 0.957983193277311
print(sc.score(test_scaled, test_target))   # 0.925

sc = SGDClassifier(loss='hinge', max_iter=100, tol=None, random_state=42)
sc.fit(train_scaled, train_target)
print(sc.score(train_scaled, train_target)) # 0.9495798319327731
print(sc.score(test_scaled, test_target))   # 0.925
