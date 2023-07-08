# 교차 검증과 그리드 서치

import pandas as pd
wine = pd.read_csv('https://bit.ly/wine_csv_data')

data = wine[['alcohol', 'sugar', 'pH']].to_numpy()
target = wine['class'].to_numpy()

from sklearn.model_selection import train_test_split
train_input, test_input, train_target, test_target = train_test_split(
    data, target, test_size=0.2, random_state=42)

sub_input, val_input, sub_target, val_target = train_test_split(
    train_input, train_target, test_size=0.2, random_state=42)

print(sub_input.shape, val_input.shape)

from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)
dt.fit(sub_input, sub_target)

print(dt.score(sub_input, sub_target))
print(dt.score(val_input, val_target))


# 교차 검증
from sklearn.model_selection import cross_validate
scores = cross_validate(dt, train_input, train_target)
print(scores)   # 기본값이 5 폴드
'''
{'fit_time': array([0.00797129, 0.00850534, 0.0069778 , 0.00697279, 0.00498533]), 
'score_time': array([0.00099778, 0.00098944, 0.00200343, 0.00099707, 0.00099683]), 
'test_score': array([0.86923077, 0.84615385, 0.87680462, 0.84889317, 0.83541867])}
'''

# 교차검증 5개 점수 평균내기
import numpy as np
print(np.mean(scores['test_score']))    # 0.855300214703487


# 분할기(splitter)를 이용한 교차검증
# StratifiedKFold(): 분류 모델에서 클래스들이 고르게 섞이도록 하는 객체
# 회귀는 KFold(). 사이킷런이 자동으로 인식해서 지정해줌
from sklearn.model_selection import StratifiedKFold
scores = cross_validate(dt, train_input, train_target, cv=StratifiedKFold())

# 10개의 폴드, 랜덤셔플
splitter = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_validate(dt, train_input, train_target, cv=splitter)
print(np.mean.scores['test_score'])


# 그리드 서치
# 하이퍼파라미터 튜닝
from sklearn.model_selection import GridSearchCV
params = {'min_impurity_decrease': [0.0001, 0.0002, 0.0003, 0.0004, 0.0005]}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)

gs.fit(train_input, train_target)

dt = gs.best_estimator_
print(dt.score(train_input, train_target))

print(gs.best_params_)

print(gs.cv_results_['mean_test_score'])

best_index = np.argmax(gs.cv_results_['mean_test_score'])
print(gs.cv_results['params'][best_index])

params = {'min_impurity_decrease': np.arange(0.0001, 0.001, 0.0001),
          'max_depth': range(5, 20, 1), 
          'min_samples_split': range(2, 100, 10)
}

gs = GridSearchCV(DecisionTreeClassifier(random_state=42), params, n_jobs=-1)
gs.fit(train_input, train_target)

print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))


# 확률 분포 선택
# 랜덤 서치
from scipy.stats import uniform, randint
rgen = randint(0, 10)
rgen.rvs(10)

np.unique(rgen.rvs(1000), return_counts=True)

ugen = uniform(0, 1)
ugen.rvs(10)

params = {'min_impurity_decrease': uniform(0.0001, 0.001), 
          'max_depth': randint(20, 50),
          'min_sample_split': randint(2, 25), 
          'min_sample_leaf': randint(1, 25)
          }

from sklearn.model_selection import RandomizedSearchCV
gs = RandomizedSearchCV(DecisionTreeClassifier(random_state=42), params, 
                        n_iter=100, n_jobs=-1, random_state=42)

gs.fit(train_input, train_target)

print(gs.best_params_)

print(np.max(gs.cv_results_['mean_test_score']))

dt = gs.best_estimator_
print(dt.score(test_input, test_target))