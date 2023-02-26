from sklearn.neighbors import KNeighborsClassifier
# train 데이터와 test 데이터는 달라야 한다.
# 저번 것은 train을 test로 써버려서 score 1.0이 나온 것.
# train 데이터와 test 데이터는 도미와 빙어가 어느 한쪽이 더 많으면 안 좋다.

# 데이터 준비
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
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 리스트 내포 # 사이킷런이 기대하는 2차원으로 만들어줘야 함
fish_data = [[l, w] for l, w in zip(length, weight)]    # 2차원 배열

# 정답 준비
fish_target = [1]*35 + [0]*14

# 훈련 세트와 테스트 세트
train_input = fish_data[:35]    # 0부터 34까지 총 35개
train_target = fish_target[:35] # 0부터 34까지 총 35개

test_input = fish_data[35:]     # 35부터 끝까지 총 14개
test_target = fish_target[35:]  # 35부터 끝까지 총 14개

kn = KNeighborsClassifier()
kn.fit(train_input, train_target)

print(kn.score(test_input, test_target))    # 0.0 나옴
# 훈련세트엔 도미 데이터만 있고 테스트 세트에는 빙어 데이터만 있기 때문 (샘플링 편향)

# numpy 사용하기
import numpy as np
input_arr = np.array(fish_data)
target_arr = np.array(fish_target)
print(input_arr)

# 데이터 섞기
index = np.arange(49)
np.random.shuffle(index)

train_input = input_arr[index[:35]]
train_target = target_arr[index[:35]]

test_input = input_arr[index[35:]]
test_target = input_arr[index[35:]]

# 확인하기
import matplotlib.pyplot as plt
plt.scatter(train_input[:, 0], train_input[:, 1])
plt.scatter(test_input[:, 0], test_input[:, 1])
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# 훈련하기
kn.fit(train_input, train_target)
print(kn.score(test_input, test_target))    # 1.0 나옴