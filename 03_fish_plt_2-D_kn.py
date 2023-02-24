# 3강. 마켓과 머신러닝
# 도미와 빙어 데이터를 학습하자.

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

# 산점도
import matplotlib.pyplot as plt # 그래프 그려주는 matplotlib을 plt 별칭
plt.scatter(bream_length, bream_weight) # 산점도(x축 데이터, y축 데이터)
plt.xlabel('length')    # x축 이름
plt.ylabel('weight')    # y축 이름
# plt.show()  # 그래프를 보여줌. 하나만 보여줄 수도 있고 여러 개를 보여줄 수도 있다.

# 빙어 데이터 14개
smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 
                12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 
                12.2, 19.7, 19.9]

# 산점도
plt.scatter(smelt_length, smelt_weight) # 산점도(x축 데이터, y축 데이터)
plt.xlabel('length')    # x축 이름
plt.ylabel('weight')    # y축 이름
plt.show()  # 그래프를 보여줌. 하나만 보여줄 수도 있고 여러 개를 보여줄 수도 있다.

# 리스트 합치기. 연산자 오버로딩?
length = bream_length + smelt_length
weight = bream_weight + smelt_weight

# 리스트 내포 # 사이킷런이 기대하는 2차원으로 만들어줘야 함
fish_data = [[l, w] for l, w in zip(length, weight)]    # 2차원 배열
print("fish_data:", fish_data)

print("")

# 정답 준비
fish_target = [1]*35 + [0]*14
print("fish_target:", fish_target)

print("")

# k-최근접 이웃: 주변의 데이터값을 보는 방식. 기본값이 5임.
from sklearn.neighbors import KNeighborsClassifier  # 사이킷런의 알고리즘을 불러옴
kn = KNeighborsClassifier() # 클래스의 인스턴스 kn을 만듦
kn.fit(fish_data, fish_target)  # fit()을 통해 학습. 이때 kn을 모델이라고 함.  
print("정확도:", kn.score(fish_data, fish_target)) # 정확도. 1.0이 나오는데 다 맞췄다는 뜻.

print("")

def answer_fish(value):
    if value == 1:
        return "도미"
    else:
        return "빙어"

# predict([[]])는 결과를 예측하는 역할. 
fish1 = kn.predict([[30, 600]])   # 이 객체를 출력하면 [1] 나옴.
print("이 생선은?:", answer_fish(fish1))


print("")

# 무조건 도미
kn49 = KNeighborsClassifier(n_neighbors=49) # 기본값이 5이므로 49로 바꿈. 근데 이러면 무조건 도미 나옴.
kn49.fit(fish_data, fish_target)
print("정확도:", kn49.score(fish_data, fish_target))    # 0.7~~
print("35 / 49 =", 35/49)    # kn49.score과 같다.
