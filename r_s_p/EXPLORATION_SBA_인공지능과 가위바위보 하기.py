#!/usr/bin/env python
# coding: utf-8

# # EXPLORATION_SBA : 가위 바위 보 하기

# ## LOAD DATA

# In[1]:


import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)   # Tensorflow의 버전을 출력


# In[2]:


# PIL 라이브러리가 설치되어 있지 않다면 설치
#!pip install pillow   

from PIL import Image
import os, glob

print("PIL 라이브러리 import 완료!")


# In[3]:


# 가위, 바위, 보 이미지를 불러와서 28x28 사이즈로 변경
import os
# 이미지가 저장된 디렉토리 아래의 모든 jpg 파일을 읽어들여서

image_list = ['rock', 'scissor', 'paper']

for i in image_list:
    path = "/project/Aiffel/r_s_p/" +i
    image_dir_path = os.getenv("HOME") + path
    print("이미지 디렉토리 경로: ", image_dir_path)
    images=glob.glob(image_dir_path + "/*.jpg")  

    # 파일마다 모두 28x28 사이즈로 바꾸어 저장합니다.
    target_size=(28,28)
    for img in images:
        old_img=Image.open(img)
        new_img=old_img.resize(target_size,Image.ANTIALIAS)
        new_img.save(img,"JPEG")

    print("%s 이미지 resize 완료!" %(i))


# In[84]:


img_path = "/home/theo/project/Aiffel/r_s_p/paper"
def load_data(img_path):
    # 가위 : 0, 바위 : 1, 보 : 2
    number_of_data=1200   # 가위바위보 이미지 개수 총합에 주의하세요.
    img_size=28
    color=3
    #이미지 데이터와 라벨(가위 : 0, 바위 : 1, 보 : 2) 데이터를 담을 행렬(matrix) 영역을 생성합니다.
    imgs=np.zeros(number_of_data*img_size*img_size*color,dtype=np.int32).reshape(number_of_data,img_size,img_size,color)
    labels=np.zeros(number_of_data,dtype=np.int32)

    idx=0
    for file in glob.iglob(img_path+'/scissor/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=0   # 가위 : 0
        idx=idx+1

    for file in glob.iglob(img_path+'/rock/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=1   # 바위 : 1
        idx=idx+1       
    
    for file in glob.iglob(img_path+'/paper/*.jpg'):
        img = np.array(Image.open(file),dtype=np.int32)
        imgs[idx,:,:,:]=img    # 데이터 영역에 이미지 행렬을 복사
        labels[idx]=2   # 보 : 2
        idx=idx+1
        
    print("학습데이터(x_train)의 이미지 개수는",idx,"입니다.")
    return imgs, labels

image_dir_path = os.getenv("HOME") + "/project/Aiffel/r_s_p"
(x_train, y_train)=load_data(image_dir_path)
x_train_norm = x_train/255.0   # 입력은 0~1 사이의 값으로 정규화

print("x_train shape: {}".format(x_train.shape))
print("y_train shape: {}".format(y_train.shape))


# In[81]:


import matplotlib.pyplot as plt
plt.imshow(x_train[1199])
print('라벨: ', y_train[1199])


# In[38]:


# 순서대로 학습 데이터, 테스트 데이터 나누기
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_train_norm, y_train, test_size=0.25, shuffle=False)

print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[85]:


# 무작위로 학습 데이터, 테스트 데이터 나누기
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(x_train_norm, y_train, test_size=0.25, shuffle=True, random_state=1000)


# In[86]:


# 학습용 데이터
print(X_train.shape)
print(X_test.shape)
print(y_train.shape)
print(y_test.shape)


# In[69]:


plt.imshow(X_train[0])
print('라벨: ', y_train[0])


# # Model Train

# In[90]:


n_channel_1=32
n_channel_2=128
n_dense=512
n_epochs = 15
n_class_num = 3

model=keras.models.Sequential()
model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
model.add(keras.layers.MaxPool2D(2,2))
model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
model.add(keras.layers.MaxPooling2D((2,2)))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(n_dense, activation='relu'))
model.add(keras.layers.Dense(n_class_num, activation='softmax'))

model.summary()
model.compile(optimizer='adam',             
              loss='sparse_categorical_crossentropy',
             metrics=['accuracy'])


# In[91]:


model.fit(X_train, y_train, epochs=n_epochs)

print(X_train.shape)
print(y_train.shape)


# In[92]:


# Model Test
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print("test_loss: {} ".format(test_loss))
print("test_accuracy: {}".format(test_accuracy))


# In[87]:


# 전수조사


# In[89]:


for i in range(5,10):
    n_channel_1= 2**i
    for j in range(5,10):
        n_channel_2 = 2**j
        for k in range(5,10):
            n_dense= 2**k
            for l in [5,10,15]:
                n_epochs = l
                n_class_num = 3

                model=keras.models.Sequential()
                model.add(keras.layers.Conv2D(n_channel_1, (3,3), activation='relu', input_shape=(28,28,3)))
                model.add(keras.layers.MaxPool2D(2,2))
                model.add(keras.layers.Conv2D(n_channel_2, (3,3), activation='relu'))
                model.add(keras.layers.MaxPooling2D((2,2)))
                model.add(keras.layers.Flatten())
                model.add(keras.layers.Dense(n_dense, activation='relu'))
                model.add(keras.layers.Dense(n_class_num, activation='softmax'))
                
                model.compile(optimizer='adam',             
                loss='sparse_categorical_crossentropy',
                metrics=['accuracy'])
                
                # Model train
                model.fit(X_train, y_train, epochs=n_epochs)
                print('n_channel_1: ', n_channel_1)
                print('n_channel_2: ', n_channel_2)
                print('n_dense: ', n_dense)
                print('n_epochs: ', n_epochs)
                
                # Model Test
                test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
                print("test_loss: {} ".format(test_loss))
                print("test_accuracy: {}".format(test_accuracy))
                


# # 최적값
# n_channel_1=32
# n_channel_2=128
# n_dense=512
# n_epochs = 15

# # 잘못 추론한 경우 확인

# In[93]:


predicted_result = model.predict(X_test)  # model이 추론한 확률값. 
predicted_labels = np.argmax(predicted_result, axis=1)

idx=100  #100번째 x_test를 살펴보자. 
print('model.predict() 결과 : ', predicted_result[idx])
print('model이 추론한 가장 가능성이 높은 결과 : ', predicted_labels[idx])
print('실제 데이터의 라벨 : ', y_test[idx])


# In[94]:


plt.imshow(X_test[idx],cmap=plt.cm.binary)
plt.show()


# In[95]:


import random
wrong_predict_list=[]
for i, _ in enumerate(predicted_labels):
    # i번째 test_labels과 y_test이 다른 경우만 모아 봅시다. 
    if predicted_labels[i] != y_test[i]:
        wrong_predict_list.append(i)

# wrong_predict_list 에서 랜덤하게 20개만 뽑아봅시다.
samples = random.choices(population=wrong_predict_list, k=20)

for n in samples:
    print("예측확률분포: " + str(predicted_result[n]))
    print("라벨: " + str(y_test[n]) + ", 예측결과: " + str(predicted_labels[n]))
    plt.imshow(X_test[n], cmap=plt.cm.binary)
    plt.show()


# In[ ]:




