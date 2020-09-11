#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200911

# # 문제 1
# 구구단 만들기

# In[1]:


for i in range(1,10):
    for j in range(1,10):
        print(i,'*',j, '=', i*j)


# # 문제 2
# [가운데 글자 가져오기](https://programmers.co.kr/learn/courses/30/lessons/12903)

# In[2]:


def solution(s):
    if len(s)%2 ==0:
        answer = s[(len(s)//2)-1]+ s[len(s)//2]
    else:
        answer = s[len(s)//2]
    return answer


# In[4]:


s = 'abcde'
solution(s)


# # 문제 3
# [K번째 수](https://programmers.co.kr/learn/courses/30/lessons/42748?language=python3)

# In[5]:


def solution(array, commands):
    answer = []
    for i in commands:
        answer_array = array[i[0]-1:i[1]]
        answer_array.sort()
        k = i[2]-1
        answer_k = answer_array[k]
        answer.append(answer_k)
        
    return answer


# In[6]:


array = [1, 5, 2, 6, 3, 7, 4]
commands = [[2, 5, 3], [4, 4, 1], [1, 7, 3]]
solution(array, commands)


# # 문제 4
# [더하기 사이클](https://www.acmicpc.net/problem/1110)

# In[7]:


# 횟수
count = 1

# 새로운 수
N = int(input())
num = ''
if N < 10:
    num = '0' + str(N)
else:
    num = str(N)
    
# 새로운 수[0] + 새로운 수[1]
num_1 = int(num[0]) + int(num[1])

# 자릿수 확인
if num_1 < 10:
    num_1 = '0'+str(num_1)
else:
    num_1 = str(num_1)
    
# 반복문
    
while True:
       
    # 새로운 수
    num = num[1] + num_1[1]
    
    # 확인
    
    if int(num) == N:
        break
    count += 1
    
    # 새로운 수[0] + 새로운 수[1]
    num_1 = int(num[0]) + int(num[1])
    
    # 자릿수 확인
    if num_1 < 10:
        num_1 = '0'+str(num_1)
    else:
        num_1 = str(num_1)
    
        
print(count)


# In[ ]:




