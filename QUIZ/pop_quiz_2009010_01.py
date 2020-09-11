#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200910_01

# # 문제 1
# [같은 숫자는 싫어](https://programmers.co.kr/learn/courses/30/lessons/12906)

# In[2]:


def solution(arr):
    answer = []
    for i in range(len(arr)):
        if [arr[i]] != arr[i+1:i+2]:
            answer.append(arr[i])
    return answer


# In[3]:


arr = [1,1,3,3,0,1,1]
solution(arr)


# # 문제 2
# [서울에서 김서방 찾기](https://programmers.co.kr/learn/courses/30/lessons/12919)

# In[2]:


def solution(seoul):
    answer = '김서방은 %d에 있다'% (int(seoul.index('Kim')))
    return answer


# # 문제 3
# [짝수와 홀수](https://programmers.co.kr/learn/courses/30/lessons/12937)

# In[ ]:


def solution(num):
    answer = ""
    if num % 2 == 0:
        answer = "Even"
    elif num == 0:
        answer = "Even"
    else:
        answer = "Odd"

    return answer

