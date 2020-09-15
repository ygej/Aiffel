#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200915_01

# # 문제 1
# [두 개 뽑아서 더하기](https://programmers.co.kr/learn/courses/30/lessons/68644)

# In[21]:


numbers = [5,0,2,7]

def solution(numbers):
    answer = []
    for i in range(len(numbers)):
        for j in range(i+1, len(numbers)):
            if numbers[i]+numbers[j] not in answer:
                answer.append(numbers[i]+numbers[j])
    answer.sort()
    return answer
solution(numbers)


# In[ ]:




