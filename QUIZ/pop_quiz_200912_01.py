#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200912_01

# # 문제 1
# [최대공약수와 최소공배수](https://programmers.co.kr/learn/courses/30/lessons/12940)

# In[3]:


def solution(n, m):
    min_num = min(n,m)
    i = [i for i in range(min_num+1,0,-1) if n%i == 0 and m%i == 0][0]
    j = int(n*m/i)
    answer = [i,j]
    return answer


# In[4]:


solution(2, 5)

