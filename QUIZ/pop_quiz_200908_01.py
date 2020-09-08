#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200908

# ### 문제 1
# 두 정수 a, b가 주어졌을 때 a와 b 사이에 속한 모든 정수의 합을 리턴하는 함수, solution을 완성하세요.
# 예를 들어 a = 3, b = 5인 경우, 3 + 4 + 5 = 12이므로 12를 리턴합니다.
# 앞으로 정답의 형태는 오늘 배운 함수로!! 진행해주시면 됩니다. 예를 들어 위의 문제 같은 경우는 입력값을 a, b로 받고 리턴은 12를 한다했으므로

# In[1]:


def solution(a,b):
    answer = 0
    for i in range (a,b+1):
        answer += i
    return answer


# In[2]:


solution(3,5)

