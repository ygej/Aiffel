#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200917_01

# # 문제 1
# [시저 암호](https://programmers.co.kr/learn/courses/30/lessons/12926)

# In[24]:


s = 'a B z'
n =  4
def solution(s, n):
    
    up_list = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    low_list = 'abcdefghijklmnopqrstuvwxyz'
    
    answer = []
    s = list(s)
    
    for i in range(len(s)):
        if s[i] in up_list:
            num = up_list.find(s[i])+n
            answer.append(up_list[num%26])
        elif s[i] in low_list:
            num = low_list.find(s[i])+n
            answer.append(low_list[num%26])
        elif s[i] == ' ':
            answer.append(' ')
            
        
    return ''.join(answer)
solution(s, n)


# In[ ]:




