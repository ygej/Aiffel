#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200912_02

# # 문제 2
# [에라토스네스의 체](https://www.acmicpc.net/problem/2960)

# In[ ]:


에라토스테네스의 체는 N보다 작거나 같은 모든 소수를 찾는 유명한 알고리즘이다.

이 알고리즘은 다음과 같다.

2부터 N까지 모든 정수를 적는다.
아직 지우지 않은 수 중 가장 작은 수를 찾는다. 이것을 P라고 하고, 이 수는 소수이다.
P를 지우고, 아직 지우지 않은 P의 배수를 크기 순서대로 지운다.
아직 모든 수를 지우지 않았다면, 다시 2번 단계로 간다.
N, K가 주어졌을 때, K번째 지우는 수를 구하는 프로그램을 작성하시오.


# In[198]:


import math
# 첫째 줄에 N과 K가 주어진다. (1 ≤ K < N, max(2, K) < N ≤ 1000)
n, k = list(map(int,input().split()))

# 소수를 찾는 함수
def find_prime_num(x):
    share = 1
    count = 2
    while count < round(math.sqrt(x)+1):
        if x % count != 0:
            count += 1
        else:
            share = 0
            break
    if share == 0:
        return False
    else:
        return True

# 2부터 N까지 모든 정수를 적는다.
num_list = [i for i in range(2, n+1)]

# 아직 지우지 않은 수 중 가장 작은 수를 찾는다. 이것을 P라고 하고, 이 수는 소수이다.
# P를 지우고, 아직 지우지 않은 P의 배수를 크기 순서대로 지운다.
# 아직 모든 수를 지우지 않았다면, 다시 2번 단계로 간다.
del_list = []
while True:
    p = list(filter(find_prime_num, num_list))[0]
    for i in num_list:
        if len(del_list) == k:
            break
        if i % p == 0:
            del_list.append(i)
            num_list.remove(i)
 
    if len(del_list) ==k:
        break
        
# K번째 지워진 수를 출력       
print(del_list[k-1])


# In[ ]:




