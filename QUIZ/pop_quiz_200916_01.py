#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200916_01

# # 문제 1
# [크레인 인형뽑기 게임](https://programmers.co.kr/learn/courses/30/lessons/64061)

# In[14]:


board = [[0,0,0,0,0],[0,0,1,0,3],[0,2,5,0,1],[4,2,4,4,2],[3,5,1,3,1]]
moves = [1,5,3,5,1,2,1,4]

def solution(board, moves):
    # 인형 바구니
    r = [] 
    
    # 사라진 인형 수
    count = []
    
    # 크레인으로 인형뽑기
    for i in moves:
        for j in range((len(board))):
            if board[j][i-1] != 0:
                r.append(board[j][i-1])
                board[j][i-1] = 0
                break
    
    # 인형 바구니에 인형 터뜨리기
    n = 100
    while True:
        for i in range(len(r)):
            if r[i:i+1] == r[i+1:i+2]:
                count.append(r[i:i+2])
                del r[i:i+2]
            else:
                pass
        if n == 0:
            break
        n -= 1
                
    answer = len(list(filter(None, count)))*2
    return answer
solution(board, moves)

