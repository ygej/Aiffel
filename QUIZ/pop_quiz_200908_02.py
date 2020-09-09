#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200908

# ### 문제 2
# 2016년 1월 1일은 금요일입니다. 2016년 a월 b일은 무슨 요일일까요? 두 수 a ,b를 입력받아 2016년 a월 b일이 무슨 요일인지 리턴하는 함수, solution을 완성하세요. 요일의 이름은 일요일부터 토요일까지 각각 SUN,MON,TUE,WED,THU,FRI,SAT
# 입니다. 예를 들어 a=5, b=24라면 5월 24일은 화요일이므로 문자열 TUE를 반환하세요.
# 참고로 2016년은 윤년입니다(2월이 29일까지 있는 해이죠)
# 앞으로 정답의 형태는 오늘 배운 함수로!! 진행해주시면 됩니다. 예를 들어 위의 문제 같은 경우는 입력값을 a, b로 받고 리턴은 12를 한다했으므로 의 꼴로 짜주시면 됩니다! 그러니 solution(5, 24)의 결과는 'TUE'가 나와야겠죠?
# 댓글로 제가 테스트 케이스까지 적어둘테니 모두 돌아가는지 확인해보세요!

# In[50]:


def solution(a,b):
    day_num = []
    day_list = ['FRI','SAT','SUN','MON','TUE','WED','THU']
    for i in range(1,13):
        if i == 1 or i == 3 or i == 5 or i == 7 or i == 8 or i == 10 or i == 12:
            for j in range(1,32):
                day_num.append([i,j])
    
        elif i == 4 or i == 6 or i == 9 or i == 11:
            for j in range(1,31):
                day_num.append([i,j])
        elif i == 2:
            for j in range(1,30):
                day_num.append([i,j])           
                
    print(day_list[day_num.index([a,b])%7])


# In[51]:


solution(5,24)
solution(12,31)
solution(1,1)
solution(3,1)
solution(12,25)


# In[ ]:




