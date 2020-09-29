#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200929_01

# # 문제 1
# 메이플스토리 능력치 뽑기 주사위 만들기
# 
# ![메이플 스토리 능력치 주사위](https://images.app.goo.gl/sZyoAhaj75UspgCCA)

# In[74]:


from random import randrange

class MapleDice:
    def __init__(self, job = None):
        self.stat_list = [i for i in range(0,4)]
        self.options = randrange(4,13)
        self.job = job
        
    def throw(self):
        stat_list = []
        for i in [i for i in range(0,4)]:
            if i == 0:
                i = randrange(4,13)
            elif i == 1 and stat_list[0] > 4 :
                i = randrange(4,18-(stat_list[0]))
            elif i == 2 and sum(stat_list) > 8:
                i = randrange(4, 22 - sum(stat_list))
            elif i == 2 and sum(stat_list) < 9:
                i = randrange(5, 13)
            elif i == 3:
                i = 25 - sum(stat_list)
            else:
                i = randrange(4,13)
            stat_list.append(i)
        print("나의 캐릭터 스탯 \nSTR:{0}, INT:{1}, DEX:{2}, LUK:{3}".format(stat_list[0],stat_list[1],stat_list[2],stat_list[3]))
    
    def auto_mouse(self, job = None):
        self.job = job
        if self.job > 4 or self.job < 0:
            print("숫자를 다시 입력하세요. \n 전사 = 0, 법사 = 1, 궁수 = 2, 도적 = 3")
            
        # 전사 & 궁수
        if self.job == 0 or self.job == 2:
            count = 0
            while True:
                stat_list = []
                for i in [i for i in range(0,4)]:
                    if i == 0:
                        i = randrange(4,13)
                    elif i == 1 and stat_list[0] > 4 :
                        i = randrange(4,18-(stat_list[0]))
                    elif i == 2 and sum(stat_list) > 8:
                        i = randrange(4, 22 - sum(stat_list))
                    elif i == 2 and sum(stat_list) < 9:
                        i = randrange(5, 13)
                    elif i == 3:
                        i = 25 - sum(stat_list)
                    else:
                        i = randrange(4,13)
                    stat_list.append(i)
                
                if stat_list[1] == 4 and stat_list[3] == 4:
                    break
                count += 1
                
        # 법사
        if self.job == 1:
            count = 0
            while True:
                stat_list = []
                for i in [i for i in range(0,4)]:
                    if i == 0:
                        i = randrange(4,13)
                    elif i == 1 and stat_list[0] > 4 :
                        i = randrange(4,18-(stat_list[0]))
                    elif i == 2 and sum(stat_list) > 8:
                        i = randrange(4, 22 - sum(stat_list))
                    elif i == 2 and sum(stat_list) < 9:
                        i = randrange(5, 13)
                    elif i == 3:
                        i = 25 - sum(stat_list)
                    else:
                        i = randrange(4,13)
                    stat_list.append(i)
                
                if stat_list[0] == 4 and stat_list[2] == 4:
                    break
                count += 1
                
        # 도적
        if self.job == 3:
            count = 0
            while True:
                stat_list = []
                for i in [i for i in range(0,4)]:
                    if i == 0:
                        i = randrange(4,13)
                    elif i == 1 and stat_list[0] > 4 :
                        i = randrange(4,18-(stat_list[0]))
                    elif i == 2 and sum(stat_list) > 8:
                        i = randrange(4, 22 - sum(stat_list))
                    elif i == 2 and sum(stat_list) < 9:
                        i = randrange(5, 13)
                    elif i == 3:
                        i = 25 - sum(stat_list)
                    else:
                        i = randrange(4,13)
                    stat_list.append(i)
                
                if stat_list[0] == 4 and stat_list[1] == 4:
                    break
                count += 1
                   
                
        print("오토마우스 결과\n나의 캐릭터 스탯\nSTR:{0}, INT:{1}, DEX:{2}, LUK:{3}\n{4}번 만에 성공했습니다!".format(stat_list[0],stat_list[1],stat_list[2],stat_list[3],count))


# In[75]:


my_stat = MapleDice()
my_stat.throw()


# In[76]:


my_stat.auto_mouse(3)

