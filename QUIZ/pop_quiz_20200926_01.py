#!/usr/bin/env python
# coding: utf-8

# # 깜퀴_20200926_01

# # 문제 1
# [클래스와 터틀 활용해서 직사각형 만들어보기](https://www.notion.so/7b79f16959b3462a98feb345cc6c5583)

# In[1]:


import turtle as t

class DrawPolynomial:
    def __init__(self, width, height, color, shape='circle'):
        self.width = width
        self.height = height
        self.color = color
        self.shape = shape


    def area(self):
        s_area = self.width * self.height
        print(s_area)


    def get_attr(self, attr):
        if attr == 'width':
            print(self.width)
        elif attr == 'height':
            print(self.height)
        elif attr == 'color':
            print(self.color)


    def change_attr(self, width=None, height=None, color=None):
        if width >= 0:
            self.width = width
        elif height >= 0:
            self.height = height
        elif self.color != None:
            self.color = color
        else:
            pass
        
        

    def draw(self):
        t.shape(self.shape)
        t.color(self.color)
        t.begin_fill()
        
        for i in range(4):
            t.forward(self.width)
            for j in range(0,1):
                t.right(90)
                t.forward(self.height)
        t.end_fill()
        t.exitonclick()


# In[2]:


poly = DrawPolynomial(width=100, height=100, color='red')
poly.area()
poly.get_attr('width')
poly.change_attr(width=120)
poly.get_attr('width')


# In[3]:


poly.draw()


# In[ ]:




