#!/usr/bin/env python
# coding: utf-8

# # Gradient Descent
# Gradient Descent is an optimization algorithm for finding a local minimum of a differentiable function. Gradient descent is simply used in machine learning to find the values of a function's parameters (coefficients) that minimize a cost function as far as possible.
# We start by defining the initial parameter's values and from there gradient descent uses calculus to iteratively adjust the values so they minimize the given cost-function.

# In[5]:


import matplotlib.pyplot as plt


# In[27]:


x = np.array([85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([250, 260, 270, 280, 290, 300, 310, 320, 330])
plt.plot(x,y,color='red',alpha=0.6)
plt.show()


# In[9]:


import numpy as np
xpoints = np.array([0, 6])
ypoints = np.array([0, 250])

plt.plot(xpoints, ypoints)
plt.show()


# In[13]:


ypoints = np.array([3, 8, 1, 10])

plt.plot(ypoints, marker = 'o')
plt.show()


# In[16]:


x = np.array([80, 85, 90, 95, 100, 105, 110, 115, 120, 125])
y = np.array([240, 250, 260, 270, 280, 290, 300, 310, 320, 330])

plt.plot(x, y)
plt.title("Sports Watch Data")
plt.xlabel("Average Pulse")
plt.ylabel("Calorie Burnage")

plt.grid(color = 'green', linestyle = '--', linewidth = 0.5)
plt.show()


# In[17]:


#plot 1:
x = np.array([0, 1, 2, 3])
y = np.array([3, 8, 1, 10])

plt.subplot(1, 2, 1)
plt.plot(x,y)

#plot 2:
x = np.array([0, 1, 2, 3])
y = np.array([10, 20, 30, 40])

plt.subplot(1, 2, 2)
plt.plot(x,y)

plt.show()


# In[18]:


#normal dist with mean 170, st.deviation 10 and sample size 250
x = np.random.normal(170, 10, 250)

plt.hist(x)
plt.show()


# In[32]:


import numpy as np

def gradient_descent(x,y):      
    m_curr = b_curr = 0
    iterations = 1000
    n = len(x)
    learning_rate = 0.08
    
    for i in range(iterations):
        y_predicted = m_curr * x + b_curr
        cost = (1/n) * sum([val**2 for val in (y-y_predicted)])
        md = - (2/n) * sum(x*(y-y_predicted))
        bd = - (2/n) * sum(y-y_predicted)
        m_curr = m_curr - learning_rate * md
        b_curr = b_curr - learning_rate * bd
        print("m{}, b {}, cost {}, iteration {}". format(m_curr,b_curr,cost,i))
        
        

x = np.array([1,2,3,4,5])
y = np.array([5,7,9,11,13])

gradient_descent(x,y)      


# In[ ]:




