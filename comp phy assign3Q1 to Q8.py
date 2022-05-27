#!/usr/bin/env python
# coding: utf-8

# In[31]:


#Q1
import numpy as np
import timeit
import matplotlib.pyplot as plt
import seaborn as sns  #for comparing LCR with uniform distribution
from scipy.stats import uniform


start1 = timeit.default_timer()
a = 9823468789     
c = 8785443547     
m = 9867553445
x = 1
LCRlist = []
for i in range(10000):
    x = (a*x+c)%m
    LCRlist.append(x/(m-1))
    
end1 = timeit.default_timer()

plt.hist(LCRlist, bins=100)
plt.title('linear congruential random number generator')
plt.xlabel('no')
plt.ylabel('frequency')
plt.show()
size = 10000
x = np.arange(0, 1, 1/size)
y = uniform.pdf(x ,0, 1)
plt.plot(x,y,label='uniform pdf')

sns.kdeplot(LCRlist,label='LCR pdf')


plt.show()


# In[32]:


#Q2
import numpy as np
import timeit
import matplotlib.pyplot as plt
from scipy.stats import uniform
import seaborn as sns  #for comparing LCR with uniform distribution
start2 = timeit.default_timer()
randlist = np.random.uniform(0,1,10000)
end2 = timeit.default_timer()

plt.hist(randlist, bins=100)
plt.title('uniformly distributed random number generator')
plt.xlabel('no')
plt.ylabel('frequency')
plt.show()
size = 10000
x = np.arange(0, 1, 1/size)
y = uniform.pdf(x ,0, 1)
plt.plot(x,y,label='uniform pdf')
sns.kdeplot(randlist,label='rand() pdf')
plt.show()


# In[33]:


#Q3
print('time taken to generate 10000 random no by LCR = ',end1-start1)
print('time taken to generate 10000 random no by rand() = ',end2-start2)


# In[34]:


#Q4
import numpy as np
import timeit
import matplotlib.pyplot as plt
import seaborn as sns  #for comparing LCR with uniform distribution
import random
import math
from scipy.stats import expon

exp=np.loadtxt("exp pdf data.txt")
plt.hist(exp, bins=100)
plt.title("exponential distribution with mean=0.5")
plt.xlabel('no')
plt.ylabel('frequency')
plt.show()


size = 10000
x = np.arange(0, 10, 1/size)
y = expon.pdf(x, loc=0, scale=2)
plt.plot(x,y,label='exp')
sns.kdeplot(exp,label='exp obtained')


plt.show()


# In[35]:


#Q5
from scipy.stats import norm
x1=np.random.rand(10000)
x2=np.random.rand(10000)
y1=np.sqrt(-2*np.log(x1))*np.cos(2*np.pi*x2)
y2=np.sqrt(-2*np.log(x1))*np.sin(2*np.pi*x2)
fig, axs = plt.subplots(1, 1,
                        tight_layout = True)
 
axs.hist(y1, bins = 100)
# Show plot

plt.show()
size = 10000

x = np.arange(-10, 10, 1/size)
y = norm.pdf(x, loc=0, scale=1)
plt.plot(x,y,label='normal')
sns.kdeplot(y1,label='normal obtained')
plt.show()


# In[36]:


#Q6
import numpy as np
import matplotlib.pyplot as plt

def f_x(x):
    f = np.sqrt(2/np.pi)*np.exp(-x**2/2);
    return f

N = 10000;
x = np.random.rand(N)*10;
y = np.random.rand(N);


x_good  = [];
for i in range(len(x)):
    if (y[i]<=f_x(x[i])):
        x_good.append(x[i])

x = np.linspace(0, 10, 10000);
plt.hist(x_good,100)
plt.show()
size = 10000

x1 = np.arange(0, 10, 1/size)
y1=[None]*len(x1)

for i in range(0,len(x1)):
    y1[i]=f_x(x1[i])
plt.plot(x1,y1,label='actual pdf')
sns.kdeplot(x_good,label='obtained pdf')
plt.show()


# In[37]:


#Q7
import numpy as np
import scipy.stats as ss
N = 144
n1 = [4,10,10,13,20,18,18,11,13,14,13]
V=0
pstandard = [1/36,1/18,1/12,1/9,5/36,1/6,5/36,1/9,1/12,1/18,1/36]
ps = np.array(pstandard)
for i in range (len(n1)):
    V +=(n1[i]-N*ps[i])**2/(N*ps[i])
PofV = 1.0 - ss.chi2.cdf(V,10.0)
print(PofV)
if PofV>0.99 or PofV<0.01:
      print("not sufficienty random")
if PofV>0.95 and PofV<0.99:
      print("suspect")
if PofV>0.01 and PofV<0.05:
      print("suspect")
if PofV>0.90 and PofV<0.95:
      print("almost suspect")
if PofV>0.05 and PofV<0.1:
      print("almost suspect")
if PofV>0.10 and PofV<0.90:
      print("sufficiently random")       


# In[38]:


#Q7
n2 = [3,7,11,15,19,24,21,17,13,9,5]
V=0
pstandard = [1/36,1/18,1/12,1/9,5/36,1/6,5/36,1/9,1/12,1/18,1/36]
ps = np.array(pstandard)
for i in range (len(n2)):
    V +=(n2[i]-N*ps[i])**2/(N*ps[i])
PofV = 1.0 - ss.chi2.cdf(V,10.0)
print(PofV)
if PofV>0.99 or PofV<0.01:
      print("not sufficiently random")
if PofV>0.95 and PofV<0.99:
      print("suspect")
if PofV>0.01 and PofV<0.05:
      print("suspect")
if PofV>0.90 and PofV<0.95:
      print("almost suspect")
if PofV>0.05 and PofV<0.1:
      print("almost suspect")
if PofV>0.10 and PofV<0.90:
      print("sufficiently random")


# In[39]:


#Q8
import numpy as np
dim=2
count_in_sphere = 0
N=10000
radius=1.0
squareside=2*radius

for count_loops in range(N):
        point = np.random.uniform(-1.0, 1.0, dim)
        distance = np.linalg.norm(point)
        if distance < radius:
            count_in_sphere += 1

vol=(squareside**dim) * (count_in_sphere / N)
vol  #volume for circle is equivalent to area


# In[40]:


#Q8
import numpy as np
dim=10
count_in_sphere = 0
N=10000
radius=1.0
squareside=2*radius

for count_loops in range(N):
    point = np.random.uniform(-1.0, 1.0, dim)
    distance = np.linalg.norm(point)
    if distance < radius:
        count_in_sphere += 1

vol=(squareside**dim) * (count_in_sphere / N)
vol

