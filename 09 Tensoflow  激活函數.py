
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from pylab import *


# In[2]:


def show_activation(activation,y_lim=5):
    x=np.arange(-10., 10., 0.01)
    ts_x = tf.Variable(x)
    ts_y =activation(ts_x )
    with tf.Session() as sess:
        init = tf.global_variables_initializer()
        sess.run(init)
        y=sess.run(ts_y)
    ax = gca()
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.spines['bottom'].set_position(('data',0))
    ax.spines['left'].set_position(('data',0))
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    lines=plt.plot(x,y)
    plt.setp(lines, color='b', linewidth=3.0)
    plt.ylim(y_lim*-1-0.1,y_lim+0.1) 
    plt.xlim(-10,10) 

    plt.show()      
   


# In[3]:


show_activation(tf.nn.sigmoid,y_lim=1)


# In[4]:


show_activation(tf.nn.softsign,y_lim=1)


# In[5]:


show_activation(tf.nn.tanh,y_lim=1)


# In[6]:


show_activation(tf.nn.relu,y_lim=10)


# In[7]:


show_activation(tf.nn.softplus,y_lim=10)


# In[8]:


show_activation(tf.nn.elu,y_lim=10)

