
# coding: utf-8

# In[1]:


#10.1建立計算圖

#1A.匯入tensorflow模組
import tensorflow as tf


# In[2]:


#1B.建立 const 常數
ts_c = tf.constant(2,name='ts_c')


# In[3]:


#1C.查看 Tensorflow 常數
ts_c


# In[4]:


#1D.建立 Tensorflow 變數
ts_x = tf.Variable(ts_c+5,name='ts_x')


# In[5]:


#1E.查看 Tensorflow 變數
ts_x


# In[6]:


#10.2 執行 計算圖
#2A.建立 Session

sess=tf.Session()


# In[7]:


#2B.執行 Tensorflow 起始化變數

init = tf.global_variables_initializer()
sess.run(init)


# In[8]:


#2C.使用 sess.run 顯示 Tensorflow 常數

print('ts_c=',sess.run(ts_c))


# In[9]:


#2D.使用 sess.run 顯示 Tensorflow 變數

print('ts_x=',sess.run(ts_x))


# In[10]:


#2E.使用 .eval ()方法顯示 TensorFlow 常數

print('ts_c=',ts_c.eval(session=sess))


# In[11]:


#2F.使用 .eval ()方法顯示 TensorFlow 變數

print('ts_x=',ts_x.eval(session=sess))


# In[12]:


#2G.關閉 TensorFlow session

sess.close()    


# In[13]:


#10.3全部指令全部一起執行

import tensorflow as tf
ts_c = tf.constant(2,name='ts_c')
ts_x = tf.Variable(ts_c+5,name='ts_x')

sess=tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
print('ts_c=',sess.run(ts_c))
print('ts_x=',sess.run(ts_x))
sess.close()


# In[14]:


#10.4 With語法開啟Session 并且自動關閉


import tensorflow as tf
ts_c = tf.constant(2,name='ts_c')
ts_x = tf.Variable(ts_c+5,name='ts_x')
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('ts_c=',sess.run(ts_c))
    print('ts_x=',sess.run(ts_x))


# In[16]:


#10.5 Tensorflow Placeholder
#5A.建立 [計算圖]

width = tf.placeholder("int32")
height = tf.placeholder("int32")
area=tf.multiply(width,height)


# In[17]:


#5B.執行 [計算圖]

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area=',sess.run(area,feed_dict={width: 6, height: 8}))


# In[18]:


#10.6 建立 1,2維 張量

#6A.建立 1維張量(向量)

ts_X = tf.Variable([0.4,0.2,0.4])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X=sess.run(ts_X)
    print(X)


# In[19]:


#6B.查看 1 維的 Tensor shape

print(X.shape)


# In[20]:


#6C. 建立 2維的 Tensor (1x3矩陣)

ts_X = tf.Variable([[0.4,0.2,0.4]])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    X=sess.run(ts_X)
    print(X)   


# In[21]:


#6D.查看 2維的 tensor shape (1x3矩陣)

print('shape:',X.shape)


# In[22]:


#6E.建立二維的張量 (3x2矩陣)

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    W_array=sess.run(W)
    print(W_array)   


# In[23]:


#6F.建立 二維的 tensor shape (3x2矩陣)


print(W_array.shape)


# In[25]:


#10.7 矩陣基本運算

#7A.矩陣乘法 (tf.matmul)

X = tf.Variable([[1.,1.,1.]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])
                        
XW =tf.matmul(X,W )
                       
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print(sess.run(XW ))


# In[26]:


#7B.矩陣加法

b = tf.Variable([[ 0.1,0.2]])
XW =tf.Variable([[-1.3,0.4]])

Sum =XW+b
with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('Sum:')    
    print(sess.run(Sum ))


# In[27]:


#7C.矩陣乘法與加法

X = tf.Variable([[1.,1.,1.]])

W = tf.Variable([[-0.5,-0.2 ],
                 [-0.3, 0.4 ],
                 [-0.5, 0.2 ]])
                         

b = tf.Variable([[0.1,0.2]])
    
XWb =tf.matmul(X,W)+b


with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('XWb:')    
    print(sess.run(XWb ))

