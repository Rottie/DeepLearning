
# coding: utf-8

# In[1]:


#11.TensorBoard

#11A.建立 TensorFLow Variable 變數

import tensorflow as tf
width = tf.placeholder("int32",name='width')
height = tf.placeholder("int32",name='height')   #設定  name 參數
area=tf.multiply(width,height,name='area')  

with tf.Session() as sess:
    init = tf.global_variables_initializer()
    sess.run(init)
    print('area=',sess.run(area,feed_dict={width: 6,height: 8}))


# In[2]:


#11B.建立 Tensorflow Variable 變數

tf.summary.merge_all()              #將要所有顯示在 TensorBoard 資料整合
train_writer = tf.summary.FileWriter('log/area',sess.graph)  
#將所有要顯示在 TensorBoard 資料,寫入 log 檔.Log 檔會儲存在目前程式執行目錄下的 log/area 子目錄


# In[ ]:


#11.C 啓動 Tensorboard
#11A.確認log 目錄檔案,是否產生
# dir C:\NLP\log\area

#啓動 tensorflow-gpu虛擬環境
#activate tensorflow-gpu

#啓動 tensorboard
#tensorboard --logdir=C:\NLP\log\area

#在 Anaconda prompt 那裏複製以下網址就可以看到圖片
#http://Alteisen:6006(每個筆電都不一樣)

