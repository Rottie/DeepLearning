
# coding: utf-8

# In[1]:



#步驟 1: 資料處理
#1A:匯入所需模組

from keras.datasets import imdb
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer

#1B.讀取 IMdb資料集目錄

#建立 rm_tag 函數移除文字中的 html tag
import re
def rm_tags(text):
    re_tag = re.compile(r'<[^>]+>')
    return re_tag.sub('',text)

#建立 read_files函數讀取 1MDb 檔案目錄
import os
def read_files(filetype):
    path = "C:/pythonwork/Keras/data/aclImdb/"
    file_list = []
    
    positive_path = path + filetype +"/pos/"
    for f in os.listdir(positive_path):
        file_list += [positive_path+f]
    
    negative_path = path + filetype +"/neg/"
    for f in os.listdir(negative_path):
        file_list += [negative_path+f]
    
    print('read',filetype,'files:',len(file_list))
    
    all_labels = ([1] * 12500 + [0] * 12500)
    all_texts = []
    
    for fi in file_list:
        with open(fi,encoding ='utf8') as file_input:
            all_texts += [rm_tags(" ".join(file_input.readlines()))]
    
    return all_labels,all_texts



# In[2]:


#讀取訓練資料
y_train,train_text=read_files("train")

#讀取測試資料
y_test,test_text=read_files("test")


# In[3]:


#1C.建立 token

token=Tokenizer(num_words=3800)
token.fit_on_texts(train_text)


# In[4]:


#1D. 將 [影評文字] 轉換成 [數字 list]

x_train_seq = token.texts_to_sequences(train_text)
x_test_seq  = token.texts_to_sequences(test_text)


# In[5]:


#1E.截長補短讓所有 [數字 list] 長度都是 100

x_train=sequence.pad_sequences(x_train_seq, maxlen=380)
x_test =sequence.pad_sequences(x_test_seq , maxlen=380)


# In[6]:


#步驟2:加入 Embedding 層
#2A.匯入所需模組

from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.embeddings import Embedding


#2B.建立模型

model =Sequential()



#2C.將 [Embedding 層] 加入模型

model.add(Embedding(output_dim=32,
                   input_dim=3800,
                   input_length=380))

model.add(Dropout(0.2))


# In[7]:


#步驟 3:建立多層感知器模型
#3A.將 [Flatten層] 加入模型

model.add(Flatten())


#3B.將 [隱藏層] 加入模型

model.add(Dense(units=256,
               activation='relu'))

model.add(Dropout(0.35))



#3C.將 [輸出層] 加入模型

model.add(Dense(units=1,
               activation='sigmoid'))


# In[8]:


#3D.查看模型的摘要

model.summary()


# In[9]:


#步驟 4:訓練模型
#4A.定義訓練方式

model.compile(loss='binary_crossentropy',
             optimizer='adam',
             metrics=['accuracy'])


#4B.開始訓練

train_history =model.fit(x_train,y_train,batch_size=100,
                        epochs=10,verbose=2,
                        validation_split=0.2)


# In[10]:


#步驟 5:評估模型準確率
#5A.評估模型準確率

scores = model.evaluate(x_test,y_test,verbose=1)          #因爲詞典大小增加3800
scores[1]


# In[11]:


#步驟6:進行預測
#6A.執行預測

predict= model.predict_classes(x_test)


# In[12]:


#6B.預測結果(查看前 10筆資料)

predict[:10]


# In[13]:


#6C.預測結果(查看1 1 維陣列 predict_class)

predict_classes=predict.reshape(-1)
predict_classes[:10]


# In[14]:


#步驟 7:查看測試資料預測結果
#7A.建立 display_test_Sentiment 函數

SentimentDict={1:'正面的',0:'負面的'}
def display_test_Sentiment(i):
    print(test_text[i])
    print('label真實值:',SentimentDict[y_test[i]],
         '預測結果:',SentimentDict[predict_classes[i]])



#7B.顯示第 2筆預測結果

display_test_Sentiment(2)


# In[15]:


#7C.顯示第 12502 筆預測結果

display_test_Sentiment(12502)

display_test_Sentiment(12499)


# In[30]:


###################################
#I.查看電影的影評
#步驟 1:複製 1MDb 影評,貼在 input_text 變數裏面
input_text='''
This is the movie I have been waiting for for a very long time. I am an avid Ironman reader. I have collected the comics all my life (from #1 in 1968 to the latest in 2008)...40 years of Ironman. Since I was a kid, I used to say to my friends that they should make an Ironman movie, but everyone laughed and said that the special effects would look ridiculous...mind you, that was back in like in the early 1980's. But, now we are in the age of CGI, and what an age it is. Just for you Ironman fans, to see him come to life with such spectacular graphics is reason enough to spend your hard-earned $10. The plot is also pretty well thought out, and the acting is just fine. What better pick could you have for Tony Stark than Robert Downey Jr.? (just wait for the sequels when they can delve into his alcoholism...Mr. Downey has been there and beyond...that's probably why they chose him for the role...fore-thought)The rest of the cast is right on par, as well. The pace is brisk, and the whole thing works as a great addition to the Marvel Universe in the Cinema!! Enjoy!
'''


#步驟 2:將 [影評文字] 轉換成 [數字 list]

input_seq = token.texts_to_sequences([input_text])




# In[31]:


#步驟 3:查看 [數字 list]

print(input_seq[0])


# In[32]:


#步驟 4:查看 [數字 list] 長度

len(input_seq[0])


# In[33]:


#步驟 5: 將 [數字 list] 截取長度為 100

pad_input_seq =sequence.pad_sequences(input_seq , maxlen=380)


# In[34]:


#步驟 6: 截長補短後查看 [數字 list]長度

len(pad_input_seq[0])


# In[35]:


#步驟 7: 使用多層感知器(MLP)模型,進行預測

predict_result = model.predict_classes(pad_input_seq)


# In[36]:



#步驟 8: 查看預測結果

predict_result


# In[37]:



#步驟 9:查看預測結果 

predict_result[0][0]


# In[38]:



#步驟 10:執行預測

SentimentDict[predict_result[0][0]]


# In[39]:



#######################
#II.用 Predict_review 方式評估其他電影評估,方便之後評估方式
#步驟 1: 建立 predict_review 函數


def predict_review(input_text):
    input_seq =token.texts_to_sequences([input_text])
    pad_input_seq =sequence.pad_sequences(input_seq, maxlen=380)
    predict_result=model.predict_classes(pad_input_seq)
    print(SentimentDict[predict_result[0][0]])




# In[40]:



#步驟 2: 測試` 1 或 0

predict_review('''
This is the movie I have been waiting for for a very long time. I am an avid Ironman reader. I have collected the comics all my life (from #1 in 1968 to the latest in 2008)...40 years of Ironman. Since I was a kid, I used to say to my friends that they should make an Ironman movie, but everyone laughed and said that the special effects would look ridiculous...mind you, that was back in like in the early 1980's. But, now we are in the age of CGI, and what an age it is. Just for you Ironman fans, to see him come to life with such spectacular graphics is reason enough to spend your hard-earned $10. The plot is also pretty well thought out, and the acting is just fine. What better pick could you have for Tony Stark than Robert Downey Jr.? (just wait for the sequels when they can delve into his alcoholism...Mr. Downey has been there and beyond...that's probably why they chose him for the role...fore-thought)The rest of the cast is right on par, as well. The pace is brisk, and the whole thing works as a great addition to the Marvel Universe in the Cinema!! Enjoy!
''')


# In[ ]:





# In[ ]:




