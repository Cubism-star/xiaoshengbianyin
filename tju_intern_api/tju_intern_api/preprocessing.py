#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
from .youdaoshibie import ASR
import sys
import shutil
import librosa
from scipy.io import wavfile
import numpy as np
import matplotlib.pyplot as plt
#https://blog.csdn.net/TH_NUM/article/details/80597495
import numpy
import scipy.io.wavfile
from hmmlearn import hmm
from matplotlib import pyplot as plt
from scipy.fftpack import dct
import soundfile as sf
import numpy as np
from scipy.signal import lfilter
import re
import tensorflow as tf
from gensim.models.word2vec import Word2Vec


# In[2]:


from tensorflow.keras import optimizers

optimizer = optimizers.Adam(lr=0.001)
def lr(y_true, y_pred):
        return optimizer.lr

def get_lr_metric(optimizer):
    def lr(y_true, y_pred):
        return optimizer.lr
    return lr
from tensorflow.keras.models import load_model
print('cwd')
cwd = os.getcwd()
print(cwd)
# abs_path = (os.path.join(cwd, 'tju_intern_api', 'model', 'maskLSTM1.h5'))
abs_path = r'C:/Users/admin/Desktop/jupyter-workspace/shixun/project\服务器相关\tju_intern_api\tju_intern_api\model\maskLSTM1.h5'
print(abs_path)
yuyin_model = load_model(abs_path, custom_objects={'lr': lr})
abs_path1 = (os.path.join( cwd, 'tju_intern_api','model','test.h5'))
wenzi_model =  load_model(abs_path1)


# In[3]:


def deltas(data,win_len=9):
    '''
    This function is used to calculate the deltas(derivatives) of a sequence use a window_len window.
    This is using a simple linear slope.
    Each row of data is filtered seperately.

    Input:
    data: the speech data to calculate delta
    win_len: the length of a window of the filter

    Output:
    delta: the  derivative of a sequence.

    Written by Li Zetian
    '''
    col,row = data.shape

    if(len(data) == 0):
        return [];
    else:

        #define window shape
        hlen = (int)(np.floor(win_len/2));
        win_len = 2 * hlen + 1;
        win = np.arange(hlen,-hlen-1, -1)

        #pad data by repeating first and last columns
        left_pad = np.repeat(data[:,0],hlen).reshape(-1,hlen)
        right_pad = np.repeat(data[:,row - 1],hlen).reshape(-1,hlen)
        pad_data = np.concatenate((left_pad,data,right_pad),axis=1)
        pad_data = np.where(pad_data == 0, np.finfo(float).eps,pad_data)

        #apply the delta filter
        delta = lfilter(win,1,pad_data,axis = 1,zi=None)

        #Trim edges
        selector = np.arange(0,row) + 2 * hlen
        delta = delta[:,selector]


        return delta
def MFCC_extractor(filepath):
    # sample_rate,signal=scipy.io.wavfile.read('LA_T_1001074.flac')
    signal,sample_rate=sf.read(filepath)

#     print(type(signal))
    print(sample_rate,len(signal))
    #读取前3.5s 的数据
    # signal=signal[0:int(3.5*sample_rate)]
    #     print(signal)
#     if len(signal)< int(1.5*sample_rate):
#         zeronp= np.zeros(int(1.5*sample_rate)-len(signal))
#         signal= list(signal)+list(zeronp)
#     else:
#         signal=signal[0:int(1.5*sample_rate)]
        
#     signal = np.array(signal)



    #预先处理
    pre_emphasis = 0.97
    emphasized_signal = numpy.append(signal[0], signal[1:] - pre_emphasis * signal[:-1])


    frame_size=0.025
    frame_stride=0.01
    frame_length,frame_step=frame_size*sample_rate,frame_stride*sample_rate
    signal_length=len(emphasized_signal)
    frame_length=int(round(frame_length))
    frame_step=int(round(frame_step))
    num_frames=int(numpy.ceil(float(numpy.abs(signal_length-frame_length))/frame_step))


    pad_signal_length=num_frames*frame_step+frame_length
    z=numpy.zeros((pad_signal_length-signal_length))
    pad_signal=numpy.append(emphasized_signal,z)


    indices = numpy.tile(numpy.arange(0, frame_length), (num_frames, 1)) + numpy.tile(numpy.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T

    frames = pad_signal[numpy.mat(indices).astype(numpy.int32, copy=False)]

#     print(frames.shape,'这里！')

    #加上汉明窗
    frames *= numpy.hamming(frame_length)
    # frames *= 0.54 - 0.46 * numpy.cos((2 * numpy.pi * n) / (frame_length - 1))  # Explicit Implementation **

    #傅立叶变换和功率谱
    NFFT = 512
    mag_frames = numpy.absolute(numpy.fft.rfft(frames, NFFT))  # Magnitude of the FFT
    #print(mag_frames.shape)
    pow_frames = ((1.0 / NFFT) * ((mag_frames) ** 2))  # Power Spectrum



    low_freq_mel = 0
    #将频率转换为Mel
    nfilt = 40
    high_freq_mel = (2595 * numpy.log10(1 + (sample_rate / 2) / 700))
    mel_points = numpy.linspace(low_freq_mel, high_freq_mel, nfilt + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz

    bin = numpy.floor((NFFT + 1) * hz_points / sample_rate)

    fbank = numpy.zeros((nfilt, int(numpy.floor(NFFT / 2 + 1))))

    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    filter_banks = numpy.dot(pow_frames, fbank.T)
    filter_banks = numpy.where(filter_banks == 0, numpy.finfo(float).eps, filter_banks)  # Numerical Stability
    filter_banks = 20 * numpy.log10(filter_banks)  # dB

    num_ceps = 20
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (num_ceps + 1)]
    (nframes, ncoeff) = mfcc.shape

    n = numpy.arange(ncoeff)
    cep_lifter =22
    lift = 1 + (cep_lifter / 2) * numpy.sin(numpy.pi * n / cep_lifter)
    mfcc *= lift  #*

    #filter_banks -= (numpy.mean(filter_banks, axis=0) + 1e-8)
    mfcc -= (numpy.mean(mfcc, axis=0) + 1e-8)
    delta = deltas(mfcc.T, 3).T
    double_delta = deltas(delta.T, 3).T
    cell = np.concatenate((mfcc,delta,double_delta),axis=1)
    return cell


# In[4]:


# -*- coding: utf-8 -*-
#
#   author: yanmeng2
#
# 非实时转写调用demo

import base64
import hashlib
import hmac
import json
import os
import time

import requests

lfasr_host = 'http://raasr.xfyun.cn/api'

# 请求的接口名
api_prepare = '/prepare'
api_upload = '/upload'
api_merge = '/merge'
api_get_progress = '/getProgress'
api_get_result = '/getResult'
# 文件分片大小10M
file_piece_sice = 10485760

# ——————————————————转写可配置参数————————————————
# 参数可在官网界面（https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html）查看，根据需求可自行在gene_params方法里添加修改
# 转写类型
lfasr_type = 0
# 是否开启分词
has_participle = 'false'
has_seperate = 'true'
# 多候选词个数
max_alternatives = 0
# 子用户标识
suid = ''


class SliceIdGenerator:
    """slice id生成器"""

    def __init__(self):
        self.__ch = 'aaaaaaaaa`'

    def getNextSliceId(self):
        ch = self.__ch
        j = len(ch) - 1
        while j >= 0:
            cj = ch[j]
            if cj != 'z':
                ch = ch[:j] + chr(ord(cj) + 1) + ch[j + 1:]
                break
            else:
                ch = ch[:j] + 'a' + ch[j + 1:]
                j = j - 1
        self.__ch = ch
        return self.__ch


class RequestApi(object):
    def __init__(self, appid, secret_key, upload_file_path):
        self.appid = appid
        self.secret_key = secret_key
        self.upload_file_path = upload_file_path

    # 根据不同的apiname生成不同的参数,本示例中未使用全部参数您可在官网(https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html)查看后选择适合业务场景的进行更换
    def gene_params(self, apiname, taskid=None, slice_id=None):
        appid = self.appid
        secret_key = self.secret_key
        upload_file_path = self.upload_file_path
        ts = str(int(time.time()))
        m2 = hashlib.md5()
        m2.update((appid + ts).encode('utf-8'))
        md5 = m2.hexdigest()
        md5 = bytes(md5, encoding='utf-8')
        # 以secret_key为key, 上面的md5为msg， 使用hashlib.sha1加密结果为signa
        signa = hmac.new(secret_key.encode('utf-8'), md5, hashlib.sha1).digest()
        signa = base64.b64encode(signa)
        signa = str(signa, 'utf-8')
        file_len = os.path.getsize(upload_file_path)
        file_name = os.path.basename(upload_file_path)
        param_dict = {}

        if apiname == api_prepare:
            # slice_num是指分片数量，如果您使用的音频都是较短音频也可以不分片，直接将slice_num指定为1即可
            slice_num = 1#int(file_len / file_piece_sice) + (0 if (file_len % file_piece_sice == 0) else 1)
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['file_len'] = str(file_len)
            param_dict['file_name'] = file_name
            param_dict['slice_num'] = str(slice_num)
        elif apiname == api_upload:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['slice_id'] = slice_id
        elif apiname == api_merge:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
            param_dict['file_name'] = file_name
        elif apiname == api_get_progress or apiname == api_get_result:
            param_dict['app_id'] = appid
            param_dict['signa'] = signa
            param_dict['ts'] = ts
            param_dict['task_id'] = taskid
        return param_dict

    # 请求和结果解析，结果中各个字段的含义可参考：https://doc.xfyun.cn/rest_api/%E8%AF%AD%E9%9F%B3%E8%BD%AC%E5%86%99.html
    def gene_request(self, apiname, data, files=None, headers=None):
        response = requests.post(lfasr_host + apiname, data=data, files=files, headers=headers)
        result = json.loads(response.text)
        if result["ok"] == 0:
            print("{} success:".format(apiname) + str(result))
            return result
        else:
            print("{} error:".format(apiname) + str(result))
            exit(0)
            return result

    # 预处理
    def prepare_request(self):
        return self.gene_request(apiname=api_prepare,
                                 data=self.gene_params(api_prepare))

    # 上传
    def upload_request(self, taskid, upload_file_path):
        file_object = open(upload_file_path, 'rb')
        try:
            index = 1
            sig = SliceIdGenerator()
            while True:
                content = file_object.read(file_piece_sice)
                if not content or len(content) == 0:
                    break
                files = {
                    "filename": self.gene_params(api_upload).get("slice_id"),
                    "content": content
                }
                response = self.gene_request(api_upload,
                                             data=self.gene_params(api_upload, taskid=taskid,
                                                                   slice_id=sig.getNextSliceId()),
                                             files=files)
                if response.get('ok') != 0:
                    # 上传分片失败
                    print('upload slice fail, response: ' + str(response))
                    return False
                print('upload slice ' + str(index) + ' success')
                index += 1
        finally:
            'file index:' + str(file_object.tell())
            file_object.close()
        return True

    # 合并
    def merge_request(self, taskid):
        return self.gene_request(api_merge, data=self.gene_params(api_merge, taskid=taskid))

    # 获取进度
    def get_progress_request(self, taskid):
        return self.gene_request(api_get_progress, data=self.gene_params(api_get_progress, taskid=taskid))

    # 获取结果
    def get_result_request(self, taskid):
        return self.gene_request(api_get_result, data=self.gene_params(api_get_result, taskid=taskid))

    def all_api_request(self):
        # 1. 预处理
        pre_result = self.prepare_request()
        taskid = pre_result["data"]
        # 2 . 分片上传
        self.upload_request(taskid=taskid, upload_file_path=self.upload_file_path)
        # 3 . 文件合并
        self.merge_request(taskid=taskid)
        # 4 . 获取任务进度
        while True:
            # 每隔20秒获取一次任务进度
            progress = self.get_progress_request(taskid)
            progress_dic = progress
            if progress_dic['err_no'] != 0 and progress_dic['err_no'] != 26605:
                print('task error: ' + progress_dic['failed'])
                return
            else:
                data = progress_dic['data']
                task_status = json.loads(data)
                if task_status['status'] == 9:
                    print('task ' + taskid + ' finished')
                    break
                print('The task ' + taskid + ' is in processing, task status: ' + str(data))

            # 每次获取进度间隔20S
            time.sleep(20)
        # 5 . 获取结果
        return self.get_result_request(taskid=taskid)


# 注意：如果出现requests模块报错："NoneType" object has no attribute 'read', 请尝试将requests模块更新到2.20.0或以上版本(本demo测试版本为2.20.0)
# 输入讯飞开放平台的appid，secret_key和待转写的文件路径


# In[ ]:





# In[5]:


def w2v():

    model_word = Word2Vec.load(os.path.join(cwd, "tju_intern_api", "model", "Word2Vec.pkl"))
    voc_dim = 150 #word的向量维度
    min_out = 10 #单词出现次数
    window_size = 7 #WordVec中的滑动窗口大小
    #print model_word.wv.vocab.keys()[54],model_word.wv.vocab.keys()
    #print len(model_word.wv.vocab.keys())
    #print model_word ['有']
    input_dim = len(model_word.wv.vocab.keys()) + 1 #下标0空出来给不够10的字
    embedding_weights = np.zeros((input_dim, voc_dim)) 
    w2dic={}
    for i in range(len(model_word.wv.vocab.keys())):
        embedding_weights[i+1, :] = model_word [list(model_word.wv.vocab.keys())[i]]
        w2dic[list(model_word.wv.vocab.keys())[i]]=i+1
    #print embedding_weights
    return input_dim,embedding_weights,w2dic

def data2inx(w2indx,X_Vec):
    data = []
    for sentence in X_Vec:
        new_txt = []
        for word in sentence:
            try:
                new_txt.append(w2indx[word])
            except:
                new_txt.append(0)
        data.append(new_txt)
    return data 


# In[51]:
def getSentRes(sentence):
    input_dim, embedding_weights, w2dic = w2v()
    X_Vec = [[i for i in sentence]]
    index = data2inx(w2dic, X_Vec)
    index2 = tf.keras.preprocessing.sequence.pad_sequences(index, maxlen=150)
    wenzi_res = wenzi_model.predict(index2)
    return wenzi_res

def preprocessing(audio_file):
    cell = MFCC_extractor(audio_file)
    result = ASR(audio_file)

    # api = RequestApi(appid="5ed7af6d", secret_key="ea99466bc343c4b47e796199340a85a3", upload_file_path=audio_file)
    # result = api.all_api_request()['data']
    # result=re.search('"onebest":.*",',result).group()[11:-2]
    
    # padding一下数据
    for idx, item in enumerate(cell):
        if len(item) > 546:
            cell[idx] = cell[idx][:546]
    X_train_pad = tf.keras.preprocessing.sequence.pad_sequences(np.array([cell]), maxlen=546, value=0, padding='post')
    input_dim,embedding_weights,w2dic = w2v()
    X_Vec = [[i for i in result]]
    index = data2inx(w2dic,X_Vec)
    index2 = tf.keras.preprocessing.sequence.pad_sequences(index, maxlen=150 )
    
    return X_train_pad,index2, result


# In[52]:


def predictOnce(audio_file):
    yuyin, wenzi, wenben = preprocessing(audio_file)
    yuyin_res = yuyin_model.predict(yuyin)
    wenzi_res = wenzi_model.predict(wenzi)
    return yuyin_res, wenzi_res, wenben


# In[54]:


def output(yuyin_res, wenzi_res, wenben):
    yuyin_res = yuyin_res[0]
    wenzi_res = wenzi_res[0]
    print('该音频的情绪预测结果如下:')
    # {'angry': 0, 'fear': 1, 'happy': 2, 'neutral': 3, 'sad': 4, 'surprise': 5}
    dict1 = {0: '愤怒' , 1: '恐惧', 2: '高兴', 3: '中性', 4: '悲伤', 5: '惊讶'}
    maxIndex = np.argmax(yuyin_res)
    for idx, item in enumerate(yuyin_res):
        print( ('%s 情绪概率为 %f') %(dict1[idx], item))
    print( '其中 %s 情绪概率最高,达到了%f' % (dict1[maxIndex], yuyin_res[maxIndex]) )
    
    print('=' * 20)
    print( '该音频的文本分析结果如下: %s' % wenben)
    print('其情感倾向如下:')
    print( ('情感倾向指数为 %f') %(wenzi_res[0]))
    if wenzi_res[0] < 0.4:
        res = '消极'  
    elif wenzi_res[0] > 0.6:
        res = '积极'
    else:
        res = '中性'
    print('是%s情感' %(res))


# In[ ]:





# # 测试代码

# In[65]:

if __name__ == '__main__':
    audio_path = r'C:\Users\admin\Desktop\jupyter-workspace\shixun\project\casia中文语音情感数据集\casia\liuchanhg\sad\214.wav'
    # yuyin, wenzi = preprocessing(audio_path)


    # In[66]:


    y1, w1, wenben = predictOnce(audio_path)


    # In[67]:


    print(y1)
    print(w1)
    print(wenben)


    # In[68]:


    output(y1, w1, wenben)


# In[ ]:




