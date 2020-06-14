# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import wave
import base64
import hashlib

from imp import reload

import time

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/asrapi'
APP_KEY = '403dbd75ae568db2'
APP_SECRET = 'ebQK9yGLPnb1B65DobfLLn4qvAPsmZoX'

def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size-10:size]

def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()

def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)

def ASR(path):
    audio_file_path = path
    lang_type = 'zh-CHS'
    extension = audio_file_path[audio_file_path.rindex('.')+1:]
    if extension != 'wav':
        print('不支持的音频类型')
        sys.exit(1)
    wav_info = wave.open(audio_file_path, 'rb')
    sample_rate = wav_info.getframerate()
    nchannels = wav_info.getnchannels()
    wav_info.close()
    with open(audio_file_path, 'rb') as file_wav:
        q = base64.b64encode(file_wav.read()).decode('utf-8')

    data = {}
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['signType'] = "v2"
    data['langType'] = lang_type
    data['rate'] = sample_rate
    data['format'] = 'wav'
    data['channel'] = nchannels
    data['type'] = 1

    response = do_request(data)
    print(response.content.decode('utf8'))
    result = eval(response.content.decode('utf8').replace('false','False'))
    return result['result'][0]

if __name__ == '__main__':
    #arg为路径
    text = ASR('248.wav')
    print(text)