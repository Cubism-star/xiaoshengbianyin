from django.http import HttpResponse
from django.shortcuts import render
import soundfile as sf
import numpy
import requests
def Home(request):
    return render(request,'index.html')

def pageone(request):
    return render(request,'pageone.html')

def pagetwo(request):
    File = request.FILES.get('soundfile', None)
    if File is None:
        return render(request,'pagetwo.html')
    with open(r"index/static/userImage/" + File.name, 'wb+') as f:
        for chunk in File.chunks():
            f.write(chunk)
    with open(r"index/static/userImage/" + File.name, 'rb') as f:
        data, sr = sf.read(f)
    # 时域x轴
    a = numpy.arange(0, len(data) / sr, 1 / sr)
    a = list(a)
    re = requests.get('http://holer.cc:50491/audioEmotion?audioName='+File.name)
    print(re.content.decode('utf8'))
    result = eval(re.content.decode('utf8'))
    dict1 = {0: '愤怒', 1: '恐惧', 2: '高兴', 3: '中性', 4: '悲伤', 5: '惊讶'}
    maxIndex = numpy.argmax(result['audioE'][0])
    emoAudio = dict1[maxIndex]
    wenzi_res = result['textE'][0]
    if wenzi_res[0] < 0.4:
        emoText = '消极'
    elif wenzi_res[0] > 0.6:
        emoText = '积极'
    else:
        emoText = '中性'
    return render(request,'pagetwo.html',{'file':File.name,'a':a,'signal':list(data),'content':result['context'],'textE':emoText,'audioE':emoAudio})

def pagethree(request):
    File = request.FILES.get('image', None)
    if File is None:
        return render(request, 'pagethree.html', {'image': "static/userImage/请上传图片.png"})
    with open(r"index/static/userImage/" + File.name, 'wb+') as f:
        for chunk in File.chunks():
            f.write(chunk)
    re = requests.get('http://holer.cc:50491/getImgRes?imgName=' + File.name)
    print(re.content.decode('utf8'))
    result = eval(re.content.decode('utf8'))
    wenzi_res = result['emotion'][0]
    if wenzi_res[0] < 0.4:
        emoText = '消极'
    elif wenzi_res[0] > 0.6:
        emoText = '积极'
    else:
        emoText = '中性'
    return render(request, 'pagethree.html', {'image': "static/userImage/"+File.name,'textChina': result['chinese'],'imageE':emoText})

def pagefour(request):
    File = request.FILES.get('soundfile', None)
    if File is None:
        return render(request,'pagefour.html')
    with open(r"index/static/userImage/" + File.name, 'wb+') as f:
        for chunk in File.chunks():
            f.write(chunk)
    with open(r"index/static/userImage/" + File.name, 'rb') as f:
        data, sr = sf.read(f)
    # 时域x轴
    a = numpy.arange(0, len(data) / sr, 1 / sr)
    a = list(a)
    re = requests.get('http://holer.cc:50491/evaluateVoice?audioName=' + File.name)
    result = eval(re.content.decode('utf8'))
    return render(request, 'pagefour.html',
                  {'file': File.name, 'a': a, 'signal': list(data), 'content': result['context'], 'score': str(int(float(result['score']))) + '分',
                   'percentage': str(int(pow(float(result['score']), 1 / 2) * 10)) + '%'})

def first(request):
    return render(request,'upload.html')

def upfile(request):
    File = request.FILES.get('soundfile', None)
    if File is None:
        return HttpResponse('No file uploaded')
    with open("" + File.name, 'wb+') as f:
        for chunk in File.chunks():
            f.write(chunk)
    with open("" + File.name, 'rb') as f:
        data, sr = sf.read(f)
    # 时域x轴
    a = numpy.arange(0, len(data) / sr, 1 / sr)
    a = list(a)
    return render(request,'upload.html',{'file':File.name,'a':a,'signal':list(data)})


def second(request):
    return render(request,'image.html')

def imageupfile(request):
    File = request.FILES.get('image', None)
    if File is None:
        return HttpResponse('No file uploaded')
    with open(r"D:/PycharmProjects/tju_intern/index/static/" + File.name, 'wb+') as f:
        for chunk in File.chunks():
            f.write(chunk)
    return render(request, 'image.html', {'image': "static/"+File.name})