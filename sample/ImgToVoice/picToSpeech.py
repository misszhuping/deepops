from aip import AipSpeech
from PIL import Image
import pytesseract
import argparse
#import cv2
import os
from pytesseract import image_to_string

#img_file = "horse.png"

img_file = "spin.png"

#print(img_file)
img = Image.open(img_file)

words = image_to_string(img)

print(words)

""" 你的 APPID AK SK """
APP_ID = '20201738'
API_KEY = 'BE6r7j2oenTFo0YMT0caMEnR'
SECRET_KEY = 'heXv7yMARgkhBO1hcCXPMP5dlbObatvq'

client = AipSpeech(APP_ID, API_KEY, SECRET_KEY)

#words = "Early in the morning, the wind blew a spider across the field. A thin, silky thread trailed from her body. The spider landed on a fence post near a farm yard."
result  = client.synthesis(words, 'zh', 1, {
   'spd':4, 'vol': 5,'per':3
})

# 识别正确返回语音二进制 错误则返回dict 参照下面错误码
if not isinstance(result, dict):
    with open('auido_horse.mp3', 'wb') as f:
        f.write(result)