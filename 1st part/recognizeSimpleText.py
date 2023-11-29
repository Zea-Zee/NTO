import cv2
import numpy as np
import requests
from PIL import Image

url = "https://ucarecdn.com/f628d67d-377e-4d1e-867c-607ebf161ecc/"
resp = requests.get(url, stream=True).raw
img = np.asarray(bytearray(resp.read()), dtype='uint8')
img = cv2.imdecode(img, cv2.IMREAD_COLOR)


def cut_corners(img):
    color_converted = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    im = Image.fromarray(color_converted)
    a = np.array(im)[:,:,:3]  # keep RGB only
    m = np.any(a != [255, 255, 255], axis=2)
    coords = np.argwhere(m)
    y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
    im = im.crop((x0, y0, x1+1, y1+1))
    ret = np.array(im)
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    return ret

def get_letter(img, i):
    img_height, img_width, img_channels = img.shape
    let_height, let_width = img_height, int(img_width / 35)
    let = img[:let_height, let_width * i : let_width * (i + 1)]
    let = cv2.resize(let, (let_width * 5, let_height * 5))
    cv2.imshow("C", let)
    cv2.waitKey(0)
    return let


url = "https://stepik.org/media/attachments/course/187016/bebra.png"
resp = requests.get(url, stream=True).raw
templ = np.asarray(bytearray(resp.read()), dtype='uint8')
templ = cv2.imdecode(templ, cv2.IMREAD_COLOR)

img = cut_corners(img)
templ = cut_corners(templ)
# cv2.imshow("img", img)
# cv2.imshow("orig", templ)
# cv2.waitKey(0)

img_height, img_width, img_channels = img.shape
templ_height, templ_width, templ_channels = templ.shape
templ = cv2.resize(templ, (int(templ_width / templ_height * img_height), img_height))
cv2.imshow("resized", templ)
cv2.waitKey(0)

for i in range(0, 36):
    get_letter(templ, i)