import cv2
import numpy as np
import requests

def cut_corners(img):
    return img
    cv2.imshow("o", img)
    cv2.waitKey(0)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    cv2.imshow("g", gray)
    cv2.waitKey(0)
    ret, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
    cv2.imshow("t", thresh)
    cv2.waitKey(0)
    coords = cv2.findNonZero(thresh)
    x, y, w, h = cv2.boundingRect(coords)
    print(img.shape)
    print(x, y, w, h)
    cropped_image = img[y:y + h, x:x + w]
    cv2.imshow("n", cropped_image)
    cv2.waitKey(0)
    return cropped_image

sing_links = {
    "Главная дорога": "https://stepik.org/media/attachments/course/187016/Главная_дорога_7155.png",
    "Стоянка запрещена": "https://stepik.org/media/attachments/course/187016/Стоянка_запрещена_6465.png",
    "Крутой подъем": "https://stepik.org/media/attachments/course/187016/Крутой_подъем_9512.png",
    "Подача звукового сигнала запрещена": "https://stepik.org/media/attachments/course/187016/Подача_звукового_сигнала_запрещена_6305.png",
    "Ограничение максимальной скорости": "https://stepik.org/media/attachments/course/187016/Ограничение_максимальной_скорости_2359.png",
    "Дети": "https://stepik.org/media/attachments/course/187016/Дети_1192.png",
    "Дорожные работы": "https://stepik.org/media/attachments/course/187016/Дорожные_работы_9507.jpg",
    "Железнодорожный переезд без шлагбаума": "https://stepik.org/media/attachments/course/187016/Железнодорожный_переезд_без_шлагбаума_3857.png",
    "Пешеходный переход": "https://stepik.org/media/attachments/course/187016/Пешеходеый_переход_3561__1_.jpg",
    "Велосипедная дорожка": "https://stepik.org/media/attachments/course/187016/Велосипедная_дорожка_1811.jpg"
}

inp_img = np.frombuffer(bytes([int(i, 16) for i in input().split()]), np.uint8)
sign = cv2.imdecode(inp_img, cv2.IMREAD_COLOR)
sign = cut_corners(sign)
h, w, c = sign.shape
# print("hwc:", h, w, c)
# cv2.imshow("sign", sign)
# cv2.waitKey(0)

signs = {}

for key in sing_links.keys():
    resp = requests.get(sing_links[key], stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    image = cut_corners(image)
    # image = cv2.resize(image, (int(w / r), (int(h / r))))
    ho, wo, co = image.shape
    # print("hwc:", h, w, c)
    # cv2.imshow("o", image)
    # cv2.waitKey(0)
    resized_image = cv2.resize(image, (w, h))
    # cv2.imshow("n", resized_image)
    # cv2.waitKey(0)
    hn, wn, cn = resized_image.shape
    # print("hwco:", ho, wo, co)
    # print("hwcn:", hn, wn, cn)
    signs[key] = resized_image
    # cv2.imshow("tst", signs[key])
    # cv2.waitKey(0)

def matchTemplater(img, tmp):
    # cv2.imshow("sign", img)
    # cv2.imshow("tmp", tmp)
    # cv2.waitKey(0)
    result = cv2.matchTemplate(img, tmp, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    # print(max_val)
    return max_val


same_sign = "Знак неизвестен"
same_sign_val = float('-inf')

for key in signs.keys():
    # print(BFMatching(sign, signs[key]))
    res = matchTemplater(sign, signs[key])
    if res > 0.4 and res > same_sign_val:
        same_sign_val = res
        same_sign = key

print(same_sign)