import cv2
import numpy as np
import requests

# ns = 1000
r = 6

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

signs = {}

for key in sing_links.keys():
    resp = requests.get(sing_links[key], stream=True).raw
    image = np.asarray(bytearray(resp.read()), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    h, w, c = image.shape
    image = cv2.resize(image, (int(w / r), (int(h / r))))
    signs[key] = image
    # cv2.imshow("tst", signs[key])
    # cv2.waitKey(0)

inp_img = np.frombuffer(bytes([int(i, 16) for i in input().split()]), np.uint8)
sign = cv2.imdecode(inp_img, cv2.IMREAD_COLOR)
# cv2.imshow("sign", sign)
# cv2.waitKey(0)

def BFMatching(img1, img2):
    feat = cv2.ORB_create(10)

    kp1, des1 = feat.detectAndCompute(img1, None)
    kp2, des2 = feat.detectAndCompute(img2, None)

    bf = cv2.BFMatcher()
    matches = bf.match(des1, des2)
    # good = []
    # matched_image = cv2.drawMatchesKnn(img1,kp1, img2, kp2, matches, None,matchColor=(0, 255, 0), matchesMask=None,singlePointColor=(255, 0, 0), flags=0)
    # for m, n in matches:
    #     if m.distance < 0.99 * n.distance:
    #         good.append([m])
    avg_dist = np.mean([match.distance for match in matches])

    # cv2.imshow("matches", matched_image)
    # cv2.waitKey(0)
    print(avg_dist)
    return avg_dist

same_sign = "Знак неизвестен"
same_sign_val = float('inf')

for key in signs.keys():
    # print(BFMatching(sign, signs[key]))
    res = BFMatching(sign, signs[key])
    if res < same_sign_val:
        same_sign_val = res
        same_sign = key

print(same_sign)