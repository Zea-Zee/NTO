import cv2
import numpy as np
import requests
from PIL import Image

# for 0.8 koef works correctly
# for 0.85 NOT
skip_koeff = 0.75
HEIGHT = 100
waitFlag = 1

def wait(w=0):
    if waitFlag or w:
        key = cv2.waitKey(0)
        if key == ord('q'):
            exit(0)
        cv2.destroyAllWindows()

def sh(window_name, img, w=1):
    if waitFlag or w == 2:
        cv2.imshow(window_name, img)
        if w > 0:
            wait(1)

def cut_corners(img):
    color_converted = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    im = Image.fromarray(color_converted)
    a = np.array(im)[:,:,:3]  # keep RGB only
    m = np.any(a != [255, 255, 255], axis=2)
    coords = np.argwhere(m)
    y0, x0, y1, x1 = *np.min(coords, axis=0), *np.max(coords, axis=0)
    im = im.crop((x0, y0, x1+1, y1+1))
    ret = np.array(im)
    ret = cv2.cvtColor(ret, cv2.COLOR_BGR2GRAY)
    return ret


def sort_by_text(lst, img):
    positions = []
    letter_images = lst.copy()
    for letter_image in letter_images:
        # cv2.imshow("let", letter_image)
        result = cv2.matchTemplate(img, letter_image, cv2.TM_CCOEFF_NORMED)
        h, w = letter_image.shape                                                       #there could be an error with overflow
        ih, iw = img.shape
        _, _, _, max_loc = cv2.minMaxLoc(result)
        # print(max_loc, w, h, max_loc[0] + w, max_loc[1] + h)
        cv2.rectangle(img, max_loc, (max_loc[0] + w, max_loc[1] + h), (0, 0, 255), -1)
        positions.append(max_loc)
        # cv2.imshow("img", img)
        # cv2.imshow("let", letter_image)
        #
        # print(lst)

    sorted_letter_images = [image for _, image in sorted(zip(positions, lst), key=lambda x: x[0])]
    for i in range(len(sorted_letter_images)):
        h, w = sorted_letter_images[i].shape
        sorted_letter_images[i] = letter = cv2.resize(sorted_letter_images[i], (int(w / h * HEIGHT) - 1, int(HEIGHT) - 1))
    return sorted_letter_images

def get_letters(templates):
    letters = []
    _, mask = cv2.threshold(templates, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letter = templates[y:y + h, x:x + w]
        # print(letter.shape)
        letters.append(letter)
    return letters

def get_letter(img, alphabet, max_vals):
# def get_letter(img, alphabet, used_width):
    biggest_val = float('-inf')
    biggest_index = 0
    biggest_pos = 0
    h, w = 0, 0
    for i in range(len(alphabet)):
        if(max_vals[i] < skip_koeff):
            continue
        # print(img.shape, template.shape)
        res = cv2.matchTemplate(img, alphabet[i], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(maxval)
        th, tw = alphabet[i].shape
        # if(i == 8):
        #     print(biggest_val, max_val, w, tw)
        #     sh("let", img, 0)
        #     sh("max let", alphabet[i])
        # if (max_val > biggest_val * 0.8 and tw > w) or (max_val > biggest_val * 1.2):
        # if tw < HEIGHT / 2 and tw > HEIGHT / 3:
        #     print(tw, HEIGHT, HEIGHT / tw)
        #     sh("narrow", alphabet[i], 2)
        max_vals[i] = max_val
        is1 = tw > HEIGHT / 2.8 and tw < HEIGHT / 2.7
        isI = tw > HEIGHT / 2.5 and tw < HEIGHT / 2.4
        isJ = tw > HEIGHT / 1.7 and tw < HEIGHT / 1.9
        if max_val > skip_koeff and max_val > biggest_val and (tw > HEIGHT / 2) or ((is1 or isI) and max_val > 0.9 and max_val > biggest_val):
            biggest_index = i
            # print(biggest_val, max_val, w, tw)
            biggest_val = max_val
            biggest_pos = max_loc
            h, w = alphabet[i].shape
            sh("let", img, 0)
            sh("max let", alphabet[i])

    if biggest_index == 0:
        biggest_index = 78
    elif biggest_index <= 13:
        biggest_index += 64
    elif biggest_index > 15 and biggest_index < 25:
        biggest_index += 66
    elif biggest_index > 13 and biggest_index < 25:
        biggest_index += 65
    else:
        biggest_index += 24
    if biggest_val < skip_koeff or w < HEIGHT / 3:
        return (-1, -1, max_vals)
        # return (-1, -1, used_width)
    # print(biggest_val, biggest_pos, biggest_index, w, h)
    # used_width += w
    cv2.rectangle(img, (biggest_pos[0] + int(w / 100), biggest_pos[1]), (biggest_pos[0] + w - int(w / 100), biggest_pos[1] + h), (0, 0, 255), -1)
    return (biggest_index, biggest_pos[0], max_vals)

# url = input()
# url = "https://stepik.org/media/attachments/course/187016/corrected_templ.png"
# image = np.frombuffer(bytes([int(i, 16) for i in input().split()]), np.uint8)
# fst = "https://ucarecdn.com/f628d67d-377e-4d1e-867c-607ebf161ecc/"
# nigger = "https://stepik.org/media/attachments/course/187016/text_recogn_test_1.png"
# Q_text = "https://stepik.org/media/attachments/course/187016/hpgQ.png"
# full_test = "https://stepik.org/media/attachments/course/187016/Full_test.png"
e_full_test = "https://stepik.org/media/attachments/course/187016/extended_Full_test.png"
# ee_full_test = "https://stepik.org/media/attachments/course/187016/eextended_Full_test.png"
url = e_full_test
resp = requests.get(url, stream=True, timeout=100.0).raw
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
oih, oiw = image.shape
image = cut_corners(image)
ih, iw = image.shape
image = cv2.resize(image, (int(iw / ih * HEIGHT), HEIGHT))
# sh("SDf", image)
# print(image.shape)
ih, iw = image.shape
# if(oiw / iw > 2):
#     exit(1)
# if(iw / ih > 25):
#     exit(1)
# if(ih < 50):
#     exit(10)
# image_contours = get_text_contours(image)

Q_url = "https://stepik.org/media/attachments/course/187016/Q.png"
resp = requests.get(Q_url, stream=True, timeout=100.0).raw
Q = np.asarray(bytearray(resp.read()), dtype="uint8")
Q = cv2.imdecode(Q, cv2.IMREAD_GRAYSCALE)
ret, Q = cv2.threshold(Q, 127, 255, cv2.THRESH_BINARY)
Q = cut_corners(Q)
Q_h, Q_w = Q.shape
Q = cv2.resize(Q, (int(Q_w / Q_h * HEIGHT) - 1, HEIGHT - 1))
Q_h, Q_w = Q.shape

cleared_q_num = 0
letters = []
Q_min_val, Q_max_val, Q_min_loc, Q_max_loc = cv2.minMaxLoc(cv2.matchTemplate(image, Q, cv2.TM_CCOEFF_NORMED))
ih, old_iw = image.shape
while Q_max_val > 0.8:
    cleared_q_num += 1
    # sh("before remove Q", image, 0)
    cv2.rectangle(image, (Q_max_loc[0] - 5, Q_max_loc[1]), (Q_max_loc[0] + Q_w + 5, Q_max_loc[1] + Q_h), (255, 255, 255), -1)
    # sh("after remove Q", image)
    letters.append(('Q', int(Q_max_loc[0])))
    Q_min_val, Q_max_val, Q_min_loc, Q_max_loc = cv2.minMaxLoc(cv2.matchTemplate(image, Q, cv2.TM_CCOEFF_NORMED))
# print(cleared_q_num)
image = cut_corners(image)
ih, iw = image.shape
image = cv2.resize(image, (int(iw / ih * HEIGHT), HEIGHT))
new_ih, new_iw = image.shape
# print(new_iw)
for i in range(len(letters)):
    letters[i] = (letters[i][0], int(letters[i][1] * new_iw / old_iw))
# sh("with no Q", image, 2)

mn = "https://stepik.org/media/attachments/course/187016/MN_templ.png"
templ = "https://stepik.org/media/attachments/course/187016/text_template.png"
templ_corrected_MN = "https://stepik.org/media/attachments/course/187016/corrected_templ.png"
with_no_q = "https://stepik.org/media/attachments/course/187016/alphabet_with_no_Q.png"
noq = "https://stepik.org/media/attachments/course/187016/literally_no_Q.png"

resp = requests.get(noq, stream=True, timeout=100.0).raw
template = np.asarray(bytearray(resp.read()), dtype="uint8")
template = cv2.imdecode(template, cv2.IMREAD_GRAYSCALE)
ret, template = cv2.threshold(template, 127, 255, cv2.THRESH_BINARY)
template = cut_corners(template)
templ_h, templ_w = template.shape
# template = cv2.resize(template, (int(templ_w / templ_h * HEIGHT), HEIGHT))
# print(template.shape)
# cv2.imshow("alphabet", template)
#
# print(template.shape)

alph_letters = get_letters(template.copy())
# cv2.imshow("letter A", alph_letters[0])

alph_txt = sort_by_text(alph_letters, template.copy())   #letters in alphabet order
alph_recs = [1 for _ in range(len(alph_txt))]
# for i in range(len(alph_txt)):
#     print(i)
#     cv2.imshow("Dfg", alph_txt[i])
#
# for el in alph_txt:
#     cv2.imshow("let", el)
#
#     cv2.destroyAllWindows()
while(True):
    # val, pos, used_width = get_letter(image, alph_txt, used_width)
    val, pos, alph_recs = get_letter(image, alph_txt, alph_recs)
    # if val == -1 or used_width > iw * 0.95:
    if val == -1:
        break
    letters.append((chr(val), pos))
    # print(chr(res))
# print(letters)
sorted_letters = sorted(letters, key=lambda x: x[1])
# for el in sorted_letters:
#     print("Chr", el[0], "at", el[1])
for el in sorted_letters:
    print(el[0], end='')
