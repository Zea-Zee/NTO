import cv2
import numpy as np
import requests
from PIL import Image

HEIGHT = 500

def wait():
    key = cv2.waitKey(0)
    if key == ord('q'):
        exit(0)
    cv2.destroyAllWindows()

def cut_corners(img):
    coords = cv2.findNonZero(img)
    x, y, w, h = cv2.boundingRect(coords)
    cropped_image = img[y:y + h, x:x + w]
    return cropped_image

def match_letter_templater(letter, alphabet):
    biggest_index = 0
    biggest_val = 0
    lh, lw = letter.shape
    # letter = cv2.resize(letter, (int(lw / lh * HEIGHT), HEIGHT))

    for i in range(len(alphabet)):
        ah, aw = alphabet[i].shape
        # letter = cv2.resize(letter, (int(lw / lh * ah), ah))
        letter = cv2.resize(letter, (int(lw / lh * ah), ah))
        # alphabet[i] = cv2.resize(alphabet[i], (int(aw / ah * HEIGHT), HEIGHT))
        if(aw < lw or ah < lh):
            continue
        # alphabet[i] = cv2.resize(alphabet[i], (int(aw / ah * HEIGHT), HEIGHT))
        res = cv2.matchTemplate(alphabet[i], letter, cv2.TM_CCOEFF_NORMED)
        minval, maxval, minloc, maxloc = cv2.minMaxLoc(res)
        # print(maxval)
        if maxval > biggest_val:
            biggest_index = i
            biggest_val = maxval
    if biggest_index == 0:
        biggest_index = 78
    elif biggest_index <= 13:
        biggest_index += 64
    elif biggest_index > 13 and biggest_index < 26:
        biggest_index += 65
    else:
        biggest_index += 23
    return chr(biggest_index)


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
        # wait()
        # print(lst)

    sorted_letter_images = [image for _, image in sorted(zip(positions, lst), key=lambda x: x[0])]
    return sorted_letter_images
flag = False
def get_letters(templates):
    letters = []
    # th, tw = templates.shape
    # templates = cv2.resize(templates, (tw * 10, th * 10))
    # cv2.imshow("mask", mask)
    # wait()
    contours, _ = cv2.findContours(templates, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    merged_contours = []
    for contour in contours:
        shower = np.zeros_like(templates)
        cv2.drawContours(shower, contour, -1, (255, 0, 0), 4)
        cv2.imshow("G cont", shower)
        merge_append_flag = False
        if flag and len(merged_contours) > 0:
            for cont_i in range(len(merged_contours)):
                x1, y1, w1, h1 = cv2.boundingRect(contour)
                x2, y2, w2, h2 = cv2.boundingRect(merged_contours[cont_i])
                if (x1 < x2 and x1 + w1 > x2) or (x2 < x1 and x2 + w2 > x1):
                    # Прямоугольники перекрываются - объединить контуры
                    merged_contours[cont_i] = np.concatenate((contour, merged_contours[cont_i]))
                    cv2.drawContours(shower, merged_contours[cont_i], -1, (255, 0, 0), 4)
                    cv2.imshow("L cont", shower)
                    # wait()
                    merge_append_flag = True
                    break
        if not merge_append_flag:
            merged_contours.append(contour)
        print(len(merged_contours))
    for contour in merged_contours:
        x, y, w, h = cv2.boundingRect(contour)
        # print(w * h)
        # if flag:
        #     if (w * h < 500):
        #         continue
        letter = templates[y:y + h, x:x + w]
        h, w = letter.shape
        letters.append(letter)
        if flag:
            cv2.imshow("Let", letter)
            wait()
    print("there is", len(letters), "letters in word")
    return letters

mn = "https://stepik.org/media/attachments/course/187016/MN_templ.png"
templ = "https://stepik.org/media/attachments/course/187016/text_template.png"
templ_corrected_MN = "https://stepik.org/media/attachments/course/187016/corrected_templ.png"

resp = requests.get(templ_corrected_MN, stream=True, timeout=100.0).raw
template = np.asarray(bytearray(resp.read()), dtype="uint8")
template = cv2.imdecode(template, cv2.IMREAD_GRAYSCALE)
_, template = cv2.threshold(template, 230, 255, cv2.THRESH_BINARY_INV)
template = cut_corners(template)
# cv2.imshow("Alphabet", template)
th, tw = template.shape
# wait()
template = cv2.resize(template, (int(tw / th * HEIGHT), HEIGHT))
# th, tw = template.shape
# template_contours = get_text_contours(template)

url = "https://stepik.org/media/attachments/course/187016/text_recogn_test_1.png"#input()
# url = "https://ucarecdn.com/f628d67d-377e-4d1e-867c-607ebf161ecc/"
# url = "https://stepik.org/media/attachments/course/187016/corrected_templ.png"
resp = requests.get(url, stream=True, timeout=100.0).raw
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_GRAYSCALE)
_, image = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY_INV)
# cv2.imshow("before", image)
image = cut_corners(image)
# cv2.imshow("agter", image)
# wait()
# cv2.imshow("sdfs", image)
# wait()
ih, iw = image.shape
image = cv2.resize(image, (int(iw / ih * HEIGHT), HEIGHT))
cv2.imshow("sdfs", image)
wait()

alph_letters = get_letters(template.copy())
flag = True
txt_letters = get_letters(image.copy())

letter_txt = sort_by_text(txt_letters, image.copy())    #letters in order like in text
alph_txt = sort_by_text(alph_letters, template.copy())   #letters in alphabet order

# for i in range(len(alph_txt)):
#     print(i)
#     cv2.imshow("Dfg", alph_txt[i])
#     wait()
# for el in alph_txt:
#     cv2.imshow("let", el)
#     wait()
#     cv2.destroyAllWindows()
for l in letter_txt:
    res = match_letter_templater(l, alph_txt)
    print(res, end='')
# txt_len = len(txt_letters)
#
# text = []
# while(len(text) != len(txt_letters)):
#     for let in txt_letters:



# cv2.imshow("Img", image)
# cv2.imshow("Img contours", image_contours)
# cv2.wait(0)