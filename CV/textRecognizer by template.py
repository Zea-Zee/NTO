import cv2
import numpy as np
import requests
from PIL import Image

HEIGHT = 66

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
    _, mask = cv2.threshold(templates, 150, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        letter = templates[y:y + h, x:x + w]
        letters.append(letter)

        template_contour = np.zeros_like(templates)
        cv2.drawContours(template_contour, contour, -1, (255, 0, 0), 4)
        # cv2.imshow("Cont", template_contour)
        # wait()
        cv2.imshow("letter", letter)
        wait()
    return letters

def get_letter(img, alphabet):
    biggest_val = float('-inf')
    biggest_index = 0
    biggest_pos = 0
    h, w = 0, 0
    for i in range(len(alphabet)):
        res = cv2.matchTemplate(img, alphabet[i], cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
        # print(maxval)
        if max_val > biggest_val:
            biggest_index = i
            biggest_val = max_val
            biggest_pos = max_loc
            h, w = alphabet[i].shape
            cv2.imshow("let", img)
            cv2.imshow("max let", alphabet[i])
            wait()
    if biggest_index == 0:
        biggest_index = 78
    elif biggest_index <= 13:
        biggest_index += 64
    elif biggest_index > 13 and biggest_index < 26:
        biggest_index += 65
    else:
        biggest_index += 23
    if biggest_val < -1:
        return -1
    print(biggest_val, biggest_pos, biggest_index, w, h)
    cv2.rectangle(img, biggest_pos, (biggest_pos[0] + w, biggest_pos[1] + h), (0, 0, 255), -1)
    return biggest_index


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
print(th)
# wait()
template = cv2.resize(template, (int(tw / th * HEIGHT), HEIGHT))
# th, tw = template.shape
# template_contours = get_text_contours(template)

# url = "https://stepik.org/media/attachments/course/187016/text_recogn_test_1.png"#input()
url = "https://ucarecdn.com/f628d67d-377e-4d1e-867c-607ebf161ecc/"
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
# cv2.imshow("sdfs", image)
# wait()

alph_letters = get_letters(template.copy())
cv2.imshow("letter A", alph_letters[0])
wait()
alph_txt = sort_by_text(alph_letters, template.copy())   #letters in alphabet order

# for i in range(len(alph_txt)):
#     print(i)
#     cv2.imshow("Dfg", alph_txt[i])
#     wait()
# for el in alph_txt:
#     cv2.imshow("let", el)
#     wait()
#     cv2.destroyAllWindows()
letters = [()]
while(True):
    res = get_letter(image, alph_letters)
    if res == -1:
        break
    letters.append(chr(res))
    print(chr(res))



# cv2.imshow("Img", image)
# cv2.imshow("Img contours", image_contours)
# cv2.wait(0)