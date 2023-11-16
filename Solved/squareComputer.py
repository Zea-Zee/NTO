import cv2
import numpy as np
import requests

url = input()
# url = url[:min(url.find(' '), url.find('\n') if url.find('\n') > 0 else url.find('\n') + len(url))]
# print(url, len(url), url.find('\n', 0), ":" + url + ":")
resp = requests.get(url, stream=True, timeout=100.0).raw
image = np.asarray(bytearray(resp.read()), dtype="uint8")
image = cv2.imdecode(image, cv2.IMREAD_COLOR)


hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_green = cv2.cvtColor(np.uint8([[[76, 177, 34]]]), cv2.COLOR_BGR2HSV)
hsv_red = cv2.cvtColor(np.uint8([[[36, 28, 237]]]), cv2.COLOR_BGR2HSV)
hsv_yellow = cv2.cvtColor(np.uint8([[[0, 242, 255]]]), cv2.COLOR_BGR2HSV)
hsv_blue = cv2.cvtColor(np.uint8([[[204, 72, 63]]]), cv2.COLOR_BGR2HSV)
hsv_magenta = cv2.cvtColor(np.uint8([[[164, 73, 163]]]), cv2.COLOR_BGR2HSV)

# print(hsv_green[0][0][0], 100, 100)

colors = {
    # "red": ([hsv_red[0][0][0] - 10 if hsv_red[0][0][0] - 10 >= 0 else hsv_red[0][0][0], 100, 100], [hsv_red[0][0][0] + 10, 255, 255]),
    "Желтый": ([hsv_yellow[0][0][0] - 10, 100, 100], [hsv_yellow[0][0][0] + 10, 255, 255]),
    # "green": ([31, 100, 100], [70, 255, 255]),
    "Зеленый": ([hsv_green[0][0][0] - 20, 100, 100], [hsv_green[0][0][0] + 20, 255, 255]),
    "Красный": ([hsv_red[0][0][0] - 20, 100, 100], [hsv_red[0][0][0] + 20, 255, 255]),
    "Синий": ([hsv_blue[0][0][0] - 20, 100, 100], [hsv_blue[0][0][0] + 20, 255, 255]),
    "Фиолетовый": ([hsv_magenta[0][0][0] - 10, 100, 100], [hsv_magenta[0][0][0] + 10, 255, 255]),
}

# print(colors["Красный"])

# dict with max area for every color
max_areas = {color: 0 for color in colors.keys()}

# Fill dict by areas
for color, (lower, upper) in colors.items():
    mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        max_areas[color] = cv2.contourArea(max_contour)

# Print res
for color, area in sorted(max_areas.items()):
    print(color + ": " + str(area))
