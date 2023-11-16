import numpy as np
import cv2

img = cv2.imread("https://stepik.org/media/attachments/course/184879/colors.png")
cv2.imshow("tst", img)

hsv_image = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Определение диапазонов HSV для каждого цвета
color_ranges = {
    "Красный": ([0, 100, 100], [10, 255, 255]),
    "Оранжевый": ([11, 100, 100], [20, 255, 255]),
    "Желтый": ([21, 100, 100], [30, 255, 255]),
    "Зеленый": ([31, 100, 100], [70, 255, 255]),
    "Голубой": ([71, 100, 100], [120, 255, 255]),
    "Синий": ([121, 100, 100], [170, 255, 255]),
    "Фиолетовый": ([171, 100, 100], [180, 255, 255])
}

# Инициализация словаря для хранения максимальной площади для каждого цвета
max_areas = {color: 0 for color in color_ranges.keys()}

# Поиск контуров для каждого цвета
for color, (lower, upper) in color_ranges.items():
    mask = cv2.inRange(hsv_image, np.array(lower), np.array(upper))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        max_contour = max(contours, key=cv2.contourArea)
        max_areas[color] = cv2.contourArea(max_contour)


# Вывод максимальных площадей в алфавитном порядке
for color, area in sorted(max_areas.items()):
    print(f"{color}: {area}")