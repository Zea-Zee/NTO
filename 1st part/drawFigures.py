import numpy as np
import cv2
import math

def draw_and_save_image(file_name, shape, color):
    # Размер изображения
    width, height = 200, 200

    # Создаем изображение с прозрачным фоном
    image = np.zeros((height, width, 4), dtype=np.uint8)

    # Определение цветов в формате BGR (Blue, Green, Red, Alpha)
    colors = {
        'red': (0, 0, 255, 255),
        'green': (0, 255, 0, 255),
        'blue': (255, 0, 0, 255),
        'yellow': (0, 255, 255, 255),
        'white': (255, 255, 255, 255)
    }

    # Определение координат для различных фигур
    shapes = {
        'circle': np.array([[width//2, height//2, height//2]]),
        'rhombus': np.array([
            [100, 0],
            [25, 100],
            [100, 200],
            [175, 100]
        ]),
        'square': np.array([
            [0, 0],
            [0, height],
            [width, height],
            [width, 0]
        ]),
        'hexagon': np.array([
            [width//2 + int(width//2 * math.cos(2 * math.pi * i / 6)), height//2 + int(height//2 * math.sin(2 * math.pi * i / 6))]
            for i in range(6)
        ]),
        'pentagon': np.array([
            [width//2 + int(width//2 * math.cos(2 * math.pi * i / 5)), height//2 + int(height//2 * math.sin(2 * math.pi * i / 5))]
            for i in range(5)
        ])
    }

    # Рисование фигуры
    if shape == 'circle':
        cv2.circle(image, (shapes[shape][0][0], shapes[shape][0][1]), shapes[shape][0][2], color=colors[color],
                   thickness=-1)
    else:
        cv2.fillPoly(image, [shapes[shape]], color=colors[color])

    # Сохранение изображения в файл в формате PNG
    cv2.imwrite(file_name, image, [cv2.IMWRITE_PNG_COMPRESSION, 9])


# Создание и сохранение изображений
draw_and_save_image('yellow_circle.png', 'circle', 'yellow')
draw_and_save_image('white_circle.png', 'circle', 'white')
draw_and_save_image('yellow_rhombus.png', 'rhombus', 'yellow')
draw_and_save_image('blue_rhombus.png', 'rhombus', 'blue')
draw_and_save_image('red_square.png', 'square', 'red')
draw_and_save_image('green_square.png', 'square', 'green')
draw_and_save_image('blue_hexagon.png', 'hexagon', 'blue')
draw_and_save_image('green_pentagon.png', 'pentagon', 'green')
draw_and_save_image('red_pentagon.png', 'pentagon', 'red')


import numpy as np

# Заданные ограничения для координат x и y
x_limits = (0, 3.3)
y_limits = (0, 5.1)

# Генерация 9 рандомных пар точек
random_points = np.random.uniform(low=(x_limits[0], y_limits[0]), high=(x_limits[1], y_limits[1]), size=(9, 2))

# Вывод сгенерированных точек
print(random_points)
