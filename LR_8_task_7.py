import numpy as np
import cv2

img = cv2.imread('coins_2.JPG')
cv2.imshow('coins', img)

##########

shifted = cv2.pyrMeanShiftFiltering(img, 20, 50)
gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

##########

# Видалення шуму
kernel = np.ones((3, 3), np.uint8)
opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=3)
# Певна фонова область
sure_bg = cv2.dilate(opening, kernel, iterations=3)
# Пошук певної області переднього плану
dist_transform = cv2.distanceTransform(sure_bg, cv2.DIST_L2, 5)
# Область переднього плану
ret, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
# Пошук невідомої області
sure_fg = np.uint8(sure_fg)
unknown = cv2.subtract(sure_bg, sure_fg)

##########

# Маркування міток
ret, markers = cv2.connectedComponents(sure_fg)
# Додаємо один до всіх міток, щоб визначений фон був не 0, а 1
markers += 1
# Позначаємо область невідомого нулем
markers[unknown == 255] = 0

##########

# Алгоритм вододілу
markers = cv2.watershed(img, markers)
# Виділяємо кожну монету із markers різним кольором
for label in range(2, ret + 1):  # Починаємо з 2, оскільки фон має значення 1
    img[markers == label] = np.random.randint(0, 255, 3)

##########

cv2.imshow("coins_markers", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
