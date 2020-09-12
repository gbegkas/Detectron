import os
import cv2

directory = '/mnt/Cargo_2/Sync/Πανεπιστημιο/Diploma Thesis/Python Scripts/vis-results'

masses = []
images = []
for root, _, files in os.walk(directory):
    for file in files:
        if 'mass' in file:
            masses.append(os.path.join(root, file))
        else:
            images.append(os.path.join(root, file))

for image in images:
    print(image)
    img = cv2.imread(image)
    for mass in masses:
        if os.path.basename(image.split('.')[0]) in os.path.basename(mass):
            m = cv2.imread(mass, 0)
            break

    x, y, w, h = cv2.boundingRect(m)
    # name = img + '.pdf'
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 5)
    cv2.imwrite(image, img)