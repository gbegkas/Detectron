import os
import img2pdf

directory = '/mnt/Cargo_2/Sync/Πανεπιστημιο/Diploma Thesis/Python Scripts/vis-results'
images = []
for root, _, files in os.walk(directory):
    for file in files:
        if '.png.png' in file and '.pdf' not in file:
            images.append(os.path.join(root, file))

for image in images:
    print(image)
    with open(image[:-8] + '.pdf', 'wb') as fp:
        fp.write(img2pdf.convert(image))
