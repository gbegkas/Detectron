import os, PythonMagick
from PythonMagick import Image
from datetime import datetime

start_time = datetime.now()

pdf_dir = "/run/media/gru/Storage/Thesis-Latex/figures/vis-results"
bg_colour = "#ffffff"

for root, _, pdfs in os.walk(pdf_dir):
    for pdf in pdfs:
        if '.pdf' in pdf:
            input_pdf = os.path.join(root, pdf)
            print(input_pdf)
            img = Image()
            # img.density('300')
            img.read(input_pdf)

            size = "%sx%s" % (img.columns(), img.rows())

            output_img = Image(size, bg_colour)
            output_img.type = img.type
            output_img.composite(img, 0, 0, PythonMagick.CompositeOperator.SrcOverCompositeOp)
            output_img.resize(str(800))
            output_img.magick('PNG')
            output_img.quality(75)

            output_jpg = input_pdf.replace(".pdf", ".png")
            output_img.write(output_jpg)

print(datetime.now() - start_time)