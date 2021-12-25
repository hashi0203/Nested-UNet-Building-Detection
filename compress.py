import os
import zipfile
import config

pred_dir = config.pred_dir
model_date = config.model_date
pred_dir = os.path.join(pred_dir, model_date)

zip_path = pred_dir + '.zip'

with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as new_zip:
    for n in range(100):
        img_name = '%d.png' % (370 + n)
        new_zip.write(os.path.join(pred_dir, img_name), arcname=img_name)