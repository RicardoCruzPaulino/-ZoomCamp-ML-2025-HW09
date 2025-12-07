from PIL import Image
import numpy as np
img = Image.open('/workspaces/-ZoomCamp-ML-2025-HW09/worshop-serveless/yf_dokzqy3vcritme8ggnzqlvwa.jpeg').convert('RGB')
img = img.resize((200,200))
arr = np.array(img).astype(np.float32) / 255.0
print('arr.shape', arr.shape)
print('first pixel RGB (H,W,3):', arr[0,0])
print('R channel value (arr[0,0,0]):', arr[0,0,0])
# show after transpose
arr_chw = arr.transpose(2,0,1)
print('arr_chw.shape', arr_chw.shape)
print('R channel first pixel (arr_chw[0,0,0]):', arr_chw[0,0,0])