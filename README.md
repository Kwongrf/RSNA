# RSNA
This model is for [RSNA Pneumonia Detection Challenge](https://www.kaggle.com/c/rsna-pneumonia-detection-challenge) based on Mask-RCNN and pre-trained on COCO

model_train_test.py includes train phase and test phase, it takes about 8 hours to train on 1 GPU( NVIDIA 1080Ti)

How to run?
1. 'bash requirement.sh' #This will download Mask-RCNN and mask_rcnn_coco.h5
2. 'python3 model_train_test.py' #Please be patient. You can also use 'nohup python3 model_train_test.py &' to run it backend.

