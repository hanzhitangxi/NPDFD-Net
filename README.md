NPDFD-Net
=
This repository is an PyTorch implementation of the paper "Non-local Prior Dense Feature Distillation Network for Image Compressive Sensing"

Train
=
We trained the model using the DIV2K dataset, which can be found at https://data.vision.ee.ethz.ch/cvl/DIV2K/.
You can augment DIV2K with./dataset./generate_train.m and train with train.py.
Training the model can be found in https://pan.baidu.com/s/1kf33dNtbTZNbTBEMH-cDlQ?pwd=molk 

test
=
To convert images into .mat format, you can use the provided script generate_test_mat.m located in the ./dataset directory. 
After preparing the data, you can use test.py to perform testing.
