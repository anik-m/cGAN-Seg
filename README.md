# cGAN-Seg

This repository hosts the training and evaluation code for our AMR project that uses the cGAN-Seg architecture, which utilizes a modified CycleGAN architecture to train cell segmentation and generation models effectively with a limited number of annotated cell images, addressing the challenge of scarce annotated data in microbial imaging. 
![Screenshot](Figure1.png)

### Datasets
We employed several segmentation datasets in our experiments, including AGAR and deepbacs dataset.

### Requirements

* Optional: Create a conda or Python virtual environment.

* Install the required packages using:
```
pip install -r requirements.txt
```

### Usage
#### Training the cGAN-Seg Model
Train the model with your dataset or use our provided cGAN-Seg dataset. Adjust hyperparameters, including the segmentation model type (DeepSea, CellPose, or UNET) and early stopping 'patience' value. During training, the script logs the training process and also saves the segmentation, generator, and discriminator checkpoints (Seg.pth, Gen.pth, D1.pth, and D2.pth) to the specified output_dir. 
```
Example:
python train.py --seg_model DeepSea --train_set_dir  ~/dataset/train/  --lr 0.0001 --p_vanilla 0.2 --p_diff 0.2 --patience 500 --output_dir output/
```

#### Testing the Segmentation Model
Evaluate your segmentation model using our cGAN-Seg dataset or yours, specifying the segmentation model type (seg_model) and its checkpoint directory (seg_ckpt_dir). This script calculates and returns segmentation metrics for the test set, including loss, Dice score, average F-score, precision, and recall. Additionally, it saves the predicted mask images to the specified output_dir.
```
Example:
python test_segmentation_model.py --seg_model DeepSea --test_set_dir ~/dataset/test/ --seg_ckpt_dir output/Seg.pth --output_dir output/segmentation/
```
#### Testing the Generator Model
Evaluate the StyleUNET generator's performance, using the synthetic or real mask images. The script calculates the FID metric and, if an output_dir is provided, generates and stores synthetic images there.
```
Example:
python test_generation_model.py --test_set_dir ~/dataset/test/ --gen_ckpt_dir output/Gen.pth --output_dir output/generation/
