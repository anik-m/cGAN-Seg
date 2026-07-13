# Import necessary libraries and packages
import os
import argparse
from data import BasicDataset, get_image_mask_pairs
import cv2
from unet_models import StyleUnetGenerator
from utils import set_requires_grad, mixed_list, noise_list, image_noise
import torch.utils.data as data
import transforms as transforms
import torch
import shutil
import numpy as np
import torch.nn as nn
import random
from tqdm import tqdm
from utils import calculate_fid

# Setting the seed for generating random numbers for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

def normalize_image(image):
    min_val = np.min(image)
    max_val = np.max(image)
    return (image - min_val) / (max_val - min_val + 1e-8)


# MODIFIED: Accepts a list of original filenames
def save_images(output_dir, real_img_list, fake_img_list, real_A_list, filename_list):
    for real_img, fake_img, real_A, orig_name in zip(real_img_list, fake_img_list, real_A_list, filename_list):
        real_img_save = (real_img * 255).astype(np.uint8)
        fake_img_save = (fake_img * 255).astype(np.uint8)
        real_A_save = (real_A * 255).astype(np.uint8)
        
        # MODIFIED: Uses the original filename instead of sequential numbering
        cv2.imwrite(os.path.join(output_dir, 'real_images', orig_name), real_img_save)
        cv2.imwrite(os.path.join(output_dir, 'gen_images', orig_name), fake_img_save)
        cv2.imwrite(os.path.join(output_dir, 'real_masks', orig_name), real_A_save)

def reshape_and_repeat(images_list, image_size, is_gray=True):
    num_images = len(images_list)
    if is_gray:
        images = np.array(images_list).reshape(num_images, 1, image_size[0], image_size[1])
        return np.repeat(images, 3, axis=1)
    else:
        return np.array(images_list).reshape(num_images, 3, image_size[0], image_size[1])

def test(args, image_size=[512, 768], image_means=[0.5], image_stds=[0.5], batch_size=1):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    test_transforms = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=image_means, std=image_stds)
    ])
    
    sample_pairs = get_image_mask_pairs(args.test_set_dir)
    assert len(sample_pairs) > 0, f'No samples found in {args.test_set_dir}'

    # ADDED: Extract and track the exact filenames from your sample pairs
    # Assumes get_image_mask_pairs returns tuples/lists where the first element is the image path
    filename_list = [os.path.basename(pair[0]) for pair in sample_pairs]

    test_data = BasicDataset(sample_pairs, transforms=test_transforms)
    test_iterator = data.DataLoader(test_data, batch_size=batch_size, shuffle=False)

    Gen = StyleUnetGenerator(style_latent_dim=128, output_nc=1).to(device)
    Gen.load_state_dict(torch.load(args.gen_ckpt_dir))
    Gen.eval()

    real_img_list = []
    real_A_list = []
    fake_img_list = []
    set_requires_grad(Gen, False)
    
    with torch.no_grad():
        for step, batch in enumerate(tqdm(test_iterator)):
            real_img = batch['image'].to(device=device, dtype=torch.float32)
            real_mask = batch['mask'].to(device=device, dtype=torch.float32)

            style = mixed_list(real_img.shape[0], 5, 128, device=device) if random.random() < 0.9 else noise_list(real_img.shape[0], 5, 128, device=device)
            im_noise = image_noise(real_mask.shape[0], image_size, device=device)

            fake_img = Gen(real_mask, style, im_noise)
            fake_img = normalize_image(fake_img.cpu().numpy()[0, 0, :, :])
            real_img = normalize_image(real_img.cpu().numpy()[0, 0, :, :])
            real_mask = normalize_image(real_mask.cpu().numpy()[0, 0, :, :])

            real_img_list.append(real_img)
            real_A_list.append(real_mask)
            fake_img_list.append(fake_img)

    if os.path.exists(os.path.join(args.test_set_dir, 'images')):
        # MODIFIED: Passing the filename list down to the save function
        save_images(args.output_dir, real_img_list, fake_img_list, real_A_list, filename_list)

        real_images = reshape_and_repeat(real_img_list, image_size, is_gray=True)
        generated_images = reshape_and_repeat(fake_img_list, image_size, is_gray=True)

        fid_score = calculate_fid(real_images, generated_images, device)
        print('FID Score: %f' % fid_score)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test_set_dir", required=True, type=str, help="path for the test dataset")
    ap.add_argument("--gen_ckpt_dir", required=True, type=str, help="path for the generator model checkpoint")
    ap.add_argument("--output_dir", required=True, type=str, help="path for saving the test outputs")

    args = ap.parse_args()

    assert os.path.isdir(args.test_set_dir), 'No such file or directory: ' + args.test_set_dir

    if os.path.exists(args.output_dir):
        shutil.rmtree(args.output_dir)

    os.makedirs(args.output_dir)
    os.makedirs(os.path.join(args.output_dir, 'real_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'gen_images'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'real_masks'), exist_ok=True)
    
    test(args)
