from skimage.metrics import peak_signal_noise_ratio as calculate_psnr
from skimage.metrics import structural_similarity as calculate_ssim
import numpy as np
from glob import glob
import cv2
from concurrent.futures import ProcessPoolExecutor


def calculate_psnr_ssim(img_dir, gtimg_dir):
    def get_psnr_ssim(img1_path, img2_path):
        im1, im2 = cv2.imread(img1_path), cv2.imread(img2_path)
        psnr = calculate_psnr(im1, im2)
        ssim = calculate_ssim(im1, im2, multichannel=True)
        return psnr, ssim
    imgpaths = glob(img_dir+'/*')
    gtpaths = glob(gtimg_dir+'/*')*5
    imgpaths.sort()
    gtpaths.sort()

    with ProcessPoolExecutor() as executor:
        results = list(tqdm(iterable=executor.map(get_psnr_ssim, input_images, gt_images), total=len(input_images), desc="Getting PSNR/SSIM"))

    psnr_list = [result[0] for result in results]
    ssim_list = [result[1] for result in results]

    print('Min_PSNR:',np.min(psnr_list),'Min_ssim:',np.min(ssim_list))
    print('PSNR:',np.mean(psnr_list),'SSIM:',np.mean(ssim_list))
    return np.mean(psnr_list),np.mean(ssim_list)