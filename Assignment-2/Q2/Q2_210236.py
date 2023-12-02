import cv2
import numpy as np


def custom_image_fusion(img1, img2, sf=100, rf=100, bf=0.9):
    smoothed_img1 = custom_bilateral_filter(img1.astype(np.float32), sf, rf)
    smoothed_img2 = custom_bilateral_filter(img2.astype(np.float32), sf, rf)

    # Compute the difference between the two smoothed images
    modulo = cv2.absdiff(smoothed_img1, smoothed_img2)

    # Apply fusion weights based on the difference
    distribution = 0.8 * (modulo > 20) + 0.2 * (modulo <= 20)

    # Combine the smoothed images using the computed weights
    fused_result = distribution * smoothed_img2 + (1 - distribution) * smoothed_img1

    # Clip and scale the final result based on brightness factor
    final_result = np.clip(fused_result * bf, 0, 255).astype(np.uint8)

    return final_result


def custom_bilateral_filter(image, spatial_deviation, range_deviation):
    rows, cols, channels = image.shape
    result = np.zeros_like(image, dtype=np.float32)

    window_size = 5  # Adjust the window size as needed
    half_window = window_size // 2

    for i in range(rows):
        for j in range(cols):
            i_start = max(0, i - half_window)
            i_end = min(rows, i + half_window + 1)
            j_start = max(0, j - half_window)
            j_end = min(cols, j + half_window + 1)

            spatial_weights = np.exp(
                -((np.arange(i_start, i_end)[:, None] - i) ** 2 +
                   (np.arange(j_start, j_end) - j) ** 2) / (2 * spatial_deviation ** 2))

            diff = image[i, j] - image[i_start:i_end, j_start:j_end]
            range_weights = np.exp(-np.sum(diff ** 2, axis=-1) / (2 * range_deviation ** 2))

            result[i, j] = np.sum(spatial_weights[..., None] * range_weights[..., None] * image[i_start:i_end, j_start:j_end], axis=(0, 1))

    result /= result.max()

    return (result * 255).astype(np.uint8)





def solution(image_path_a, image_path_b):
    ############################
    ############################
    ## image_path_a is path to the non-flash high ISO image
    ## image_path_b is path to the flash low ISO image
    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##

    # pass

    image = cv2.imread(image_path_b)

    flash_image = cv2.imread(image_path_b)
    noflash_image = cv2.imread(image_path_a)

    fused_image = custom_image_fusion(flash_image, noflash_image)

        
    

    return fused_image
