import cv2
import numpy as np
import librosa

def solution(audio_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    y, sr = librosa.load(audio_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    spectrogram = 255 * np.clip(spectrogram, 0, np.max(spectrogram))
    spectrogram = spectrogram.astype(np.uint8)
    image = cv2.cvtColor(spectrogram, cv2.COLOR_GRAY2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sum_intensity = 0
    total_pixels = 0

    for row in gray:
        for pixel in row:
            sum_intensity += pixel
            total_pixels += 1   

    mean_intensity = sum_intensity / total_pixels
    
   
    if mean_intensity > 25 :
       class_name = 'metal'
    else:   
       class_name = 'cardboard'   
    
    return class_name
