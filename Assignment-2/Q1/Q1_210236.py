import cv2
import numpy as np

# Usage

def apply_dilation_erosion(image, kernel, iterations=6):
    """
    Apply a sequence of dilation and erosion operations on the input image.

    Parameters:
    - image: The input image.
    - kernel: The structuring element for morphological operations.
    - iterations: Number of iterations for both dilation and erosion (default is 4).

    Returns:
    - processed_image: The image after the morphological operations.
    """
    processed_image = image.copy()  # Make a copy of the original image

    # Define the operations and their order
    operations = [(cv2.dilate, iterations), (cv2.erode, iterations),
                  (cv2.erode, iterations), (cv2.dilate, iterations),
                  ]

    # Apply the operations in the specified order
    for operation, iteration in operations:
        processed_image = operation(processed_image, kernel, iterations=iteration)

    return processed_image

def solution(image_path):
    image= cv2.imread(image_path)
    ######################################################################
    ######################################################################
    '''
    The pixel values of output should be 0 and 255 and not 0 and 1
    '''
    #####  WRITE YOUR CODE BELOW THIS LINE ###############################
    # image = cv2.GaussianBlur(image, (11, 11), 0)
    # hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # lower_lava = np.array([0, 100, 100])
    # upper_lava = np.array([20, 255, 255])
    # mask = cv2.inRange(hsv, lower_lava, upper_lava)
    # kernel = np.ones((5, 5), np.uint8)
    # mask = cv2.dilate(mask, kernel, iterations=4)
    # mask = cv2.erode(mask, kernel, iterations=4)
    # mask = cv2.erode(mask, kernel, iterations=4)
    # mask = cv2.dilate(mask, kernel, iterations=4)
    # mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    
    # cv2.imshow('frame', image)
    # cv2.imshow('mask', mask)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    
    image = cv2.GaussianBlur(image, (13, 13), 0)
    image_mod = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    threshold1 = np.array([0, 100, 100])  # Lower bound for reddish-orange tones
    threshold2 = np.array([60, 255, 255])  # Upper bound for reddish-orange tones

    lava_mask = cv2.inRange(image_mod, threshold1, threshold2)


    lava_contours, _ = cv2.findContours(lava_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lava_mask = np.zeros_like(lava_mask)

    for contour in lava_contours:
       cv2.drawContours(lava_mask, [contour], 0, 255, -1)


    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    lava_mask = apply_dilation_erosion(lava_mask, kernel) 
    lava_mask = cv2.cvtColor(lava_mask, cv2.COLOR_GRAY2BGR)
   
   #  cv2.imshow('lava_mask',lava_mask)
   #  cv2.waitKey(0)
   #  cv2.destroyAllWindows()   

    







    ######################################################################  
    return lava_mask
