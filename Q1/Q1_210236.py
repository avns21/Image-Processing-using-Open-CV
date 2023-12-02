import cv2
import numpy as np



def solution(image_path):
    

    image = cv2.imread(image_path)
    copy = image.copy()

    
    padding = (10, 10, 10, 10)  # Adjust the values as needed

    # Add padding to the image on all four sides
    image = cv2.copyMakeBorder(image, padding[2], padding[3], padding[0], padding[1], cv2.BORDER_CONSTANT,
                               value=(0, 0, 0))


    gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Threshold the image to convert non-black pixels to white
    
    _, binary_image = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

    binary_image_blurr = cv2.GaussianBlur(binary_image, (3,3), cv2.BORDER_DEFAULT)
   
    gray = binary_image_blurr.copy()
  
   
    corners = cv2.goodFeaturesToTrack(gray, 4, 0.02, 50)
    corners = np.int0(corners)

    # Create an empty list to store the corner coordinates
    corner_coordinates = []

    # we iterate through each corner,
    # making a circle at each point that we think is a corner.

    for i in corners:
        x, y = i.ravel()


        # Append the coordinates to the list
        corner_coordinates.append((x, y))
    
    # Assuming you have extracted corner_coordinates as mentioned in the previous code
    corner_coordinates = np.array(corner_coordinates)

    # Calculate the center of the points
    center = np.mean(corner_coordinates, axis=0)

    # Calculate the quadrant (top-left, top-right, bottom-left, bottom-right) for each corner
    quadrants = np.zeros(4, dtype=int)
    for i, coord in enumerate(corner_coordinates):
        if coord[0] < center[0] and coord[1] < center[1]:
            quadrants[i] = 0  # Top-left
        elif coord[0] >= center[0] and coord[1] < center[1]:
            quadrants[i] = 1  # Top-right
        elif coord[0] < center[0] and coord[1] >= center[1]:
            quadrants[i] = 2  # Bottom-left
        else:
            quadrants[i] = 3  # Bottom-right

    # Assign labels to the corners based on the quadrants

    top_left_idx = np.where(quadrants == 0)[0][0]
    top_right_idx = np.where(quadrants == 1)[0][0]
    bottom_left_idx = np.where(quadrants == 2)[0][0]
    bottom_right_idx = np.where(quadrants == 3)[0][0]

    top_left = corner_coordinates[top_left_idx]
    top_right = corner_coordinates[top_right_idx]
    bottom_left = corner_coordinates[bottom_left_idx]
    bottom_right = corner_coordinates[bottom_right_idx]



    corner_coordinates = corner_coordinates - 10
    top_left = corner_coordinates[top_left_idx]
    top_right = corner_coordinates[top_right_idx]
    bottom_left = corner_coordinates[bottom_left_idx]
    bottom_right = corner_coordinates[bottom_right_idx]


    trans1 = np.float32([top_left, top_right, bottom_left, bottom_right])
    # TL TR BL BR

    trans2 = np.float32([[0, 0], [600, 0], [0, 600], [600, 600]])
    M =    cv2.getPerspectiveTransform(trans1, trans2)
    output_image = cv2.warpPerspective(copy, M, (600, 600))

    return output_image