import cv2
import numpy as np

def solution(image_path):
    ############################
    ############################

    ############################
    ############################
    ## comment the line below before submitting else your code wont be executed##
    # pass
    image = cv2.imread(image_path)
    #cv2.imshow(image)

    p_w = (20, 20, 20, 20)

# Add padding to the image on all four sides
    image = cv2.copyMakeBorder(image, p_w[2], p_w[3], p_w[0], p_w[1], cv2.BORDER_CONSTANT, value=(255, 255, 255))
    #cv2.imshow(image)

    gry_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gry_img = cv2.bitwise_not(gry_img)

    address = np.column_stack (np.where (gry_img > 0))
    theta = cv2.minAreaRect (address) [-1]

    #print(theta)

    height, width = image.shape [:2]
    centroid = (width / 2, height / 2)

    if(theta<45):
       theta = -theta+180

    elif(theta>45):
         theta = -theta+90

    RMX = cv2.getRotationMatrix2D (centroid, theta, 1.0)
    rotated_img = cv2.warpAffine(image, RMX, (width, height),flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    #cv2.imshow(rotated_img)

    (h, w) = rotated_img.shape[:2]

# Check if the image needs to be rotated
    if w < h:
       rotated_img = cv2.rotate(rotated_img, cv2.ROTATE_90_CLOCKWISE)

# Now, 'rotated_img' has its width greater than its height
    #cv2.imshow(rotated_img)
    finalimg = rotated_img.copy()

# Define the path where you want to save the image
    #output_path = 'saved_image.jpg'
    (h, w) = rotated_img.shape[:2]
# Save the image
    #cv2.imwrite(output_path, rotated_img)  # Use 'frame' if capturing from a webcam, or 'image' if loading an existing image

    def detect_orientation(image):
    # Load the image
        #image = cv2.imread(image_path)

    # Convert the image to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply edge detection to the grayscale image
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        #cv2.imshow(edges)

    # Detect lines in the edge image using the Hough Line Transform
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=110, minLineLength=110, maxLineGap=20)


        if lines is not None:
        # Create a copy of the original image to draw lines on
           image_w_l = image.copy()

           for line in lines:
               xx1, yy1, xx2, yy2 = line[0]

            # Draw the line on the image
               cv2.line(image_w_l, (xx1, yy1), (xx2, yy2), (0, 0, 255), 2)  # Red color for lines

        # Save or display the image with detected lines
           #cv2.imwrite('image_w_l.jpg', image_w_l)
           #cv2.imshow(image_with_lines)


        if lines is not None:
        # Calculate the midpoint of the image
           mid_height = h // 2
           #print(mid_height)

        # Calculate the average y-coordinate of the detected lines
           avg_y = np.mean([line[0][1] for line in lines])
          
        # If the average y-coordinate is closer to the top, it's upright; otherwise, it's upside down
           if avg_y < mid_height:
               return "Upright"
           else:
               return "Upside_Down"

        return "Orientation_not_detected"

# Example usage

    orientation = detect_orientation(rotated_img)
   
    if(orientation =="Upside_Down"):
       result = cv2.rotate(finalimg, cv2.ROTATE_180)
    else:
       result = finalimg

   

    return result
