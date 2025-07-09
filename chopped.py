import cv2
import preprocess as pre
import numpy as np

# Pad the image to a square canvas with white background and resize to 32x32
def pad_to_square(img):
    h, w = img.shape[:2]
    side = int(max(h, w) * 1.2)
    top = (side - h) // 2
    bottom = side - h - top
    left = (side - w) // 2
    right = side - w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=255)
    return cv2.resize(img, (32, 32))

# Remove small components by keeping the largest contour
def remove_small_components(img):
    inv = cv2.bitwise_not(img)
    contours, _ = cv2.findContours(inv, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    max_area = 0
    max_cnt = None
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h < max_area:
            img[y:y + h, x:x + w] = 255
        else:
            if (max_cnt != None):
                x0, y0, w0, h0 = max_cnt
                img[y0:y0 + h0, x0:x0 + w0] = 255
            max_cnt = (x, y, w, h)
            max_area = w * h
    return img

# Check if background is black based on border pixel intensity
def is_black_background(img):
    h, w = img.shape
    side = [np.mean(img[0,:w]), np.mean(img[h-1,:w]), np.mean(img[:h, 0]), np.mean(img[:h, w-1])]
    return np.mean(side) < 128

# Slice characters from license plate image and save to ./chopped
def view_chopped_char(address):
    img = cv2.imread(address)
    gray_img = pre.gray(img)
    candidates = pre.identify_candidate(gray_img)
    candidates = pre.identify_plate(candidates)
    i = 0
    for (x, y, w, h) in candidates:
        cropped = gray_img[y:y+h, x:x+w]
        # turn the backgroup to white
        if is_black_background(cropped):
            cropped = cv2.bitwise_not(cropped)
        cropped = remove_small_components(cropped)
        cropped = pad_to_square(cropped)
        filename = f"./chopped/{i}.jpg"
        cv2.imwrite(filename, cropped)
        i += 1