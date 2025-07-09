import preprocess as pre
import chopped
import torch
import torch.nn as nn
import cv2
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Supported image formats
SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".webp")

# Character labels (excluding O to avoid confusion)
LABEL = list("0123456789ABCDEFGHIJKLMNPQRSTUVWXYZ")

# Lightweight CNN for character classification
class TinyCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(16 * 8 * 8, 64), nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# Read and recognize characters from a license plate image
def read_img(filename, test=False, output_dir = "./output"):
    init_img = cv2.imread(filename)

    if init_img is None:
        print(f"Failed to read {filename}")
        return []

    # Preprocess the image to find character candidates
    img = pre.gray(init_img)
    candidates = pre.identify_candidate(img)
    candidates = pre.identify_plate(candidates)

    # Load model and label encoder
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    label_encoder = LabelEncoder()
    label_encoder.fit(LABEL)

    model = TinyCNN(num_classes=len(LABEL)).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Preprocess each candidate region
    char_list = []
    for candidate in candidates:
        char_img = pre.chop(img, candidate)
        if char_img is None or char_img.size == 0:
            continue
        # Invert background if it's black
        if chopped.is_black_background(char_img):
            char_img = cv2.bitwise_not(char_img)

        # Clean and resize
        char_img = chopped.remove_small_components(char_img)
        char_img = chopped.pad_to_square(char_img)
        char_img = char_img.astype(np.float32) / 255.0
        char_img = np.expand_dims(char_img, axis=0)
        char_list.append(char_img)
    
    if not char_list:
        print("No characters detected.")
        return []
    
    # Predict batch
    batch_tensor = torch.tensor(np.stack(char_list)).to(device)

    with torch.no_grad():
        outputs = model(batch_tensor)
        pred_indices = outputs.argmax(dim=1).cpu().numpy()
        pred_chars = label_encoder.inverse_transform(pred_indices)

    # If in test mode, visualize predictions on image
    if test:
        font = cv2.FONT_HERSHEY_SIMPLEX
        thickness = 2
        color = (0, 255, 0) 
        output = init_img.copy()
        output = pre.draw_candidate(output, candidates)

        for i in range(len(pred_chars)):
            (x, y, w, h) = candidates[i]
            prediction = pred_chars[i]
            char_x = x
            char_y = y + 2 * h
            output = cv2.putText(output, prediction, (char_x, char_y), font, 1.2, color, thickness, cv2.LINE_AA)

        basename = os.path.basename(filename)
        output_path = os.path.join(output_dir, f"output_{basename}")
        cv2.imwrite(output_path, output)

    return pred_chars.tolist()

# Draw the entire license plate string under or above the image
def draw_plate_label(image, plate_text, position='bottom', font_scale=1.2):

    font = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    color = (0, 255, 0) 

    # Measure text size
    text_size, _ = cv2.getTextSize(plate_text, font, font_scale, thickness)
    text_width, text_height = text_size

    img_h, img_w = image.shape[:2]

    # Compute text position
    x = (img_w - text_width) // 2
    if position == 'bottom':
        y = img_h - 10
    else:
        y = text_height + 10
    
    cv2.putText(image, plate_text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)
    return image

# Entry point: read and predict all images in input folder
if __name__ == "__main__":
    input_folder = "./test_img"
    output_folder = "./output"
    model_path = "char_cnn.pth"
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(SUPPORTED_EXT):
            continue
        print("Processing:", f"{filename}")
        filepath = os.path.join(input_folder, filename)
        char_list = read_img(filepath, True)
        plate_str = "".join(char_list)
        print(f"{filename} -> result: {plate_str}")


