import cv2
import numpy as np
import torch
from PIL import Image
from itertools import product
from torchvision.transforms import transforms
from torchvision.models import mobilenet_v2
import os
import pandas as pd

model_path = '/path_to_ColD_model_weight/ColD_model.pt'
input_folder = '/path_to_afm_images/data'
output_folder_name = os.path.basename(input_folder) + '_result'

num_splits = 10
num_pixels = 51

# num_splits = 4
# num_pixels = 128

# Load the model and state dictionary
model = mobilenet_v2(weights=None)
# Change the output size of the last fully connected layer to 2
model.classifier[1] = torch.nn.Linear(model.last_channel, 2)

state_dict = torch.load(model_path)
model.load_state_dict(state_dict)

model.eval()

output_folder_path = os.path.join(desktop_path, output_folder_name)
if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

image_paths = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

results_data = []

for image_path in image_paths:
    # Load the image
    image = Image.open(image_path)

    # Convert the image to grayscale
    gray_image = image.convert('L')

    # Split the image into 100 sub-images
    sub_images = []
    image_width, image_height = image.size
    sub_image_width, sub_image_height = image_width // num_splits, image_height // num_splits

    for i in range(num_splits):
        for j in range(num_splits):
            sub_image = image.crop((j * sub_image_width, i * sub_image_height,
                                    (j + 1) * sub_image_width, (i + 1) * sub_image_height))
            sub_images.append(sub_image)

    # Preprocess the sub-images
    preprocessed_sub_images = []
    transform = transforms.Compose([
        transforms.Resize((num_pixels, num_pixels)),  # Resize the sub-image to the size your model expects
        transforms.ToTensor()])

    for sub_image in sub_images:
        preprocessed_sub_image = transform(sub_image)
        preprocessed_sub_images.append(preprocessed_sub_image.unsqueeze(0))

    results = []

    with torch.no_grad():
        for sub_image in preprocessed_sub_images:
            output = model(sub_image)
            pred = output.argmax(dim=1).item()
            results.append(pred)

    # Convert the grayscale image to a NumPy array and set the color channel to 3
    gray_np_image = np.array(gray_image)
    color_image = cv2.cvtColor(gray_np_image, cv2.COLOR_GRAY2BGR)

    # Draw rectangles on the original image based on the model's predictions
    thickness = 2
    gap = 2

    for idx, (i, j) in enumerate(product(range(num_splits), range(num_splits))):
        x1, y1 = j * sub_image_width + gap // 2, i * sub_image_height + gap // 2
        x2, y2 = (j + 1) * sub_image_width - gap // 2, (i + 1) * sub_image_height - gap // 2
        color = (0, 255, 0) if results[idx] == 1 else (0, 0, 255)

        # Set the thickness value based on the color
        if color == (0, 255, 0):
            thickness = 4#4  # Set the thickness for the green box border
        else:
            thickness = 2#2  # Set the thickness for the red box border

        cv2.rectangle(color_image, (x1, y1), (x2, y2), color, thickness)

    # # Show the original image with rectangles
    # cv2.imshow('Results', color_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Save the result image to the new folder on the desktop
    image_name = os.path.basename(image_path)
    result_image_path = os.path.join(output_folder_path, image_name)
    cv2.imwrite(result_image_path, color_image)

    # Calculate the percentage of "Yes" and "No" predictions
    total_predictions = len(results)
    yes_count = results.count(1)
    no_count = results.count(0)
    yes_percentage = (yes_count / total_predictions) * 100
    no_percentage = (no_count / total_predictions) * 100

    # Save the results to a list
    results_data.append([image_name, yes_percentage, no_percentage])

# Create a DataFrame and save it as an Excel file
results_df = pd.DataFrame(results_data, columns=['Image Name', 'Yes (%)', 'No (%)'])
excel_file_path = os.path.join(output_folder_path, 'results.xlsx')
results_df.to_excel(excel_file_path, index=False)
