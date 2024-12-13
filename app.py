import os
from flask import Flask, request, render_template, redirect, url_for, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.utils import save_image
from io import BytesIO 
import cv2
import numpy as np


# Flask app initialization
app = Flask(__name__)

# Device configuration
device = "cpu"

# Attention Mechanism
class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        batch_size, C, height, width = x.size()
        query = self.query_conv(x).view(batch_size, -1, height * width)
        key = self.key_conv(x).view(batch_size, -1, height * width)
        value = self.value_conv(x).view(batch_size, -1, height * width)

        energy = torch.bmm(query.permute(0, 2, 1), key)  # BxNqxNk
        attention = torch.softmax(energy, dim=-1)  # BxNq x Nk

        out = torch.bmm(value, attention.permute(0, 2, 1))  # BxNc x Nq
        out = out.view(batch_size, C, height, width)
        out = self.gamma * out + x
        return out

# Encoder and Decoder Blocks
class Encoder(nn.Module):
    def __init__(self, input_nc):
        super(Encoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_nc, 64, kernel_size=7, stride=1, padding=3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.encoder(x)


class Decoder(nn.Module):
    def __init__(self, output_nc):
        super(Decoder, self).__init__()
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, output_nc, kernel_size=7, stride=1, padding=3),
            nn.Tanh()
        )

    def forward(self, x):
        return self.decoder(x)

# Generator definition (combining Encoder, Attention, and Decoder)
class Generator(nn.Module):
    def __init__(self, input_nc, output_nc):
        super(Generator, self).__init__()
        self.encoder = Encoder(input_nc)
        self.attention = AttentionBlock(256)
        self.decoder = Decoder(output_nc)

    def forward(self, x):
        x = self.encoder(x)
        x = self.attention(x)
        x = self.decoder(x)
        return x


# Initialize the Generator
G1 = Generator(input_nc=1, output_nc=3).to(device)
G1.load_state_dict(torch.load(r"weights\best_G1.pth", map_location=torch.device('cpu')))
G1.eval()


# Image transformation pipeline
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
])

# Sliding Window function
def sliding_window_colorization(input_image, model, patch_size=64, overlap=32):
    img_width, img_height = input_image.size
    patches = []

    # Slide window over the image and process each patch
    for y in range(0, img_height - patch_size + 1, patch_size - overlap):  
        for x in range(0, img_width - patch_size + 1, patch_size - overlap):  
            patch = input_image.crop((x, y, x + patch_size, y + patch_size)) 
            patch = transform(patch).unsqueeze(0).to(device)
            
            # Generate colorized patch
            with torch.no_grad():
                colorized_patch = model(patch)

            patches.append((x, y, colorized_patch))  

    # Handle rightmost edge (if necessary)
    if img_width % patch_size != 0:
        x = img_width - patch_size
        for y in range(0, img_height - patch_size + 1, patch_size - overlap):  
            patch = input_image.crop((x, y, x + patch_size, y + patch_size)) 
            patch = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                colorized_patch = model(patch)
            patches.append((x, y, colorized_patch))  

    if img_height % patch_size != 0:
        y = img_height - patch_size
        for x in range(0, img_width - patch_size + 1, patch_size - overlap):  
            patch = input_image.crop((x, y, x + patch_size, y + patch_size)) 
            patch = transform(patch).unsqueeze(0).to(device)
            with torch.no_grad():
                colorized_patch = model(patch)
            patches.append((x, y, colorized_patch))  

    # Combine the patches into a single image
    output_image = torch.zeros((1, 3, img_height, img_width), device=device)
    count_map = torch.zeros((1, 3, img_height, img_width), device=device)

    for (x, y, colorized_patch) in patches:
        output_image[:, :, y:y + patch_size, x:x + patch_size] += colorized_patch
        count_map[:, :, y:y + patch_size, x:x + patch_size] += 1

    output_image /= count_map
    return output_image

# Helper function to calculate greenery rate
def calculate_greenery_rate(image_path):

    # Read the image
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image not found at path: {image_path}")

    # Convert the image from BGR to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Define the HSV range for detecting greenery
    lower_green = np.array([30, 40, 40])
    upper_green = np.array([90, 255, 255])


    # Create a binary mask where green colors are in range
    green_mask = cv2.inRange(hsv_image, lower_green, upper_green)

    # Calculate the number of green pixels
    green_pixel_count = np.sum(green_mask > 0)

    # Calculate the total number of pixels in the image
    total_pixel_count = image.shape[0] * image.shape[1]

    # Calculate the greenery rate
    greenery_rate = (green_pixel_count / total_pixel_count) * 100

    return greenery_rate

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    file = request.files['file']
    if file.filename == '':
        return "No selected file"
    if file:
        uploaded_image_path = os.path.join("static", "uploaded_image.png")
        file.save(uploaded_image_path)

        # Process the SAR image
        image = Image.open(uploaded_image_path).convert("L")  # Convert to grayscale

        # Apply sliding window colorization
        colorized_image = sliding_window_colorization(image, G1, patch_size=256, overlap=128)
        
        # Denormalize and save the colorized image
        colorized_image = colorized_image * 0.5 + 0.5  # Denormalize
        colorized_image_path = os.path.join("static", "colorized_image.png")
        save_image(colorized_image, colorized_image_path)

        # Calculate greenery rate
        greenery_rate = calculate_greenery_rate(colorized_image_path)

        return render_template(
            'result.html',
            uploaded_image_url="/static/uploaded_image.png",
            colorized_image_url="/static/colorized_image.png",
            greenery_rate=f"{greenery_rate:.2f}%"
        )

# Folder to save uploaded images
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/greenery', methods=['GET'])
def greenery_rate_page():
    return render_template('greenery.html')


@app.route('/greenery-upload', methods=['POST'])
def greenery_upload():
    if 'file' not in request.files:
        return redirect(request.url)
    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)
    if file:
        # Save the file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Dummy greenery rate calculation (replace with your logic)
        greenery_rate = calculate_greenery_rate(filepath)

        # Pass the image URL and greenery rate to the template
        uploaded_image_url = url_for('static', filename=f'uploads/{filename}')
        return render_template('greenery.html', uploaded_image_url=uploaded_image_url, greenery_rate=f"{greenery_rate:.2f}%")

# Predefined chatbot data
faq_data = [
    {
        "question": "What is SAR?",
        "keywords": ["sar", "what"],
        "response": "SAR stands for Synthetic Aperture Radar. It is a type of radar used to create two-dimensional images or three-dimensional reconstructions of objects.",
    },
    {
        "question": "Advantages of SAR",
        "keywords": ["advantages", "sar"],
        "response": "Advantages of SAR include all-weather imaging, ability to penetrate clouds and darkness, and high-resolution imaging over large areas.",
    },
    {
        "question": "Disadvantages of SAR",
        "keywords": ["disadvantages", "sar", "cons"],
        "response": "Disadvantages of SAR include high operational costs, susceptibility to speckle noise, and complexity in data interpretation.",
    },
    {
        "question": "Applications of SAR",
        "keywords": ["applications", "sar", "usage"],
        "response": "Applications of SAR include disaster management, environmental monitoring, urban planning, agriculture, and military surveillance.",
    },
    {
        "question": "SAR Image Colorization",
        "keywords": ["sar", "colorization"],
        "response": "SAR image colorization involves using algorithms or deep learning models to add colors to grayscale SAR images for better visual interpretation.",
    },
    {
        "question": "Limitations of SAR",
        "keywords": ["limitations", "sar"],
        "response": "Limitations of SAR include difficulty in interpretation due to speckle noise and high computational costs for processing.",
    },
    {
        "question": "Pros and Cons of SAR",
        "keywords": ["pros", "cons", "sar"],
        "response": "Pros of SAR include all-weather capability, cloud penetration, and high-resolution imaging. Cons include high costs, noise, and complexity in interpretation.",
    },
    {
        "question": "Components of a SAR system",
        "keywords": ["components", "sar", "system"],
        "response": "A SAR system typically consists of a radar antenna, transmitter, receiver, signal processor, and platform (airborne or satellite-based).",
    },
    {
        "question": "SAR for disaster management",
        "keywords": ["sar", "disaster", "management"],
        "response": "SAR is used in disaster management for monitoring floods, landslides, and earthquakes, providing timely information for relief and recovery operations.",
    },
]

@app.route('/chatbot')
def chatbot():
    return render_template('chatbot.html')

@app.route('/chatbot/ask', methods=['POST'])
def chatbot_ask():
    user_question = request.json.get('question', '').lower()

    # Scoring mechanism for relevance
    best_match = None
    max_score = 0

    for entry in faq_data:
        score = sum(keyword in user_question for keyword in entry["keywords"])
        if score > max_score:
            best_match = entry
            max_score = score

    # Return the best match response if found, otherwise a default message
    response = best_match["response"] if best_match else "Sorry, I couldn't find an answer to your question."
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
