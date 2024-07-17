import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

load_dotenv()
API_KEY = os.getenv('GMAPS_API_KEY')

address = "37.7199513,-121.9240798"

def get_address(address, map_type):
    encoded_address = urllib.parse.quote(address)
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={encoded_address}&zoom=17&size=800x800&maptype={map_type}&key={API_KEY}"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image.show()
    return image

def segment_image(image, num_clusters=5):
    # Ensure the image is in RGB format
    if image.mode != 'RGB':
        image = image.convert('RGB')

    # Convert image to numpy array
    image_np = np.array(image)

    # Reshape the image to a 2D array of pixels
    pixels = image_np.reshape(-1, 3)

    # Apply KMeans to cluster the pixels
    kmeans = KMeans(n_clusters=num_clusters, random_state=0).fit(pixels)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_

    # Create an array to hold the segmented image
    segmented_img = np.zeros_like(pixels)

    # Assign each pixel to the color of its cluster center
    for i in range(num_clusters):
        segmented_img[labels == i] = cluster_centers[i]

    # Reshape back to the original image shape
    segmented_img = segmented_img.reshape(image_np.shape).astype(np.uint8)

    # Map each cluster to a distinct color for highlighting
    highlight_colors = np.array([
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255],    # Blue
        [255, 255, 0],  # Yellow
        [255, 0, 255]   # Magenta
    ])

    highlighted_img = np.zeros_like(pixels)

    # Assign each pixel to the highlight color of its cluster
    for i in range(num_clusters):
        highlighted_img[labels == i] = highlight_colors[i]

    # Reshape back to the original image shape
    highlighted_img = highlighted_img.reshape(image_np.shape).astype(np.uint8)

    # Display the original, segmented, and highlighted images
    fig, ax = plt.subplots(1, 3, figsize=(18, 6))
    ax[0].imshow(image)
    ax[0].set_title('Original Image')
    ax[0].axis('off')
    ax[1].imshow(segmented_img)
    ax[1].set_title('Segmented Image')
    ax[1].axis('off')
    ax[2].imshow(highlighted_img)
    ax[2].set_title('Highlighted Segments')
    ax[2].axis('off')
    plt.show()

    return segmented_img, highlighted_img

# Get the image from the address
image = get_address(address, "satellite")

# Segment the image and display results
segmented_img, highlighted_img = segment_image(image, num_clusters=3)
