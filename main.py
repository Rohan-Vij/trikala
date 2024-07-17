import requests
from PIL import Image
from io import BytesIO
from dotenv import load_dotenv
import os
import urllib.parse
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from fpdf import FPDF
import datetime

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GMAPS_API_KEY')

site = "Dublin High School"
address = "37.7199513,-121.9240798"

def get_address(address, map_type):
    encoded_address = urllib.parse.quote(address)
    url = f"https://maps.googleapis.com/maps/api/staticmap?center={encoded_address}&zoom=17&size=800x800&maptype={map_type}&key={API_KEY}"
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    image.show()
    return image

def closest_color(requested_color):
    basic_colors = {
        'black': (0, 0, 0),
        'white': (255, 255, 255),
        'red': (255, 0, 0),
        'lime': (0, 255, 0),
        'blue': (0, 0, 255),
        'yellow': (255, 255, 0),
        'cyan': (0, 255, 255),
        'magenta': (255, 0, 255),
        'silver': (192, 192, 192),
        'gray': (128, 128, 128),
        'maroon': (128, 0, 0),
        'olive': (128, 128, 0),
        'green': (0, 128, 0),
        'purple': (128, 0, 128),
        'teal': (0, 128, 128),
        'navy': (0, 0, 128),
    }
    min_distance = float('inf')
    closest_name = None
    for name, color in basic_colors.items():
        rd = (color[0] - requested_color[0]) ** 2
        gd = (color[1] - requested_color[1]) ** 2
        bd = (color[2] - requested_color[2]) ** 2
        distance = rd + gd + bd
        if distance < min_distance:
            min_distance = distance
            closest_name = name
    return closest_name

def segment_image(image, num_clusters=3):
    if image.mode != 'RGB':
        image = image.convert('RGB')
    image_np = np.array(image)
    pixels = image_np.reshape(-1, 3)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(pixels)
    labels = kmeans.labels_
    cluster_centers = kmeans.cluster_centers_
    segmented_img = np.zeros_like(pixels)
    for i in range(num_clusters):
        segmented_img[labels == i] = cluster_centers[i]
    segmented_img = segmented_img.reshape(image_np.shape).astype(np.uint8)
    
    highlight_colors = np.array([
        [255, 0, 0],    # Red
        [0, 255, 0],    # Green
        [0, 0, 255]     # Blue
    ])
    
    highlighted_img = np.zeros_like(pixels)
    for i in range(num_clusters):
        highlighted_img[labels == i] = highlight_colors[i]
    highlighted_img = highlighted_img.reshape(image_np.shape).astype(np.uint8)
    
    return segmented_img, highlighted_img, labels, cluster_centers, highlight_colors

def get_cluster_info(labels, cluster_centers, cluster_num):
    cluster_pixels = labels == cluster_num
    cluster_size = np.sum(cluster_pixels)
    total_size = labels.size
    cluster_percentage = (cluster_size / total_size) * 100
    avg_color = cluster_centers[cluster_num].astype(int)
    return {
        'size': cluster_size,
        'percentage': cluster_percentage,
        'average_color': avg_color,
        'average_color_name': closest_color(tuple(avg_color))
    }

def create_report(site_name, cluster_info, highlight_colors, image, segmented_img, highlighted_img):
    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)

    # ENGIE Branding
    pdf.set_text_color(0, 51, 102)  # ENGIE Blue
    # pdf.cell(200, 10, txt="ENGIE-TRIKALA Image Segmentation Report", ln=0, align='C')
    # pdf.cell(200, 80, txt=site_name, ln=True, align='C')

    pdf.multi_cell(200, 10, f"{site_name}\nENGIE-TRIKALA Image Segmentation Report", border = 0, 
                align = 'C', fill = False)
    
    pdf.set_text_color(0, 0, 0)  # Black

    # Add Images
    image.save('img/original_image.png')
    segmented_img = Image.fromarray(segmented_img)
    segmented_img.save('img/segmented_image.png')
    highlighted_img = Image.fromarray(highlighted_img)
    highlighted_img.save('img/highlighted_image.png')

    pdf.ln(20)
    pdf.cell(90, 10, txt="Original Image", ln=False, align='C')
    pdf.cell(90, 10, txt="Segmented Image", ln=True, align='C')
    pdf.image('img/original_image.png', x=10, y=40, w=90)
    pdf.image('img/segmented_image.png', x=110, y=40, w=90)
    pdf.ln(100)
    pdf.cell(90, 10, txt="Highlighted Segments", ln=True, align='C')
    pdf.image('img/highlighted_image.png', x=10, y=140, w=90)
    
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.set_text_color(0, 51, 102)
    pdf.cell(200, 10, txt="Cluster Information", ln=True, align='C')
    pdf.set_text_color(0, 0, 0)

    pdf.set_font("Arial", size=10)
    pdf.ln(10)
    
    pdf.cell(40, 10, 'Cluster Type', 1)
    pdf.cell(40, 10, 'Size (pixels)', 1)
    pdf.cell(40, 10, 'Percentage (%)', 1)
    pdf.cell(40, 10, 'Average Color', 1)
    pdf.cell(30, 10, 'Label Color', 1)
    pdf.ln(10)

    for label, info in cluster_info.items():
        highlight_color = highlight_colors[info['cluster_num']].astype(int)
        pdf.cell(40, 10, label.capitalize(), 1)
        pdf.cell(40, 10, str(info['size']), 1)
        pdf.cell(40, 10, f"{info['percentage']:.2f}", 1)
        pdf.cell(40, 10, info['average_color_name'], 1)
        pdf.cell(30, 10, closest_color(tuple(highlight_color)), 1)
        pdf.ln(10)

    # PDF
    os.makedirs('reports', exist_ok=True)
    pdf_output_path = f"reports/ENGIE-TRIKALA-{site_name.replace(' ', '-')}-{now}-report.pdf"
    pdf.output(pdf_output_path)
    print(f"Report saved to {pdf_output_path}")

image = get_address(address, "satellite")

segmented_img, highlighted_img, labels, cluster_centers, highlight_colors = segment_image(image, num_clusters=3)

plt.imshow(highlighted_img)
plt.title('Segmented Image with Highlighted Clusters')
plt.axis('off')
plt.show()

cluster_labels = ['road', 'campus', 'greenery']
cluster_info = {}

for i in range(3):
    print(f"Cluster {i}: Average color {cluster_centers[i].astype(int)} ({closest_color(tuple(cluster_centers[i].astype(int)))})")
    highlight_color = highlight_colors[i].astype(int)
    label = input(f"Which type of area does this cluster represent (road/campus/greenery)? Label color: {closest_color(tuple(highlight_color))} ")
    while label not in cluster_labels:
        print("Invalid input. Please enter 'road', 'campus', or 'greenery'.")
        label = input(f"Which type of area does this cluster represent (road/campus/greenery)? Label color: {closest_color(tuple(highlight_color))} ")
    cluster_labels.remove(label)
    info = get_cluster_info(labels, cluster_centers, i)
    info['cluster_num'] = i
    cluster_info[label] = info

create_report(site, cluster_info, highlight_colors, image, segmented_img, highlighted_img)
