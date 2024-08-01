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

class ImageSegmentation:
    def __init__(self, api_key, site_name, address):
        self.api_key = api_key
        self.site_name = site_name
        self.address = address
        self.image = None
        self.segmented_img = None
        self.highlighted_img = None
        self.labels = None
        self.cluster_centers = None
        self.highlight_colors = None

    def get_address(self, map_type="satellite"):
        encoded_address = urllib.parse.quote(self.address)
        url = f"https://maps.googleapis.com/maps/api/staticmap?center={encoded_address}&zoom=17&size=800x800&maptype={map_type}&key={self.api_key}"
        response = requests.get(url)  # getting the image from google maps
        self.image = Image.open(BytesIO(response.content))  # open the image
        # self.image.show()

    def closest_color(self, requested_color):
        # basic colors for matching
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
            distance = rd + gd + bd  # calculating euclidean distance
            if distance < min_distance:
                min_distance = distance
                closest_name = name  # found a closer color
        return closest_name

    def segment_image(self, num_clusters=3):
        if self.image.mode != 'RGB':
            self.image = self.image.convert('RGB')  # ensure image is in rgb mode
        image_np = np.array(self.image)
        pixels = image_np.reshape(-1, 3)  # reshape for kmeans
        kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init='auto').fit(pixels)
        self.labels = kmeans.labels_
        self.cluster_centers = kmeans.cluster_centers_
        self.segmented_img = np.zeros_like(pixels)
        for i in range(num_clusters):
            self.segmented_img[self.labels == i] = self.cluster_centers[i]
        self.segmented_img = self.segmented_img.reshape(image_np.shape).astype(np.uint8)
        
        self.highlight_colors = np.array([
            [255, 0, 0],    # red
            [0, 255, 0],    # green
            [0, 0, 255]     # blue
        ])
        
        self.highlighted_img = np.zeros_like(pixels)
        for i in range(num_clusters):
            self.highlighted_img[self.labels == i] = self.highlight_colors[i]
        self.highlighted_img = self.highlighted_img.reshape(image_np.shape).astype(np.uint8)
    
    def get_cluster_info(self, cluster_num):
        cluster_pixels = self.labels == cluster_num
        cluster_size = np.sum(cluster_pixels)
        total_size = self.labels.size
        cluster_percentage = (cluster_size / total_size) * 100
        
        original_image_np = np.array(self.image)
        cluster_pixels_reshaped = cluster_pixels.reshape(original_image_np.shape[:2])  # reshape to match image dimensions
        avg_color = np.mean(original_image_np[cluster_pixels_reshaped], axis=0).astype(int)
        
        return {
            'size': cluster_size,
            'percentage': cluster_percentage,
            'average_color': avg_color,
            'average_color_name': self.closest_color(tuple(avg_color))
        }

    def create_report(self, cluster_info, identifier):
        now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)

        # ENGIE branding
        pdf.set_text_color(0, 51, 102)  # ENGIE blue
        pdf.multi_cell(200, 10, f"{self.site_name}\nENGIE-TRIKALA Image Segmentation Report", border=0, align='C', fill=False)
        pdf.set_text_color(0, 0, 0)  # black

        # Add images
        self.image.save('static/img/original_image.png')
        segmented_img_pil = Image.fromarray(self.segmented_img)
        segmented_img_pil.save('static/img/segmented_image.png')
        highlighted_img_pil = Image.fromarray(self.highlighted_img)
        highlighted_img_pil.save('static/img/highlighted_image.png')

        pdf.ln(20)
        pdf.cell(90, 10, txt="Original Image", ln=False, align='C')
        pdf.cell(90, 10, txt="Segmented Image", ln=True, align='C')
        pdf.image('static/img/original_image.png', x=10, y=40, w=90)
        pdf.image('static/img/segmented_image.png', x=110, y=40, w=90)
        pdf.ln(100)
        pdf.cell(90, 10, txt="Highlighted Segments", ln=False, align='C')
        # pdf.cell(90, 10, txt="Overlay Image", ln=True, align='C')
        pdf.image('static/img/highlighted_image.png', x=10, y=140, w=90)
        
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

        greenspace_coverage = 0
        brightness_values = []
        urban_areas = 0
        total_pixels = sum(info['size'] for info in cluster_info.values())

        for label, info in cluster_info.items():
            highlight_color = self.highlight_colors[info['cluster_num']].astype(int)
            pdf.cell(40, 10, label.capitalize(), 1)
            pdf.cell(40, 10, str(info['size']), 1)
            pdf.cell(40, 10, f"{info['percentage']:.2f}", 1)
            pdf.cell(40, 10, info['average_color_name'], 1)
            pdf.cell(30, 10, self.closest_color(tuple(highlight_color)), 1)
            pdf.ln(10)

            # Calculate greenspace coverage
            if label.lower() == 'greenery':
                greenspace_coverage += info['percentage']

            # Calculate brightness values
            avg_color = info['average_color']
            brightness = (0.299 * avg_color[0] + 0.587 * avg_color[1] + 0.114 * avg_color[2])
            brightness_values.append(brightness)

            # Calculate urban area coverage
            if label.lower() in ['road', 'campus']:
                urban_areas += info['size']

        avg_brightness = np.mean(brightness_values)
        urban_area_percentage = (urban_areas / total_pixels) * 100

        # Urban Heat Island Effect Analysis
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(200, 10, txt="Urban Heat Island Effect Analysis", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=10)
        pdf.ln(10)
        pdf.multi_cell(0, 10, "The urban heat island effect is a phenomenon where urban areas experience higher temperatures than their rural surroundings. This effect is mainly due to human activities. The segmentation results indicate that areas classified as 'road' and 'campus' could contribute to higher heat retention. The presence of 'greenery' helps mitigate this effect by providing cooling through evapotranspiration.")
        pdf.ln(5)
        pdf.cell(40, 10, f"Urban Area Coverage: {urban_area_percentage:.2f}%", ln=True)

        # Greenspace Coverage Analysis
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(200, 10, txt="Greenspace Coverage Analysis", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=10)
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"Greenspace coverage is essential for maintaining ecological balance and providing recreational areas for urban residents. The analysis shows that the greenspace coverage is {greenspace_coverage:.2f}%. Increasing greenspace can provide significant environmental benefits, including temperature regulation and improved air quality.")
        
        if greenspace_coverage < 20:
            pdf.ln(5)
            pdf.multi_cell(0, 10, "Warning: The greenspace coverage is below the recommended threshold for urban areas. Consider implementing strategies to increase greenspace.")

        # Solar Reflectance Index (SRI) Analysis
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(200, 10, txt="Solar Reflectance Index (SRI) Analysis", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=10)
        pdf.ln(10)
        pdf.multi_cell(0, 10, f"The Solar Reflectance Index (SRI) measures the ability of a surface to reflect solar heat. A higher SRI indicates a cooler surface. The average brightness value for the segmented areas is {avg_brightness:.2f}. Surfaces with higher brightness values contribute to lower surface temperatures, helping to reduce the urban heat island effect.")
        
        if avg_brightness < 128:
            pdf.ln(5)
            pdf.multi_cell(0, 10, "The average brightness is relatively low. Consider using materials with higher reflectance to reduce heat absorption.")

        # Detailed Cluster Analysis
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(200, 10, txt="Detailed Cluster Analysis", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=10)
        pdf.ln(10)
        for label, info in cluster_info.items():
            pdf.set_font("Arial", size=10, style='B')
            pdf.cell(200, 10, txt=f"{label.capitalize()} (Cluster {info['cluster_num'] + 1})", ln=True)
            pdf.set_font("Arial", size=10)
            pdf.cell(50, 10, txt=f"Size: {info['size']} pixels", ln=True)
            pdf.cell(50, 10, txt=f"Percentage: {info['percentage']:.2f}%", ln=True)
            pdf.cell(50, 10, txt=f"Average Color: {info['average_color_name']} ({info['average_color']})", ln=True)
            pdf.cell(50, 10, txt=f"Label Color: {self.closest_color(tuple(self.highlight_colors[info['cluster_num']].astype(int)))}", ln=True)
            pdf.ln(5)

            if label.lower() == 'road':
                pdf.multi_cell(0, 10, "Roads contribute to the urban heat island effect by absorbing heat. Consider using materials with higher reflectance for road surfaces.")
            elif label.lower() == 'campus':
                pdf.multi_cell(0, 10, "Campus areas typically include buildings and paved surfaces. Increasing tree coverage and green roofs can help mitigate heat absorption.")
            elif label.lower() == 'greenery':
                pdf.multi_cell(0, 10, "Greenery areas provide cooling effects through shade and evapotranspiration. Maintain and expand these areas to enhance environmental benefits.")
            pdf.ln(10)

        # Additional Insights
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.set_text_color(0, 51, 102)
        pdf.cell(200, 10, txt="Additional Insights", ln=True, align='C')
        pdf.set_text_color(0, 0, 0)
        pdf.set_font("Arial", size=10)
        pdf.ln(10)
        pdf.multi_cell(0, 10, "Further analysis can include evaluating the impact of different surface materials on environmental factors, analyzing the spatial distribution of greenspace, and assessing the potential for renewable energy installations based on surface reflectance properties.")

        # Save PDF
        os.makedirs('reports', exist_ok=True)
        pdf_output_path = f"reports/ENGIE-TRIKALA-{self.site_name.replace(' ', '-')}-{identifier}-report.pdf"
        pdf.output(pdf_output_path)
        print(f"Report saved to {pdf_output_path}")



if __name__ == "__main__":
    # load environment variables
    load_dotenv()
    API_KEY = os.getenv('GMAPS_API_KEY')

    site = "Dublin High School"
    address = "37.7199513,-121.9240798"

    segmenter = ImageSegmentation(API_KEY, site, address)
    segmenter.get_address()

    segmenter.segment_image(num_clusters=3)

    plt.imshow(segmenter.highlighted_img)
    plt.title('Segmented Image with Highlighted Clusters')
    plt.axis('off')
    plt.show()

    cluster_labels = ['road', 'campus', 'greenery']
    cluster_info = {}

    for i in range(3):
        print(f"\nCluster {i+1}: Average color {segmenter.cluster_centers[i].astype(int)} ({segmenter.closest_color(tuple(segmenter.cluster_centers[i].astype(int)))})")
        highlight_color = segmenter.highlight_colors[i].astype(int)
        label = input(f"Which type of area does this cluster represent (road/campus/greenery)? Label color: {segmenter.closest_color(tuple(highlight_color))}\nArea: ")
        while label not in cluster_labels:
            print("Invalid input. Please enter 'road', 'campus', or 'greenery'.")
            label = input(f"Which type of area does this cluster represent (road/campus/greenery)? Label color: {segmenter.closest_color(tuple(highlight_color))}\nArea: ")
        cluster_labels.remove(label)
        info = segmenter.get_cluster_info(i)
        info['cluster_num'] = i
        cluster_info[label] = info

    segmenter.create_report(cluster_info)
