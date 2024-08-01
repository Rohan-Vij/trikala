from flask import Flask, render_template, request, redirect, url_for, send_from_directory, flash
from main import ImageSegmentation
import os
from dotenv import load_dotenv
from PIL import Image

app = Flask(__name__)
app.secret_key = 'your_secret_key'

# Load environment variables
load_dotenv()
API_KEY = os.getenv('GMAPS_API_KEY')

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        site = request.form['site']
        address = request.form['address']
        num_clusters = int(request.form['num_clusters'])

        segmenter = ImageSegmentation(API_KEY, site, address)
        segmenter.get_address()
        segmenter.segment_image(num_clusters=num_clusters)

        # Save images to static/img folder
        segmenter.image.save('static/img/original_image.png')
        Image.fromarray(segmenter.segmented_img).save('static/img/segmented_image.png')
        Image.fromarray(segmenter.highlighted_img).save('static/img/highlighted_image.png')

        return redirect(url_for('results', site=site, address=address, num_clusters=num_clusters))

    return render_template('index.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    address = request.args.get('address')
    site = request.args.get('site')
    num_clusters = int(request.args.get('num_clusters'))
    highlight_colors = ['Red', 'Green', 'Blue']  # Default highlight colors

    if request.method == 'POST':
        segmenter = ImageSegmentation(API_KEY, site, address)  # recreate the segmenter to get cluster info
        segmenter.get_address()
        segmenter.segment_image(num_clusters=num_clusters)

        cluster_info = {}
        for i in range(num_clusters):
            label = request.form[f'label_{i}']
            info = segmenter.get_cluster_info(i)
            info['cluster_num'] = i
            cluster_info[label] = info

        segmenter.create_report(cluster_info)
        return redirect(url_for('report', site=site))

    return render_template('results.html', num_clusters=num_clusters, highlight_colors=highlight_colors)

@app.route('/report')
def report():
    site = request.args.get('site')
    report_path = f"ENGIE-TRIKALA-{site.replace(' ', '-')}-report.pdf"
    return render_template('report.html', report_path=report_path)

@app.route('/reports/<path:filename>', methods=['GET', 'POST'])
def download(filename):
    return send_from_directory(directory='reports', path=filename)

if __name__ == '__main__':
    app.run(debug=True)
