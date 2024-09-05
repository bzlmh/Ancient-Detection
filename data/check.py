import os
import xml.etree.ElementTree as ET
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


def draw_boxes(image_path, xml_path, output_folder):
    # Load the image
    image = Image.open(image_path)
    draw = ImageDraw.Draw(image)

    # Parse the XML file
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # Extract object details and draw bounding boxes
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)

        # Draw the bounding box
        draw.rectangle([(xmin, ymin), (xmax, ymax)], outline="red", width=3)

    # Save the image with bounding boxes
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_name = os.path.basename(image_path)
    output_path = os.path.join(output_folder, image_name)
    image.save(output_path)

    # Display the image
    plt.imshow(image)
    plt.axis('off')
    plt.show()


# Example usage
image_folder = 'Degard/train/img'
xml_folder = 'Degard/train/xml'

output_folder = 'Degard/train/withbox'

for filename in os.listdir(image_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):
        image_path = os.path.join(image_folder, filename)
        xml_path = os.path.join(xml_folder, os.path.splitext(filename)[0] + '.xml')

        if os.path.exists(xml_path):
            draw_boxes(image_path, xml_path, output_folder)
        else:
            print(f"XML file not found for {filename}")
