import cv2
import os

def adaptive_binarization(input_folder, output_folder):
    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的所有文件
    for filename in os.listdir(input_folder):
        if filename.endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff')):  # 你可以根据需要添加更多图像格式
            # 读取图像
            img_path = os.path.join(input_folder, filename)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

            # 应用自适应二值化
            binary_img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

            # 保存二值化图像到输出文件夹，文件名保持不变
            output_path = os.path.join(output_folder, filename)
            cv2.imwrite(output_path, binary_img)

if __name__ == "__main__":
    input_folder = "../data/MTH1000/test/test_images/"   # 替换为你的输入文件夹路径
    output_folder = "../data/MTH1000/test/b_img/"        # 替换为你的输出文件夹路径
    adaptive_binarization(input_folder, output_folder)
