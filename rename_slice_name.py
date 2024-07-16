# import os

# def add_slice_name_to_images(path):
#     for folder_name in os.listdir(path):
#         folder_path = os.path.join(path, folder_name)
#         if os.path.isdir(folder_path):
#             slice_name = folder_name
#             file_path = os.path.join(folder_path,"he")
#             image_names = os.listdir(file_path)
#             for image_name in image_names:
#                 image_path = os.path.join(file_path, image_name)
#                 new_image_name = f"{slice_name}_{image_name}"  # 在原始图像名称前添加切片名称
#                 new_image_path = os.path.join(file_path, new_image_name)
#                 os.rename(image_path, new_image_path)


# add_slice_name_to_images("/root/projects/wu/Dataset/test_P63_FROZEN")


import os
from PIL import Image

def check_images(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                file_path = os.path.join(root, file)
                try:
                    image = Image.open(file_path)
                    image.verify()
                    # print(f"Image {file_path} is valid.")
                    image.close()
                except (IOError, SyntaxError) as e:
                    print(f"Image {file_path} is corrupted: {e}")

# 调用函数并传入文件夹路径
folder_path = '/root/projects/wu/Dataset/test_P63_FROZEN'
check_images(folder_path)