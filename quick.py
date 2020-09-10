import os

'''
    获取所有patch的信息，以便手动划分训练集和验证集
    算了别了，我直接隔10个抽一个算了
'''
from pinglib.files import get_file_list_recursive
from pinglib.toolkits.file_info_to_xls import file_info_to_xls

image_folder = r"D:\Projects\MARS-Stomach\Patches\slide"
image_path = get_file_list_recursive(image_folder)
file_info_to_xls(image_path, os.path.join(image_folder, 'patch_info.xlsx'))


image_folder = r"D:\Projects\MARS-Stomach\Patches\mask"
image_path = get_file_list_recursive(image_folder)
file_info_to_xls(image_path, os.path.join(image_folder, 'patch_mask_info.xlsx'))
