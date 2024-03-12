import os
import time
from pathlib import Path
import random
import shutil
from tqdm import tqdm
import xml.etree.ElementTree as ET
import pickle
from os import listdir, getcwd
from os.path import join


def png_match_txt(png_folder, txt_folder, output_folder):
    png_folder_path = Path(png_folder)
    txt_folder_path = Path(txt_folder)
    output_folder_path = Path(output_folder)
    
    # 创建输出文件夹
    output_folder_path.mkdir(parents=True, exist_ok=True)
    
    # 获取 PNG 文件夹中的所有文件名（不包括后缀）
    png_filenames = set(file.stem for file in png_folder_path.glob('*'))
    # 遍历 TXT 文件夹，复制与 PNG 文件夹中的文件名匹配的 TXT 文件到输出文件夹
    for txt_file in tqdm(txt_folder_path.glob('*.txt')):
        txt_filename = txt_file.stem
        if txt_filename in png_filenames:
            shutil.copy(txt_file, output_folder_path)
            print(f"找到匹配的 TXT 文件：{txt_file.name}，已复制到输出文件夹")


def split_workloads(img_dir, num_parts, saved_dir):
    """
        划分标注包
    """
    print(f'当前路径: {Path.cwd()}')

    img_dir_path = Path(img_dir)
    files = list(img_dir_path.glob('*'))
    print(f'{img_dir}中有{len(files)}个文件')
    random.shuffle(files)

    total_files = len(files)
    files_per_part  = len(files) // num_parts
    remainder = total_files % num_parts

    saved_dir_path = Path(saved_dir)
    saved_dir_path.mkdir(parents=True, exist_ok=True)

    for i in range(num_parts):
        part_folder = saved_dir_path / f"part_{i+1}"
        part_folder.mkdir(exist_ok=True)

    start_index = 0
    for i in tqdm(range(num_parts)):
        part_size = files_per_part + (1 if i < remainder else 0)
        part_files = files[start_index:start_index + part_size]
        
        part_folder = saved_dir_path / f"part_{i+1}"
        print(f"第 {i+1} 份保存到文件夹 {part_folder}:")
        for file in part_files:
            shutil.copy(file, part_folder)  # 将文件复制到对应的文件夹中
            print(file)
        print()
        
        start_index += part_size

    print(f"文件夹中的文件已经被随机分成了 {num_parts} 份")



def modify_class_index(txt_dir: str, origin_index: int, modify_index: int, saved_dir: str):
    """
        修改txt中的类别

        txt_dir: 待修改的txt文件夹
        origin_index: 原始的索引值
        modify_index: 修改之后的索引值
        saved_dir: 修改完成的txt保存路径
    """
    txt_path = Path(txt_dir)

    saved_path = Path(saved_dir)
    saved_path.mkdir(parents=True, exist_ok=True)

    count_modify = 0  # 需要修改的txt数量
    count_no = 0  # 不需要修改的txt数量

    txt_files = txt_path.glob('*')
    for file in txt_files:
        with open(file, 'r') as f:
            lines = f.readlines()
        
        # 检查是否需要修改文件内容
        modified_lines = []
        need_modify = False
        for line in lines:
            first_character = line.strip()[0]  # 获取每行的第一个字符
            if first_character.isdigit() and int(first_character) == origin_index:
                line = str(modify_index) + line[1:]  # 修改第一个数字
                need_modify = True
            modified_lines.append(line)

        # 如果需要修改文件内容，则保存修改后的文件；否则直接复制原文件到输出文件夹
        if need_modify:
            count_modify += 1
            saved_file_path = saved_path / file.name
            with open(saved_file_path, 'w') as f:
                f.writelines(modified_lines)
                # print(f"文件 {file.name} 处理完成，已保存到 {saved_file_path}")
        else:
            count_no += 1
            saved_file_path = saved_path / file.name
            shutil.copy(file, saved_file_path)
            print(f"文件 {file.name} 不需要修改，已复制到 {saved_file_path}")

    print(f'{txt_dir}中共有{len(list(txt_path.iterdir()))}个文件')
    print(f'修改的txt数量: {count_modify}')
    print(f'未修改的txt数量: {count_no}')


def random_split_datasets():
    """
        将yolo格式的数据集划分为训练集和验证集
    :return: 
    """
    # TODO  后续改为pathlib库实现
    print('🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃')
    print('🏃🏃🏃🏃🏃🏃🏃🏃🏃 run func random_split_datasets 🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃')
    print('🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃🏃')

    source_images_path = '/home/tuc/shuziyunying/img'  # 图片文件夹
    source_labels_path = '/home/tuc/shuziyunying/txts'  # txt文件夹
    print('img的数量:', len(os.listdir(source_images_path)))
    print('txt的数量:', len(os.listdir(source_labels_path)))

    base_path = '/home/tuc/shuziyunying/datasets/bank_counter_violation_detection_v1.1'  # 存放分割后的文件夹

    images_path = os.path.join(base_path, "images")
    train_images_path = os.path.join(images_path, "train")
    val_images_path = os.path.join(images_path, "val")
    test_images_path = os.path.join(images_path, "test")

    labels_path = os.path.join(base_path, "labels")
    train_labels_path = os.path.join(labels_path, "train")
    val_labels_path = os.path.join(labels_path, "val")
    test_labels_path = os.path.join(labels_path, "test")

    for path in [base_path, train_images_path, val_images_path, test_images_path, train_labels_path, val_labels_path,
                 test_labels_path]:
        if os.path.exists(path):
            pass
            # del_folder_contents(path)
        else:
            os.makedirs(path)

    print(train_images_path)
    print(val_images_path)
    print(test_images_path)
    print(train_labels_path)
    print(val_labels_path)
    print(test_labels_path)

    # 划分训练集和验证集图片
    num_train_img = 0
    num_val_img = 0
    num_test_img = 0
    source_images_list = os.listdir(source_images_path)
    rate = 0.80
    for image in source_images_list:
        gailv = random.random()
        if rate < gailv <= 0.9:
            # 划为验证集图片
            shutil.copy2(os.path.join(source_images_path, image), val_images_path)
            num_val_img += 1
        elif gailv <= rate:
            # 划为训练集图片
            shutil.copy2(os.path.join(source_images_path, image), train_images_path)
            num_train_img += 1
        elif 0.9 < gailv <= 1.0:
            # 划为测试集图片
            shutil.copy2(os.path.join(source_images_path, image), test_images_path)
            num_test_img += 1

    print('验证集图片的数量:', str(num_val_img))
    print('训练集图片的数量:', str(num_train_img))
    print('测试集图片的数量:', str(num_test_img))

    # 划分训练集和验证集标签
    num_train_label = 0
    num_val_label = 0
    num_test_label = 0
    source_labels_list = os.listdir(source_labels_path)

    train_images_list = os.listdir(train_images_path)
    for tr_image in tqdm(train_images_list):
        tr_image_name = tr_image.split('.')[0]
        for label in source_labels_list:
            label_name = label.split('.')[0]
            if label_name == tr_image_name:
                # 拷贝txt到labels/train文件夹下
                shutil.copy2(os.path.join(source_labels_path, label), train_labels_path)
                num_train_label += 1
                break

    time.sleep(5)
    val_images_list = os.listdir(val_images_path)
    for val_image in tqdm(val_images_list):
        val_image_name = val_image.split('.')[0]
        for label in source_labels_list:
            label_name = label.split('.')[0]
            if label_name == val_image_name:
                # 拷贝txt到labels/train文件夹下
                shutil.copy2(os.path.join(source_labels_path, label), val_labels_path)
                num_val_label += 1
                break

    time.sleep(5)
    test_images_list = os.listdir(test_images_path)
    for test_image in tqdm(test_images_list):
        test_image_name = test_image.split('.')[0]
        for label in source_labels_list:
            label_name = label.split('.')[0]
            if label_name == test_image_name:
                # 拷贝txt到labels/train文件夹下
                shutil.copy2(os.path.join(source_labels_path, label), test_labels_path)
                num_test_label += 1
                break

    print('验证集标签的数量:', str(num_val_label))
    print('训练集标签的数量:', str(num_train_label))
    print('测试集标签的数量:', str(num_test_label))


def check_class_index(txt_dir, check_index, saved_dir):
    """
        排查txt文件中的异常索引
        txt_dir: 待检查的txt文件夹
        check_index: 正确的索引
        saved_dir: 异常索引txt文件保存路径
    """

    print()
    txt_path = Path(txt_dir)
    txt_files = txt_path.glob('*.txt')

    num_error = 0
    for file in txt_files:
        copy_file = False

        with open(file, 'r') as f:
            lines = f.readlines()
        for line in lines:
            first_character = line.strip()[0]  # 获取每行的第一个字符
            if int(first_character) != check_index:
                num_error += 1
                # print(f'{file.name} 中有异常索引。')
                copy_file = True
                break
        if copy_file:
            if num_error == 1:
                saved_path = Path(saved_dir)
                saved_path.mkdir(parents=True, exist_ok=True)
            saved_file_path = saved_path / file.name
            shutil.copy(file, saved_file_path)
            print(f"文件 {file.name} 存在异常索引，已复制到 {saved_dir} 中")
    
    print()
    num_txt_files = len(list(txt_path.glob('*.txt')))
    print(f"{txt_dir} 中的 .txt 文件数量: {num_txt_files}")
    print(f'{txt_dir} 中存在异常索引的txt文件数量是: {num_error}')


def count_box_num():
    txt_path = ''  # txt文件所在路径
    class_num = 5  # 样本类别数
    class_list = [i for i in range(class_num)]
    class_num_list = [0 for i in range(class_num)]
    labels_list = os.listdir(txt_path)
    for i in labels_list:
        if i == 'classes.txt':
            print('exit classes.txt')
            continue
        file_path = os.path.join(txt_path, i)
        file = open(file_path, 'r')  # 打开文件
        file_data = file.readlines()  # 读取所有行
        for every_row in file_data:
            class_val = every_row.split(' ')[0]
            class_ind = class_list.index(int(class_val))
            class_num_list[class_ind] += 1
        file.close()
    # 输出每一类的数量以及总数
    print(class_num_list)
    print('total:', sum(class_num_list))




def convert(size, box):
    x_center = (box[0] + box[1]) / 2.0
    y_center = (box[2] + box[3]) / 2.0
    x = x_center / size[0]
    y = y_center / size[1]
    w = (box[1] - box[0]) / size[0]
    h = (box[3] - box[2]) / size[1]
    return (x, y, w, h)


def convert_format(xml_files_path, save_txt_files_path, classes):
    xml_files = os.listdir(xml_files_path)
    print(xml_files)
    for xml_name in xml_files:
        print(xml_name)
        xml_file = os.path.join(xml_files_path, xml_name)
        out_txt_path = os.path.join(save_txt_files_path, xml_name.split('.')[0] + '.txt')
        out_txt_f = open(out_txt_path, 'w')
        tree = ET.parse(xml_file)
        root = tree.getroot()
        size = root.find('size')
        w = int(size.find('width').text)
        h = int(size.find('height').text)

        for obj in root.iter('object'):
            difficult = obj.find('difficult').text
            cls = obj.find('name').text
            if cls not in classes or int(difficult) == 1:
                continue
            cls_id = classes.index(cls)
            xmlbox = obj.find('bndbox')
            b = (float(xmlbox.find('xmin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymin').text),
                 float(xmlbox.find('ymax').text))
            # b=(xmin, xmax, ymin, ymax)
            print(w, h, b)
            bb = convert((w, h), b)
            out_txt_f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')


def xml_to_txt():
    # 需要转换的类别，需要一一对应
    classes = ['person', 'drawer_open', 'key', 'seal', 'cash', 'cash_box_open', 'cash_box_close', 'head']
    # 2、voc格式的xml标签文件路径
    xml_files = 'label_cash_box_open_D06_20240306163200'
    # 3、转化为yolo格式的txt标签文件存储路径
    save_txt_files = 'saved_txt'
    # 如果文件夹不存在，则创建
    if not os.path.exists(save_txt_files):
        os.makedirs(save_txt_files)

    convert_format(xml_files, save_txt_files, classes)


if __name__=='__main__':
    # split_workloads('p1/frame-videos_fps_1-img', 2, 'p1-save-img')

    # png_match_txt('p2/frame-videos-add_fps_1-img', 'p2/frame-videos-add_fps_1-img-labels-seal', 'p2-save-img/part_1_label_seal')

    # modify_class_index(txt_dir='/home/tuc/shuziyunying/yolov5-6.1/runs/detect/exp3/labels', 
    #                    origin_index=0, 
    #                    modify_index=2,
    #                    saved_dir='/home/tuc/shuziyunying/save-label')

    # random_split_datasets()

    # 异常索引排查
    # check_class_index(txt_dir='/home/tuc/shuziyunying/txt/seal-label',
    #                   check_index=3,
    #                   saved_dir='error_seal_txt')

    # xml格式转txt格式
    xml_to_txt()

