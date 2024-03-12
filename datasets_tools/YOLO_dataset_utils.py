import random
import shutil
from os import PathLike
from pathlib import Path
from typing import List, Tuple, Union

from loguru import logger
import re

from tqdm import tqdm


class YOLOLabelValidator(object):
    def __init__(self):
        self.categories = ['train', 'val', 'test']
        self.percent = [0.8, 0.1, 0.1]
        self.image_suffix = ['.jpg', '.png', '.jpeg']

    @staticmethod
    def cxcy_to_xyxy(box):
        """
        将中心点坐标和宽高转换为左上角和右下角坐标
        :param box:
        :return:
        """
        cx, cy, w, h = box
        x1 = cx - w / 2
        y1 = cy - h / 2
        x2 = cx + w / 2
        y2 = cy + h / 2
        return x1, y1, x2, y2

    def iou(self, box1, box2):
        """
        计算两个矩形框的IOU(交并比),输入参数为两个矩形框的坐标，格式为[cls, cx, cy, w, h]
        :param box1:
        :param box2:
        :return: 输出为一个浮点数，表示两个矩形框的IOU值，范围为[0, 1]
        """
        # 获取两个矩形框的中心点坐标和宽高
        w1, h1 = box1[2:]
        w2, h2 = box2[2:]
        # 计算两个矩形框的左上角和右下角坐标
        x1, y1, x2, y2 = self.cxcy_to_xyxy(box1)
        x3, y3, x4, y4 = self.cxcy_to_xyxy(box2)

        # 计算两个矩形框的相交区域的坐标
        x_inter1, y_inter1 = max(x1, x3), max(y1, y3)
        x_inter2, y_inter2 = min(x2, x4), min(y2, y4)

        # 判断两个矩形框是否相交，如果不相交，则IOU为0
        if x_inter1 >= x_inter2 or y_inter1 >= y_inter2:
            return 0.0
        # 计算两个矩形框的相交区域的面积
        area_inter = (x_inter2 - x_inter1) * (y_inter2 - y_inter1)
        # 计算两个矩形框的并集区域的面积
        area_union = w1 * h1 + w2 * h2 - area_inter
        # 计算两个矩形框的IOU值
        iou = area_inter / area_union
        # 返回IOU值
        return iou

    # 定义一个函数，用于统计同一类别的标注框面积
    # 输入参数为一个文件名，文件中每一行为一个矩形框的坐标，格式为[cls, cx, cy, w, h]
    # 输出为一个字典，键为类别，值为该类别的所有标注框的面积之和
    def area(self, label_file):
        """
        统计同一类别的标注框面积,输入参数为一个文件内容为YOLO格式的标签文件，统计每个类别的标注框的面积，输出为一个字典，键为类别，值为该类别的所有标注框的面积列表
        :param label_file:
        :return: dict {key:[area1,area2]}
        """
        if isinstance(label_file, PathLike):
            label_file = Path(label_file)
        assert label_file.exists(), "label_file not exists"

        # 创建一个空字典，用于存储类别和面积
        area_dict = {}
        # 打开文件，读取所有的矩形框坐标
        with open(label_file, 'r') as f:
            import numpy as np
            for line in f:
                # 将每一行的字符串转换为浮点数数组
                box = np.array([float(x) for x in line.split()])
                # 获取矩形框的类别和宽高
                cls, w, h = int(box[0]), box[3], box[4]
                # 计算矩形框的面积
                area = w * h
                # 如果类别在字典中，则累加面积
                if cls in area_dict:
                    area_dict[cls].append(area)
                # 如果类别不在字典中，则创建新的键值对
                else:
                    area_dict[cls] = [area]
        # 返回字典
        return area_dict


class YOLODatasetUtils(object):
    def __init__(self):
        self.categories = ['train', 'val', 'test']
        self.percent = [0.8, 0.1, 0.1]
        self.image_suffix = ['.jpg', '.png', '.jpeg']

    @staticmethod
    def _validate_str_as_float(float_str):
        """
        验证字符串是否是浮点数
        :param float_str:
        :return:
        """
        float_str_pattern = r'^[0-1]\.[0-9]+$'
        if re.match(float_str_pattern, float_str):
            return True
        else:
            return False

    @staticmethod
    def _validate_str_as_int(int_str):
        """
        验证字符串是否是整数
        :param int_str:
        :return:
        """
        int_str_pattern = r'^[0-9]+$'
        if re.match(int_str_pattern, int_str):
            return True
        else:
            return False

    def validate_yolo_format(self, label_dir) -> Tuple[List[Path], List[Path], List[Path], List[Path]]:
        """
        验证YOLO格式的label文件的格式，正确的格式为：cls x y w h，其中cls为int类型,表示类别，x,y,w,h为浮点数，表示bbox的中心点坐标和宽高
        不合规的label文件包括：
        1. 空白label文件
        2. 每行数据不是5个
        3. cls不是int类型
        4. x,y,w,h不是float类型
        5. 行尾有空格或tab
        :return: 返回三个列表，分别为空白标签文件列表，无效标签文件列表，有效标签文件列表
        TODO 需要兼容当 w 和 h 为 1时，通过验证
        """
        blank_label_files = []
        invalid_label_files = []
        valid_label_files = []
        space_end_line_label_files = []
        label_files = list(label_dir.glob('*.txt'))
        logger.info("日志校验中......")

        for label_file in tqdm(label_files):
            if label_file.name == "classes.txt":
                continue
            with open(label_file, 'r', encoding='utf-8') as fr:
                lines = [line.strip() for line in fr.readlines()]
            if len(lines) == 0:
                blank_label_files.append(label_file)
                continue
            for line in lines:
                if len(line) == 0:
                    blank_label_files.append(label_file)
                    break
                if line[-1] == ' ' or line[-1] == '\t':
                    space_end_line_label_files.append(label_file)
                    break
                if len(line.split()) != 5:
                    invalid_label_files.append(label_file)
                    break
                cls, x, y, w, h = line.split()
                if (not all(list(map(self._validate_str_as_float, [x, y, w, h])))) or (
                        not self._validate_str_as_int(cls)):
                    invalid_label_files.append(label_file)
                    break

            else:
                valid_label_files.append(label_file)
        status = f"""
        空白标签文件：{len(blank_label_files)},
        无效标签文件：{len(invalid_label_files)},
        行尾空格标签文件：{len(space_end_line_label_files)},
        有效标签文件：{len(valid_label_files)}       
        """
        logger.info(status)
        if len(blank_label_files) > 0:
            logger.info(f"空白标签文件：{blank_label_files}")
        if len(invalid_label_files) > 0:
            logger.info(f"无效标签文件：{invalid_label_files}")
        if len(space_end_line_label_files) > 0:
            logger.info(f"行尾空格标签文件：{space_end_line_label_files}")

        return blank_label_files, invalid_label_files, space_end_line_label_files, valid_label_files

    @staticmethod
    def _sort_yolo_data_by_area(data):
        def calculate_area(box):
            _, cx, cy, w, h = box
            area = w * h
            return area

        # 将数据按照面积排序
        sorted_data = sorted(data, key=calculate_area, reverse=True)

        return sorted_data

    def sort_label_files(self, label_dir, overwrite=True):
        """
        对label文件进行排序，按照面积从大到小排序，排序后使用LabelImg查看可以避免大面积标签遮挡小面积标签的问题
        :param overwrite:
        :return:
        """
        label_files = list(label_dir.glob('**/*.txt'))

        if not overwrite:
            dst_label_dir = Path(str(label_dir) + "_sorted")
            dst_label_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"sorted label files will be saved in {dst_label_dir}")

        for label_file in tqdm(label_files):
            if label_file.name == "classes.txt":
                continue
            # 读取txt文件中的数据
            with open(label_file, 'r', encoding='utf-8') as fr:
                yolo_data = [list(map(float, line.strip().split())) for line in fr]

            # 调用函数进行排序
            try:
                sorted_yolo_data = self._sort_yolo_data_by_area(yolo_data)
            except:
                print(label_file)
                continue

            # 将排序后的数据写入txt文件
            if not overwrite:
                dst_label_file = dst_label_dir / label_file.name
            else:
                dst_label_file = label_file

            with open(dst_label_file, 'w+', encoding='utf-8') as fw:
                for data in sorted_yolo_data:
                    cls = int(data[0])
                    fw.write(f"{cls} {data[1]} {data[2]} {data[3]} {data[4]}\n")

    def delete_labels_by_class(self, label_dir, del_labels: List[int]):
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        assert label_dir.exists(), "label_dir not exists"

        label_files = list(label_dir.glob('*.txt'))
        logger.info(f"delete labels with class {del_labels} ......")
        for label_file in tqdm(label_files):
            if label_file.name == "classes.txt":
                continue
            with open(label_file, 'r', encoding='utf-8') as fr:
                lines = [line.strip() for line in fr.readlines()]
            with open(label_file, 'w+', encoding='utf-8') as fw:
                for line in lines:
                    cls = int(line.split()[0])
                    if cls not in del_labels:
                        fw.write(f"{line}\n")

    def delete_non_corresponding_files(self, image_dir, label_dir):
        """
        delete the blank label files,and all the image/label files which have no corresponding label/image files
        :param image_dir:
        :param label_dir:
        :return:
        """
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        assert image_dir.exists() and label_dir.exists(), "image_dir or label_dir not exists"

        image_files = list(image_dir.glob('*.png'))
        label_files = [item for item in label_dir.glob('*.txt') if item.name != "classes.txt"]

        # step 1. delete blank label files
        logger.info("delete blank label files ......")
        blank_label_files, _, _, _ = self.validate_yolo_format(label_dir)
        for blank_label_file in blank_label_files:
            logger.info(f"delete blank label file: {blank_label_file}")
            blank_label_file.unlink()

        # step 2. delete the label files which have no corresponding image files
        logger.info("delete the label files which have no corresponding image files ......")
        for label_file in tqdm(label_files):
            corresponding_image_file = image_dir / (label_file.stem + ".png")
            if not corresponding_image_file.exists():
                logger.info(f"delete label file: {label_file}")
                label_file.unlink()

        # step 3. delete the image files which have no corresponding label files
        logger.info("delete the image files which have no corresponding label files ......")
        for image_file in tqdm(image_files):
            corresponding_label_file = label_dir / (image_file.stem + ".txt")
            if not corresponding_label_file.exists():
                logger.info(f"delete image file: {image_file}")
                image_file.unlink()

    def change_label_classes(self, label_dir, corrigenda: dict, overwrite=True):
        """
        将label_dir中的标签类别按照corrigenda进行更改,corrigenda的格式为 {原类别:新类别}
        :param label_dir:
        :param corrigenda: dict 格式 {原类别:新类别}
        :param overwrite: 默认为True，表示覆盖原有的label文件，否则将更改后的label文件保存在label_dir_changed目录下
        :return:
        """
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        assert label_dir.exists(), "label_dir not exists"

        label_files = list(label_dir.glob('**/*.txt'))

        if not overwrite:
            dst_label_dir = Path(str(label_dir) + "_changed")
            dst_label_dir.mkdir(parents=True, exist_ok=True)

            logger.info(f"changed label files will be saved in {dst_label_dir}")

        for label_file in tqdm(label_files):
            if label_file.name == "classes.txt":
                continue
            # 读取txt文件中的数据
            with open(label_file, 'r', encoding='utf-8') as fr:
                lines = [line.strip() for line in fr.readlines()]

            new_lines = []
            for line in lines:
                cls, cx, cy, w, h = line.split()
                if int(cls) in corrigenda.keys():
                    cls = corrigenda[int(cls)]
                line = f"{cls} {cx} {cy} {w} {h}"
                new_lines.append(line)
            # 将更改后的数据写入txt文件
            if not overwrite:
                dst_label_file = dst_label_dir / label_file.name
            else:
                dst_label_file = label_file
            with open(dst_label_file, 'w+', encoding='utf-8') as fw:
                for line in new_lines:
                    fw.write(line + '\n')

    def get_classes(self, label_dir):
        """
        获取label_dir中所有的类别
        :param label_dir:
        :return:
        """
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        assert label_dir.exists(), 'label_dir not exists'

        label_files = list(label_dir.glob('*.txt'))
        classes = []
        for label_file in tqdm(label_files):
            if label_file.name == 'classes.txt':
                continue
            with open(label_file, 'r', encoding='utf-8') as fr:
                lines = [item.strip() for item in fr.readlines()]

            for line in lines:
                cls = line.split()[0]
                if cls not in classes:
                    classes.append(cls)
        logger.info(f"classes: {classes}")
        return classes

    def split_dataset(self, image_dir: Union[Path, str], label_dir: Union[Path, str]):
        """
        split the dataset into training set, validation set and testing set
        :param image_dir:
        :param label_dir:
        :return:
        """
        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        assert image_dir.exists() and label_dir.exists(), "image_dir or label_dir not exists"

        image_files = list(image_dir.glob('*.jpg'))
        random.shuffle(image_files)

        train_percent = self.percent[self.categories.index('train')]
        val_percent = self.percent[self.categories.index('val')]
        training_image_num = int(len(image_files) * train_percent)
        validation_image_num = int(len(image_files) * val_percent)

        training_images = image_files[:training_image_num]
        validation_images = image_files[training_image_num:training_image_num + validation_image_num]
        testing_images = image_files[training_image_num + validation_image_num:]

        dst_root_dir = image_dir.parent / 'data'
        dst_train_image_dir = dst_root_dir / 'images/train'
        dst_train_label_dir = dst_root_dir / 'labels/train'
        dst_train_image_dir.mkdir(exist_ok=True, parents=True)
        dst_train_label_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"training set will be saved in {dst_train_image_dir},processing ......")
        for image_file in tqdm(training_images):
            corresponding_label_file = label_dir / (image_file.stem + ".txt")
            if corresponding_label_file.name == "classes.txt":
                continue
            shutil.copy(str(image_file), str(dst_train_image_dir))
            shutil.copy(str(corresponding_label_file), str(dst_train_label_dir))

        dst_val_image_dir = dst_root_dir / 'images/val'
        dst_val_label_dir = dst_root_dir / 'labels/val'
        dst_val_image_dir.mkdir(exist_ok=True, parents=True)
        dst_val_label_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"validation set will be saved in {dst_val_image_dir},processing ......")
        for image_file in tqdm(validation_images):
            corresponding_label_file = label_dir / (image_file.stem + ".txt")
            if corresponding_label_file.name == "classes.txt":
                continue
            shutil.copy(str(image_file), str(dst_val_image_dir))
            shutil.copy(str(corresponding_label_file), str(dst_val_label_dir))

        dst_test_image_dir = dst_root_dir / 'images/test'
        dst_test_label_dir = dst_root_dir / 'labels/test'
        dst_test_image_dir.mkdir(exist_ok=True, parents=True)
        dst_test_label_dir.mkdir(exist_ok=True, parents=True)
        logger.info(f"testing set will be saved in {dst_test_image_dir},processing ......")
        for image_file in tqdm(testing_images):
            corresponding_label_file = label_dir / (image_file.stem + ".txt")
            if corresponding_label_file.name == "classes.txt":
                continue
            shutil.copy(str(image_file), str(dst_test_image_dir))
            shutil.copy(str(corresponding_label_file), str(dst_test_label_dir))

    def concat_labels_with_same_name(self, src_label_dirs: List[Path], dst_label_dir: Path):
        """
        concat the content of the label files with the same name in src_label_dirs and write the result into the dst_label_dir
        :param src_label_dirs:
        :param dst_label_dir:
        :return:
        """
        dst_label_dir.mkdir(parents=True, exist_ok=True)
        label_filenames = []
        for src_label_dir in src_label_dirs:
            label_filenames.extend([label_file.name for label_file in src_label_dir.glob('*.txt')])

        label_filenames = list(set(label_filenames))

        for label_file in tqdm(label_filenames):
            if label_file == "classes.txt":
                continue
            lines = []
            for src_label_dir in src_label_dirs:
                if (src_label_dir / label_file).exists():
                    with open(src_label_dir / label_file, 'r', encoding='utf-8') as fr:
                        lines.extend([line.strip() for line in fr.readlines()])
            with open(dst_label_dir / label_file, 'w+', encoding='utf-8') as fw:
                for line in lines:
                    fw.write(line + '\n')

    def reset_image_label_files(self, image_root_dir, label_root_dir):
        """
        get all the image files in image_root_dir and its subdirectories,  and move them to a new directory
        named "images",and then do the same things to label files
        :param image_root_dir:
        :param label_root_dir:
        :return:
        """
        if isinstance(image_root_dir, str):
            image_root_dir = Path(image_root_dir)
        if isinstance(label_root_dir, str):
            label_root_dir = Path(label_root_dir)

        label_suffix = ['.txt', '.xml']

        image_files = []
        label_files = []
        logger.info("get all the image files and label files ......")
        for image_suffix_item in self.image_suffix:
            image_files.extend(list(image_root_dir.glob(f'**/*{image_suffix_item}')))

        for label_suffix_item in label_suffix:
            label_files.extend(list(label_root_dir.glob(f'**/*{label_suffix_item}')))

        logger.info("move all the image files and label files to a new directory named images_ori and labels_ori......")
        dst_image_dir = image_root_dir / 'images_ori'
        dst_label_dir = label_root_dir / 'labels_ori'
        dst_image_dir.mkdir(parents=True, exist_ok=True)
        dst_label_dir.mkdir(parents=True, exist_ok=True)

        for image_file in tqdm(image_files):
            shutil.copy2(str(image_file), str(dst_image_dir))

        for label_file in tqdm(label_files):
            shutil.copy2(str(label_file), str(dst_label_dir))

    def yolo_to_voc(self, image_dir, label_dir: Union[Path, str], labels, dst_dir: Union[Path, str]):
        """
        将YOLO格式的标注文件转换为VOC格式的标注文件
        :param label_dir:
        :param dst_dir:
        :return:
        """

        if isinstance(image_dir, str):
            image_dir = Path(image_dir)
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        if isinstance(dst_dir, str):
            dst_dir = Path(dst_dir)

        assert label_dir.exists(), "label_dir not exists"

        label_classes = self.get_classes(label_dir)
        # 校验labels是否数量上与classes一致
        # assert len(labels) == len(label_classes), "labels数量与classes数量不一致"
        label_mapping = dict(zip(label_classes, labels))
        logger.info(f"label映射关系: {label_mapping}")
        logger.info("开始转换......")
        import cv2
        label_files = list(label_dir.glob('*.txt'))
        dst_dir.mkdir(parents=True, exist_ok=True)
        for label_file in tqdm(label_files):
            if label_file.name == "classes.txt":
                continue
            # TODO define a function to get the corresponding image file in case of the image suffix is not .jpg
            corresponding_image_file = image_dir / (label_file.stem + ".jpg")
            if not corresponding_image_file.exists():
                continue
            image = cv2.imread(str(corresponding_image_file))
            try:
                image_height, image_width, _ = image.shape
            except:
                logger.error(f"image file: {corresponding_image_file.name} is broken")
                continue
            with open(label_file, 'r', encoding='utf-8') as fr:
                lines = [line.strip() for line in fr.readlines()]
            with open(dst_dir / label_file.name.replace('.txt', '.xml'), 'w+', encoding='utf-8') as fw:
                fw.write('<?xml version="1.0" encoding="utf-8"?>\n')
                fw.write('<annotation>\n')
                fw.write(f'\t<filename>{label_file.name.replace(".txt", ".jpg")}</filename>\n')
                fw.write('\t<size>\n')
                fw.write(f'\t\t<width>{image_width}</width>\n')
                fw.write(f'\t\t<height>{image_width}</height>\n')
                fw.write('\t\t<depth>3</depth>\n')
                fw.write('\t</size>\n')
                fw.write('\t<segmented>0</segmented>\n')
                for line in lines:
                    cls, cx, cy, w, h = line.split()
                    fw.write('\t<object>\n')
                    fw.write(f'\t\t<name>{labels[int(cls)]}</name>\n')
                    fw.write('\t\t<pose>Unspecified</pose>\n')
                    fw.write('\t\t<truncated>0</truncated>\n')
                    fw.write('\t\t<difficult>0</difficult>\n')
                    fw.write('\t\t<bndbox>\n')
                    fw.write(f'\t\t\t<xmin>{int(float(cx) * image_width - float(w) * image_width / 2)}</xmin>\n')
                    fw.write(f'\t\t\t<ymin>{int(float(cy) * image_height - float(h) * image_height / 2)}</ymin>\n')
                    fw.write(f'\t\t\t<xmax>{int(float(cx) * image_width + float(w) * image_width / 2)}</xmax>\n')
                    fw.write(f'\t\t\t<ymax>{int(float(cy) * image_height + float(h) * image_height / 2)}</ymax>\n')
                    fw.write('\t\t</bndbox>\n')
                    fw.write('\t</object>\n')
                fw.write('</annotation>\n')

    def split_src_images(self, image_dir, num_of_split=10):
        """
        split the image files in the image_dir and split them evenly into num_of_split parts,and then create num_of_split subdirs and copy the images into them
        :param image_dir: the directory of the images
        :return:
        """
        image_files = list(image_dir.glob('*.jpg'))

        # shuffle the image files
        random.shuffle(image_files)

        # split the image files into num_of_split parts
        split_image_files = []
        for i in range(num_of_split):
            split_image_files.append(
                image_files[int(i * len(image_files) / num_of_split):int((i + 1) * len(image_files) / num_of_split)])

        # create the subdirs
        for i in range(num_of_split):
            # create the subdirs with the same level of the image_dir
            dst_image_dir = image_dir.parent / f"part_{i}"
            dst_image_dir.mkdir(exist_ok=True, parents=True)
            # copy the images into the subdirs
            for image_file in tqdm(split_image_files[i]):
                shutil.copy(str(image_file), str(dst_image_dir))

    def find_label_files_by_class(self,image_dir, label_dir, classes: List[int]):
        """
        find the label files which contain the classes in classes
        :param label_dir:
        :param classes:
        :return:
        """
        if isinstance(label_dir, str):
            label_dir = Path(label_dir)
        assert label_dir.exists(), "label_dir not exists"

        dst_label_dir = label_dir.parent / 'labels_with_classes'
        dst_label_dir.mkdir(exist_ok=True, parents=True)

        label_files = list(label_dir.glob('*.txt'))
        label_files = [label_file for label_file in label_files if label_file.name != "classes.txt"]
        logger.info(f"find label files which contain the classes: {classes}")
        for label_file in tqdm(label_files):
            with open(label_file, 'r', encoding='utf-8') as fr:
                lines = [line.strip() for line in fr.readlines()]
            for line in lines:
                cls = int(line.split()[0])
                if cls in classes:
                    logger.info(f"label file: {label_file}")
                    corresponding_image_file = image_dir / (label_file.stem + ".jpg")
                    shutil.copy(str(label_file), str(dst_label_dir))
                    shutil.copy(str(corresponding_image_file), str(dst_label_dir))
                    break

    def test(self):
        """
        测试
        :return:
        """
        x, y, w, h = ['0.571875', '0.12d5463', '0.039583', '0.063889']
        print(list(map(self._validate_str_as_float, [x, y, w, h])))


if __name__ == '__main__':
    yolo_dataset_utils = YOLODatasetUtils()

    # root_dir = Path(r'H:\Aoto_labeling_work\samples')
    root_dir = Path('/home/tuc/shuziyunying/txt')
    # image_dir = root_dir / 'images'
    # label_dir = root_dir / 'labels'

    # 获取所有的类别
    # yolo_dataset_utils.get_classes(label_dir)
    # yolo_dataset_utils.find_label_files_by_class(image_dir, label_dir, [4])

    # 标注文件合法性校验
    # yolo_dataset_utils.validate_yolo_format(label_dir)

    # 删除标注中的某些类别
    # yolo_dataset_utils.delete_labels_by_class(label_dir, [0,1, 4])

    # 将标注中的某些类别更改为其他类别(0改为4)
    # yolo_dataset_utils.change_label_classes(label_dir, {4:1})

    # 将标注文件按照面积从大到小排序
    # yolo_dataset_utils.sort_label_files(label_dir, overwrite=True)

    # 将多个同名标注文件合并成一个
    # label_dirs = [
    #     root_dir / 'cash-label',
    #     root_dir / 'key-label',
    #     root_dir / 'seal-label'
    # ]
    # dst_label_dir = root_dir / 'txts'
    # yolo_dataset_utils.concat_labels_with_same_name(label_dirs, dst_label_dir)

    # 删除冗余文件，包括空白标注文件，行尾有空格的标注文件，没有对应图片的标注文件，没有对应标注文件的图片文件
    # yolo_dataset_utils.delete_non_corresponding_files(image_dir, label_dir)

    # 划分训练集、验证集和测试集
    # yolo_dataset_utils.split_dataset(image_dir='/home/tuc/shuziyunying/img/', 
    #                                  label_dir='/home/tuc/shuziyunying/txts/')

    # 重置图片和标注文件，将所有子文件夹中的图片和标注文件移动到根目录下的images_ori和labels_ori文件夹中
    # yolo_dataset_utils.reset_image_label_files(root_dir, root_dir)

    # 将YOLO格式的标注文件转换为VOC格式的标注文件
    # image_dir = Path(r"E:\python\yolo_project\yolov5-6.1\new_all_smoke_data\smoke9")
    # label_dir = Path(r"E:\python\yolo_project\yolov5-6.1\runs\detect\smoke9\labels")
    # label_list = ['head', 'hand', 'smoke']
    # dst_dir = image_dir.parent / (image_dir.name + 'labels_voc')
    # yolo_dataset_utils.yolo_to_voc(image_dir, label_dir, label_list, dst_dir)

    # image_file = Path(r"c:\Users\lujx\Desktop\20240120155125.png")
    # import cv2
    # image = cv2.imread(str(image_file),cv2.IMREAD_GRAYSCALE)
    # print(image.shape)
    # cv2.imshow("image",image)
    # cv2.waitKey(0)
