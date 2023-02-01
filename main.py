from pathlib import Path
import os

import numpy as np

# pip install opencv-python==4.5.5.62
import cv2.cv2 as cv2


class PngAnalyser():
    def __init__(self):
        self.width = 0
        self.height = 0
        self.max_width = 0
        self.max_height = 0
        self.min_width = 10000
        self.min_height = 10000

    def size_finder(self, image_path):
        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        self.height = img.shape[0]
        self.width = img.shape[1]
        return self.height, self.width

    def save_in_grey(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        img_new_name = img_path[:-4] + '-g.png'
        cv2.imwrite(img_new_name, img)

    def open_image(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

        # print(img.shape)

        height = img.shape[0]
        width = img.shape[1]

        if height % 2 == 1 and width % 2 == 1:
            img = img[:-1, :-1]

        if height % 2 == 1 and width % 2 != 1:
            img = img[:-1, :]

        if height % 2 != 1 and width % 2 == 1:
            img = img[:, :-1]

        # height, width, channels = img.shape
        # print(img.shape)
        # img = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)

        # print(img[0])
        # print('Количество строк = ', len(img))
        # print('Количество колонн = ', len(img[0]))
        # print('Каждый пиксель из строки из колонны состоит из ', len(img[0][0]), 'значений по РГБ')
        # print('Один белый пиксель это массив из чисел 255')
        # print('Высота у нас составляет: ', img.shape[0], type(img.shape[0]))
        # print('Ширина у нас составляет: ', img.shape[1], type(img.shape[1]))

        return img

    def create_white_img(self, x=600, y=600):
        # one_line_RGB = np.full((x, 3), 255, dtype=np.uint8)
        # print(one_line_RGB)
        # img = np.full((x, y, 3), 255, dtype=np.uint8)  # Создаем NP массив и заполяем его значениями 255
        img = np.full((x, y), 255, dtype=np.uint8)  # Создаем NP массив и заполяем его значениями 255

        # # print(img)
        # print(len(img))
        # print(len(img[0]))
        # print(img[0][0])

        # print(img.shape)
        # print(img)

        # cv2.imshow('500x500 white', img)  # show
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        return img

    # def change_height(self, img_path):
    #     img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    #     # img.shape[0] += 1
    #
    #     list_of_array = []
    #     while range(0, img.shape[1]):
    #         list_of_array.append(img[0])
    #
    #     array_of_list = np.array(list_of_array)
    #     np.append(img, array_of_list)
    #
    #     cv2.imshow('I am crazy!', img)
    #     cv2.waitKey(0)

    def loading_displaying_saving(self):
        img = cv2.imread('1.png', cv2.IMREAD_GRAYSCALE)
        # img = cv2.imread('1.png', cv2.IMREAD_UNCHANGED)
        cv2.imshow('1', img)
        print("Высота:" + str(img.shape[0]))
        print("Ширина:" + str(img.shape[1]))
        cv2.waitKey(0)
        cv2.imwrite('1-grey.png', img)


class PngRefactor(PngAnalyser):
    def find_max_size(self, list_of_image_path):
        for image_path in list_of_image_path:
            self.size_finder(image_path)
            if self.width > self.max_width:
                self.max_width = self.width
            if self.height > self.max_height:
                self.max_height = self.height
        return self.max_height, self.max_width

    def find_min_size(self, list_of_image_path):
        for image_path in list_of_image_path:
            self.size_finder(image_path)
            if self.width < self.min_width:
                self.min_width = self.width
            if self.height < self.min_height:
                self.min_height = self.height
        return self.min_height, self.min_width

    def past_img_into_img(self, img1, img2, *bias):
        back_rows, back_cols = img1.shape[:2]
        # rows, cols, channels = img2.shape
        rows, cols = img2.shape[:2]
        # Дабы картинка вставлялась в центр другой картинки
        # Таким образом roi это запись img[a:b, c:d] означает, что
        # Как в случае со строками мы берем строки с a до b
        # В этих строках с c до d
        if len(bias) == 2:
            upper_rows = int(int(back_rows / 2) - int(rows / 2) + bias[0])
            downer_point = int(int(back_rows / 2) + int(rows / 2) + bias[0])
            lefter_point = int(int(back_cols / 2) - int(cols / 2) + bias[1])
            righter_point = int(int(back_cols / 2) + int(cols / 2) + bias[1])
        else:
            upper_rows = int(int(back_rows / 2) - int(rows / 2))
            downer_point = int(int(back_rows / 2) + int(rows / 2))
            lefter_point = int(int(back_cols / 2) - int(cols / 2))
            righter_point = int(int(back_cols / 2) + int(cols / 2))
        # Это у нас координаты куда надо вставить мое изображение
        # roi = img1[upper_rows:downer_point, lefter_point:righter_point]
        # roi = img2
        try:
            # print('shape of peace img1', img1[upper_rows:downer_point, lefter_point:righter_point].shape)
            img1[upper_rows:downer_point, lefter_point:righter_point] = img2
        except:
            print('EXCEPT!', 'img1', img1.shape)
            print('EXCEPT!', 'img2', img2.shape)
            cv2.imshow(img2, 'img2')
            cv2.waitKey(0)
            cv2.imshow(img2, 'img1')
            cv2.waitKey(0)
        # print(type(img1gray))
        # print(f'Размеры 1 матрицы равны: {img1gray.shape[0]} x {img1gray.shape[1]}')
        #
        # print(type(img2gray))
        # print(f'Размеры 2 матрицы равны: {img2gray.shape[0]} x {img2gray.shape[1]}')
        return img1

    def resize(self, img, height=100, width=100):
        dim = (height, width)
        # уменьшаем изображение до подготовленных размеров
        resize_img = cv2.resize(img, dim, interpolation=cv2.INTER_AREA)
        cv2.imshow("Resize image", resize_img)
        cv2.waitKey(0)
        return resize_img

    def rotate(self, img, angle):
        # if degrees < 0 - rotated in clockwise
        (h, w) = img.shape[:2]
        center = (w / 2, h / 2)
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated_img = cv2.warpAffine(img, M, (w, h))
        # cv2.imshow("Rotated image", rotated_img)
        # cv2.waitKey(0)
        return rotated_img

    def inversion(self, img):
        inversion_img = cv2.bitwise_not(img)
        return inversion_img


class DatasetCreator:
    def __init__(self):
        self.PngRefactor = PngRefactor()
        # All raw images path
        self.Alphabet_path = self.alphabet_path()
        self.new_alphabet_path = {}
        self.new_letter_name = {}

    def alphabet_path(self):
        Alphabet_path = {}
        alphabet_folders = Path(os.getcwd(), 'Alphabet')
        list_of_alphabet_folders = os.listdir(alphabet_folders)
        for letter_folder in list_of_alphabet_folders:
            list_of_letter = os.listdir(Path(alphabet_folders, str(letter_folder)))
            list_with_path = []
            for letter in list_of_letter:
                list_with_path.append(Path(alphabet_folders, str(letter_folder), str(letter)))
            Alphabet_path[str(letter_folder)] = list_with_path
        return Alphabet_path

    def saver_img(self, img, img_type):
        def which_a_slesh(path):
            if '/' in str(path):
                slesh = '/'
            elif '\\' in str(path):
                slesh = '\\'
            else:
                slesh = '/'
            return slesh

        new_alphabet_folder = Path(os.getcwd(), 'new_Alphabet')
        if img_type not in os.listdir(new_alphabet_folder):
            os.mkdir(str(new_alphabet_folder) + which_a_slesh(new_alphabet_folder) + img_type)
            self.new_letter_name[img_type] = 1
            self.new_alphabet_path[img_type] = []

        img_new_name = str(self.new_letter_name[img_type]) + '-g.png'
        self.new_letter_name[img_type] += 1
        slesh = which_a_slesh(new_alphabet_folder)
        new_img_path = str(new_alphabet_folder) + slesh + img_type + slesh + img_new_name
        self.new_alphabet_path[img_type].append(new_img_path)
        cv2.imwrite(new_img_path, img)

    def aspect_ratio(self, img, final_height=None, final_width=None):
        # Нам надо сохранить соотношение сторон
        # чтобы изображение не исказилось при уменьшении
        # для этого считаем коэф. уменьшения стороны
        if final_width is not None and final_height is None:
            r = float(final_width) / img.shape[1]
            dim = (final_width, int(img.shape[0] * r))
            return dim
        elif final_height is not None and final_width is None:
            r = float(final_height) / img.shape[0]
            dim = (int(img.shape[1] * r), final_height)
            return dim
        else:
            return False


def path_of_images():
    path_of_folders = Path(os.getcwd(), 'Alphabet')
    folder_list = os.listdir(path_of_folders)
    path_files_list = []
    for folder in folder_list:
        path_of_files = Path(path_of_folders, str(folder))
        files_list = os.listdir(path_of_files)

        for files in files_list:
            path_files_list.append(str(Path(path_of_files, str(files))))

    return path_files_list


if __name__ == '__main__':
    # PngRefactor = PngRefactor()
    # # PngRefactor.loading_displaying_saving()
    # image_paths = path_of_images()
    #
    # # print(PngRefactor.find_max_size(image_paths))
    # # print(PngRefactor.find_min_size(image_paths))
    #
    # # PngRefactor.open_image(image_paths[0])
    # img_base = PngRefactor.create_white_img()
    # img_pasted = PngRefactor.open_image(image_paths[0])
    #
    # created_img = PngRefactor.past_img_into_img(img_base, img_pasted)
    # # created_img = PngRefactor.overlay(img_base, img_pasted)
    #
    # inversion_img = PngRefactor.inversion(created_img)
    #
    # rotated_img = PngRefactor.rotate(inversion_img, 30)
    #
    # cv2.imshow('my crate img', created_img)
    # cv2.waitKey(0)
    # PngRefactor.change_height(image_paths[0])

    Data_creator = DatasetCreator()

    list_of_all_img_path = []
    for letter_type in Data_creator.Alphabet_path:
        for path in Data_creator.Alphabet_path[letter_type]:
            list_of_all_img_path.append(str(path))

    Data_creator.PngRefactor.find_max_size(list_of_all_img_path)
    print('Max raw width:', Data_creator.PngRefactor.max_width)
    print('Max raw height:', Data_creator.PngRefactor.max_height)

    # I want squared image
    if Data_creator.PngRefactor.max_width >= Data_creator.PngRefactor.max_height:
        max_side = Data_creator.PngRefactor.max_width
    else:
        max_side = Data_creator.PngRefactor.max_height
    size_side = ((max_side // 100) + 1) * 100
    print(f'Size new images: {size_side} x {size_side}')

    size = (size_side, size_side)

    for letter_type in Data_creator.Alphabet_path:
        # Здесь мы имеем тип буквы
        # Create a simple dataset
        dataset = []
        for letter_img in Data_creator.Alphabet_path[letter_type]:
            blanc_img = Data_creator.PngRefactor.create_white_img(size[0], size[1])
            raw_img = Data_creator.PngRefactor.open_image(str(letter_img))

            # Create simple image
            simple_img = Data_creator.PngRefactor.past_img_into_img(blanc_img, raw_img)
            new_simple_img = Data_creator.PngRefactor.inversion(simple_img)
            dataset.append(new_simple_img)

            # Create rotated images
            angles = (-30, -20, -10, 10, 20, 30)
            for angle in angles:
                rotated_img = Data_creator.PngRefactor.rotate(new_simple_img, angle)
                dataset.append(rotated_img)

            # Create biased images
            bias_y = int((blanc_img.shape[0] - raw_img.shape[0]) / 4)
            bias_x = int((blanc_img.shape[1] - raw_img.shape[1]) / 4)
            if bias_y % 2 != 0:
                bias_y -= 1
            if bias_x % 2 != 0:
                bias_x -= 1
            biases = ((bias_y, bias_x), (-bias_y, -bias_x))
            for bias in biases:
                blanc_img = Data_creator.PngRefactor.create_white_img(size[0], size[1])
                biased_img = Data_creator.PngRefactor.past_img_into_img(blanc_img, raw_img, *[bias[0], 0])
                new_beased_img = Data_creator.PngRefactor.inversion(biased_img)
                dataset.append(new_beased_img)

                blanc_img = Data_creator.PngRefactor.create_white_img(size[0], size[1])
                biased_img = Data_creator.PngRefactor.past_img_into_img(blanc_img, raw_img, *[0, bias[1]])
                new_beased_img = Data_creator.PngRefactor.inversion(biased_img)
                dataset.append(new_beased_img)

                blanc_img = Data_creator.PngRefactor.create_white_img(size[0], size[1])
                biased_img = Data_creator.PngRefactor.past_img_into_img(blanc_img, raw_img, *[-bias[0], bias[1]])
                new_beased_img = Data_creator.PngRefactor.inversion(biased_img)
                dataset.append(new_beased_img)

                blanc_img = Data_creator.PngRefactor.create_white_img(size[0], size[1])
                biased_img = Data_creator.PngRefactor.past_img_into_img(blanc_img, raw_img, *bias)
                new_beased_img = Data_creator.PngRefactor.inversion(biased_img)
                dataset.append(new_beased_img)

        for created_img in dataset:
            # resized_img = Data_creator.PngRefactor.resize(created_img, 100, 100)
            Data_creator.saver_img(created_img, letter_type)

        # Обнуляем перед следующим циклом
        dataset.clear()

    print(' * End of program! * ')







