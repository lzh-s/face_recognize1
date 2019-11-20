#-*- coding:utf-8 -*-

import os
from tkinter import *
from tkinter.ttk import *
from tkinter.messagebox import *
import cv2
import numpy as np
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import platform

class ApplicationUI(Frame):
    """ 这个类仅实现界面生成功能 """

    def __init__(self, master=None, width=600, height=450):
        Frame.__init__(self, master)
        self.master.title('人脸识别大作业')
        # 居中显示
        ws = self.master.winfo_screenwidth()
        hs = self.master.winfo_screenheight()
        x = int((ws / 2) - (width / 2))
        y = int((hs / 2) - (height / 2))
        self.master.geometry('{}x{}+{}+{}'.format(width, height, x, y))
        self.create_widgets()

    def create_widgets(self):
        self.top = self.winfo_toplevel()

        self.style = Style()

        self.notebook = Notebook(self.top)
        self.notebook.place(relx=0.02, rely=0.02, relwidth=0.96, relheight=0.96)

        self.notebook_tab1 = Frame(self.notebook)
        self.notebook_tab1_lbl1 = Label(self.notebook_tab1, text='姓名：')
        self.notebook_tab1_lbl1.grid(row=0, column=0, padx=(50, 10), pady=50)
        # self.notebook_tab1_lbl.place()
        self.collect_name = StringVar()
        self.notebook_tab1_entry = Entry(self.notebook_tab1, width=20, textvariable=self.collect_name)
        self.collect_name.set("")
        self.notebook_tab1_entry.grid(row=0, column=1, pady=50)
        # self.notebook_tab1_entry.pack(side=LEFT)
        self.notebook_tab1_btn = Button(self.notebook_tab1, text="确认姓名并采集图像", command=self.confirm_collect)
        self.notebook_tab1_btn.grid(row=0, column=2, pady=50)
        # self.notebook_tab1_btn.pack()
        self.notebook_tab1_lbl2 = Label(self.notebook_tab1, text='说明：')
        self.notebook_tab1_lbl2.grid(row=1, column=0, padx=(50, 10), sticky=NW)
        self.notebook_tab1_lbl3 = Label(self.notebook_tab1, wraplength=320, text='如果输入的姓名与之前输入过的姓名重复，程序会删除该姓名下之前采集的图像并重新采集新的图像，以作为该姓名下的训练数据')
        self.notebook_tab1_lbl3.grid(row=1, column=1, columnspan=2, sticky=W)
        self.notebook.add(self.notebook_tab1, text='图像采集')

        self.notebook_tab2 = Frame(self.notebook)
        self.notebook_tab2_btn1 = Button(self.notebook_tab2, text='  svm识别  ', command=self.svm_recognize)
        self.notebook_tab2_btn1.grid(row=0, column=0, padx=(125, 0), pady=50, sticky=E)
        self.notebook_tab2_btn2 = Button(self.notebook_tab2, text='knn识别 k=', command=self.knn_recognize)
        self.notebook_tab2_btn2.grid(row=1, column=0, padx=(125, 0), pady=(0, 50), sticky=E)
        self.k_value = StringVar()
        self.notebook_tab2_entry = Entry(self.notebook_tab2, width=2, textvariable=self.k_value)
        self.k_value.set("5")
        self.notebook_tab2_entry.grid(row=1, column=1, padx=(0, 125), pady=(0, 50), sticky=W)
        self.notebook_tab2_lbl = Label(self.notebook_tab2, text='说明：如果想要停止检测识别，请按"q"键退出')
        self.notebook_tab2_lbl.grid(row=2, column=0, padx=125, columnspan=2)
        self.notebook.add(self.notebook_tab2, text='人脸识别')


class Application(ApplicationUI):
    """ 这个类实现具体的事件处理回调函数，子类 """

    def __init__(self, master=None, parent_path=None, cascade_path=None):
        ApplicationUI.__init__(self, master)
        self.parent_path = parent_path
        self.cascade_path = cascade_path

    def confirm_collect(self):
        name = self.collect_name.get()
        print(name)
        if name == "":
            showwarning('提示', '请输入正确的被采集人姓名')
            return

        # 判断 如果目录不存在，则创建目录，如果存在，则删除目录下所有图像
        path = self.parent_path + "/" + name
        if os.path.exists(path):
            files = os.listdir(path)
            for f in files:
                file_path = os.path.join(path, f)
                os.remove(file_path)
        else:
            os.makedirs(path)


        # print(path)
        collector = ImageCollector("采集人脸照片", 100, path, self.cascade_path)
        collector.capture_images()

    def svm_recognize(self):
        recognizer = FaceRecognizer("svm", image_num=100, parent_path=self.parent_path, cascade_path=self.cascade_path)
        recognizer.recognize()

    def knn_recognize(self):
        k_value = self.k_value.get()
        print(k_value)
        if not k_value.isdigit() or int(k_value) < 1:
            showwarning('提示', '请输入正确的k值（大于0的整数）')
            return
        recognizer = FaceRecognizer("knn", knn_k=int(k_value), image_num=100, parent_path=self.parent_path, cascade_path=self.cascade_path)
        recognizer.recognize()


class ImageCollector():
    """ 通过电脑摄像头检测人脸并截图 """

    def __init__(self, window_name, catch_pic_num, path_name, cascade_path):
        self.window_name = window_name
        self.catch_pic_num = catch_pic_num
        self.path_name = path_name
        self.cascade_path = cascade_path

    def text_to_image(self, image, text, position, font_size, font_color):
        img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('../fonts/simsun.ttc', font_size)
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, text, font=font, fill=font_color)
        image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        return image

    def capture_images(self):
        cv2.namedWindow(self.window_name)

        # 摄像头捕获的实时视频流
        capture = cv2.VideoCapture(0)
        # 人脸识别分类器
        classfier = cv2.CascadeClassifier(self.cascade_path)
        # 边框颜色
        color = (0, 255, 0)

        num = 0
        while capture.isOpened():
            # 读取一帧视频
            _, frame = capture.read()
            if not _:
                break

            # 图像灰度化，这是为了降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # 识别出人脸区域，这里的scaleFactor=1.2和minNeighbors=3分别表示图片缩放比例和需要检测的有效点数
            face_rects = classfier.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            # 检测到了人脸
            if len(face_rects) > 0:
                # 单独框出每一张人脸
                for face_rect in face_rects:
                    x, y, imgcol, imgrow = face_rect

                    # 将当前帧保存为图片
                    img_name = "%s/%d.jpg" % (self.path_name, num)
                    print(img_name)
                    image = frame[y - 10: y + imgrow + 10, x - 10: x + imgcol + 10]
                    cv2.imwrite(img_name, image, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])

                    num += 1
                    # 如果超过指定最大保存数量退出循环
                    if num > self.catch_pic_num-1:
                        break

                    # 画出矩形框
                    cv2.rectangle(frame, (x - 10, y - 10), (x + imgcol + 10, y + imgrow + 10), color, 2)

                    # 实时显示采集的数量
                    frame = self.text_to_image(frame, "数量:%d/%d" % (num+1, self.catch_pic_num), (x + 30, y + 30), 30, (255, 0, 255))

            # 如果超过指定最大保存数量结束程序
            if num > self.catch_pic_num-1:
                break

            # 显示图像
            cv2.imshow(self.window_name, frame)
            # 等待按键输入，10毫秒
            c = cv2.waitKey(10)
            # 退出循环
            if c & 0xFF == ord("q"):
                break

        # 释放摄像头
        capture.release()
        # 销毁所有窗口
        cv2.destroyAllWindows()

class FaceRecognizer():
    """ 通过电脑摄像头识别人脸 """

    def __init__(self, algorithm="svm", knn_k=5, image_num=100, parent_path=None, cascade_path=None):
        self.algorithm = algorithm
        self.knn_k = knn_k
        self.image_num = image_num
        self.parent_path = parent_path
        self.cascade_path = cascade_path

    IMAGE_SIZE = 224

    def is_hidden_file(self, path, dir_name):
        """ 是否是隐藏文件（注：linux下隐藏文件以.开头，windows下属性判断） """

        # 如果是windows系统
        if 'Windows' in platform.system():
            import win32file
            import win32con
            file_attribute = win32file.GetFileAttributes(path)
            if file_attribute & win32con.FILE_ATTRIBUTE_HIDDEN:
                return True
            return False
        # Linux等其他系统
        if dir_name.startswith('.'):
            return True
        return False

    def is_dir_and_not_hidden(self, path, dir_name):
        """ 是目录 且 不为隐藏目录 """

        if os.path.isdir(path) and not self.is_hidden_file(path, dir_name):
            return True
        return False


    def load_data(self):
        """ 载入训练数据集 """

        data = []
        label = []
        label_dict = {}

        directories = os.listdir(self.parent_path)
        j = 0
        for dir in directories:
            # 是目录 且 不为隐藏目录（注：linux下隐藏文件以.开头，windows下属性判断）
            if self.is_dir_and_not_hidden(os.path.join(self.parent_path, dir), dir):
                j = j + 1
                for number in range(self.image_num):
                    path_full = os.path.join(self.parent_path, dir) + "/" + str(number) + ".jpg"
                    image = Image.open(path_full).convert("L")
                    image = image.resize((self.IMAGE_SIZE, self.IMAGE_SIZE), Image.ANTIALIAS)
                    img = np.reshape(image, (1, self.IMAGE_SIZE * self.IMAGE_SIZE))
                    data.extend(img)
                label_dict[j] = dir
                label.extend(np.ones(self.image_num, dtype=np.int) * j)
        data = np.reshape(data, (self.image_num * j, self.IMAGE_SIZE * self.IMAGE_SIZE))
        return np.matrix(data), np.matrix(label).T, label_dict  # 返回数据、标签及标签对应的姓名

    def svm(self, category_num, train_data, train_label, test_data):
        # 这里的 C=category_num 表示共 category_num 个分类数目
        # print("category_num=%s" % str(category_num))
        clf3 = SVC(C=category_num, gamma="auto")
        clf3.fit(train_data, train_label)
        return clf3.predict(test_data)

    def knn(self, neighbor, train_data, train_label, test_data):
        neigh = KNeighborsClassifier(n_neighbors=neighbor)
        neigh.fit(train_data, train_label)
        return neigh.predict(test_data)

    def text_to_image(self, image, text, position, font_size, font_color):
        img_PIL = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('../fonts/simsun.ttc', font_size)
        draw = ImageDraw.Draw(img_PIL)
        draw.text(position, text, font=font, fill=font_color)
        image = cv2.cvtColor(np.asarray(img_PIL), cv2.COLOR_RGB2BGR)
        return image

    def recognize(self):
        data, label, label_dict = self.load_data()
        # 创建PCA类对象，0.9表示保留了90%的数据方差
        pca = PCA(0.9, True, True)
        # 训练数据
        train_data = pca.fit_transform(data)
        # 边框颜色
        color = (0, 255, 0)
        # 摄像头捕获的实时视频流
        capture = cv2.VideoCapture(0)

        # 开始循环检测识别人脸
        while True:
            # 读取一帧视频
            _, frame = capture.read()
            # 图像灰度化，这是为了降低计算复杂度
            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            # 人脸识别分类器
            cascade = cv2.CascadeClassifier(self.cascade_path)

            # 识别出人脸区域，这里的scaleFactor=1.2和minNeighbors=3分别表示图片缩放比例和需要检测的有效点数
            face_rects = cascade.detectMultiScale(frame_gray, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
            # 检测到了人脸
            if len(face_rects) > 0:
                # 单独框出每一张人脸
                for face_rect in face_rects:
                    x, y, imgcol, imgrow = face_rect

                    # 截取脸部图像提交给模型识别这是谁
                    m = frame_gray[y - 10: y + imgrow + 10, x - 10: x + imgcol + 10]

                    top, bottom, left, right = (0, 0, 0, 0)
                    image = m
                    # 获取图像的行、列尺寸
                    imgrow, imgcol = image.shape

                    # 如果图片的行、列不相等
                    longest_edge = max(imgrow, imgcol)

                    # 行小列大，行上需要增加像素至等长
                    if imgrow < longest_edge:
                        dh = longest_edge - imgrow
                        top = dh // 2
                        bottom = dh - top
                    # 行大列小，列上需要增加像素至等长
                    elif imgcol < longest_edge:
                        dw = longest_edge - imgcol
                        left = dw // 2
                        right = dw - left
                    else:
                        pass

                    # [0]表示黑色
                    constant = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0])
                    # 调整图像大小
                    image = cv2.resize(constant, (self.IMAGE_SIZE, self.IMAGE_SIZE))
                    img_test = np.reshape(image, (1, self.IMAGE_SIZE * self.IMAGE_SIZE))
                    # 降维测试数据
                    test_data = pca.transform(img_test)
                    if self.algorithm == "svm":
                        # 用SVM算法分类，调试后的正确代码
                        result = self.svm(len(label_dict), train_data, label.getA().ravel(), test_data)
                    else:
                        # 用KNN算法分类，其中knn_k是调用端传入的参数，默认为5，为最近邻居数，调试后的正确代码
                        result = self.knn(self.knn_k, train_data, label.getA().ravel(), test_data)
                    result_id = result[0]

                    # 识别出了结果
                    if result_id > 0:
                        cv2.rectangle(frame, (x - 10, y - 10), (x + imgcol + 10, y + imgrow + 10), color, thickness=2)
                        # 将识别出来的人的姓名打印在图像上
                        result_name = label_dict[result_id]
                        frame = self.text_to_image(frame, result_name, (x + 30, y + 30), 30, (255, 0, 255))

            cv2.imshow("识别结果出来了", frame)

            # 等待按键输入，10毫秒
            k = cv2.waitKey(10)
            # 退出循环
            if k & 0xFF == ord("q"):
                break

        # 释放摄像头
        capture.release()
        # 销毁所有窗口
        cv2.destroyAllWindows()



if __name__ == "__main__":
    parent_path = "/Users/leoliu/scikit_learn_data/bxp_faces"
    # cascade_path = "haarcascade_frontalface_alt.xml"
    cascade_path = "haarcascade_frontalface_alt2.xml"
    top = Tk()
    Application(top, parent_path, cascade_path).mainloop()