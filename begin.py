import tkinter as tk
from tkinter.filedialog import *
from PIL import Image, ImageTk
from tkinter import messagebox
from common import charRec,resize
from detect import predict
import numpy as np


window = tk.Tk()
window.title('快译菜单')
window.geometry('400x400')


def index():
    global l
    global b1
    l = tk.Label(window, text='Welcome!', font=('Arial', 12), width=15,
                 height=2)
    l.pack()

    b1 = tk.Button(window, text='开始识别', width=15, height=2, command=start)
    b1.place(x=140, y=150)
    print(window)


def notify():
    tk.messagebox.showinfo(title='识别成功！', message='图片和文档放在test_result文件中')


def start():
    global image_file
    l.destroy()
    b1.destroy()
    tip = tk.Label(window, text='请找到你要识别的图片', font=('Arial', 12), width=20,
                   height=2)
    tip.pack()
    fd = LoadFileDialog(window)  # 创建打开文件对话框
    filename = fd.go()  # 显示打开文件对话框，并获取选择的文件名称
    file = os.path.split(filename)[-1]
    file = os.path.join('./test_images/', file)
    print(file)
    pilImage = Image.open(file)
    image = np.array(Image.open(file).convert('RGB'))

    # 识别
    tran = input("英文菜单输入“en”，中文菜单输入“ch”：")
    text_recs = predict.predict(file)
    charRec(image, text_recs, language=tran)

    pilImage = pilImage.resize((400, 400), Image.ANTIALIAS)
    image_file = ImageTk.PhotoImage(image=pilImage)
    label = tk.Label(window, image=image_file)
    label.pack()

    b2 = tk.Button(window, text='下一步', width=15, height=2, command=notify)
    b2.place(x=280, y=350)


if __name__ == '__main__':
    index()
    window.mainloop()
