import cv2
from math import *
import numpy as np
from recognize.crnn_recognizer import PytorchOcr
from en_recognize import en_index
from ch_recognize import ch_index
from translate import tran
recognizer = PytorchOcr()


def dis(image):
    cv2.imshow('image', image)
    cv2.waitKey(0)


def sort_box(box):
    """
    对box进行排序
    """
    box = sorted(box, key=lambda x: sum([x[1], x[3], x[5], x[7]]))
    return box


def dumpRotateImage(img, degree, pt1, pt2, pt3, pt4):
    height, width = img.shape[:2]
    heightNew = int(width * fabs(sin(radians(degree))) + height * fabs(cos(radians(degree))))
    widthNew = int(height * fabs(sin(radians(degree))) + width * fabs(cos(radians(degree))))
    matRotation = cv2.getRotationMatrix2D((width // 2, height // 2), degree, 1)
    matRotation[0, 2] += (widthNew - width) // 2
    matRotation[1, 2] += (heightNew - height) // 2
    imgRotation = cv2.warpAffine(img, matRotation, (widthNew, heightNew), borderValue=(255, 255, 255))
    pt1 = list(pt1)
    pt3 = list(pt3)

    [[pt1[0]], [pt1[1]]] = np.dot(matRotation, np.array([[pt1[0]], [pt1[1]], [1]]))
    [[pt3[0]], [pt3[1]]] = np.dot(matRotation, np.array([[pt3[0]], [pt3[1]], [1]]))
    ydim, xdim = imgRotation.shape[:2]
    imgOut = imgRotation[max(1, int(pt1[1])): min(ydim - 1, int(pt3[1])),
             max(1, int(pt1[0])): min(xdim - 1, int(pt3[0]))]

    return imgOut


def en_recognize(partimg):
    return en_index.en_recognize(partimg)


def ch_recognize(partimg):
    return ch_index.ch_recognize(partimg)


def change_text_recs(text_recs):
    for rec in text_recs:
        tool_people = rec[2]
        rec[2] = rec[6]
        rec[6] = tool_people
        tool_people = rec[3]
        rec[3] = rec[7]
        rec[7] = tool_people

    return text_recs


def charRec(img, text_recs, language, adjust=False):
    """
    加载OCR模型，进行字符识别
    """
    text = []
    text_recs = change_text_recs(text_recs)

    for index, rec in enumerate(text_recs):
        # 将字符串类型强制转换为float类型
        for i in range(8):
            rec[i] = float(rec[i])
            rec[i] = int(rec[i])

        # 截切图片
        partImg = img[min(rec[1],rec[3]) + 2:max(rec[5],rec[7]) + 2, min(rec[0],rec[6]) + 2:max(rec[2],rec[4]) + 2]

        if partImg.shape[0] < 1 or partImg.shape[1] < 1 or partImg.shape[0] > partImg.shape[1]:  # 过滤异常图片
            continue

        if language == 'en':
            if index==0:
                print("正在进行识别...")
            text.append(en_recognize(partImg))
        else:
            if index==0:
                print("正在进行识别...")
            text.append(ch_recognize(partImg))

    print("快译菜单正在帮您翻译...")
    print("翻译结果如下：")
    for i in text:
        tran.youdao_translate(i)



def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv2.resize(image, dim, interpolation=inter)

    # return the resized image
    return resized
