import torch
from torchvision import transforms
from model import EAST     # 导入模型

import numpy as np
from PIL import Image, ImageDraw

import cfg     # 导入配置文件

from preprocess import resize_image
from nms import nms


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def load_pil(img):
    """convert PIL Image to torch.Tensor
    """
    t = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5),std=(0.5, 0.5, 0.5))])
    return t(img).unsqueeze(0)


def detect(img_path, model, device,pixel_threshold,quiet=True):
    """
        检测函数
    :param img_path:   图片路径
    :param model:    模型
    :param device:   gpu
    :param pixel_threshold:    阈值
    :param quiet:
    :return:   检测到的目标的四个顶点坐标
    """
    text_rects = []
    img = Image.open(img_path)      # 打开图片
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)  # 返回适合模型运算的图片宽高
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')    # 设置宽高
    with torch.no_grad():   # 清除梯度
        east_detect=model(load_pil(img).to(device))
    y = np.squeeze(east_detect.cpu().numpy(), axis=0)   # 降维度   其实y就是预测值了
    y[:3, :, :] = sigmoid(y[:3, :, :])   # 代入sigmoid函数
    cond = np.greater_equal(y[0, :, :], pixel_threshold)  # 返回的是两个array每个位置的>=比较后的布尔值
    activation_pixels = np.where(cond)   # 这是np.where(condition)情况，返回true的各维坐标
    quad_scores, quad_after_nms = nms(y, activation_pixels)   # 经过nms算法排除多余检测，返回score列表
    with Image.open(img_path) as im:    # 异常处理语句
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)    # 为啥重复？
        # 宽高比例的计算
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height

        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()    # 复制图片
        draw = ImageDraw.Draw(im)
        for i, j in zip(activation_pixels[0], activation_pixels[1]):   # zip（） 打包两个列表
            px = (j + 0.5) * cfg.pixel_size
            py = (i + 0.5) * cfg.pixel_size
            line_width, line_color = 1, 'red'
            if y[1, i, j] >= cfg.side_vertex_pixel_threshold:
                if y[2, i, j] < cfg.trunc_threshold:
                    line_width, line_color = 2, 'yellow'
                elif y[2, i, j] >= 1 - cfg.trunc_threshold:
                    line_width, line_color = 2, 'green'
            draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                       (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                       (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                      width=line_width, fill=line_color)
        im.save(img_path + '_act.jpg')
        quad_draw = ImageDraw.Draw(quad_im)
        txt_items = []
        for score, geo, s in zip(quad_scores, quad_after_nms,
                                 range(len(quad_scores))):

            if np.amin(score) > 0:     # 如果score大于零
                quad_draw.line([tuple(geo[0]),
                                tuple(geo[1]),
                                tuple(geo[2]),
                                tuple(geo[3]),
                                tuple(geo[0])], width=2, fill='red')

                rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                txt_item = ','.join(map(str, rescaled_geo_list))
                txt_items.append(txt_item)
            elif not quiet:
                print('quad invalid with vertex num less then 4.')
        quad_im.save(img_path + '_predict.jpg')
        # if cfg.predict_write2txt and len(txt_items) > 0:
        #     with open(img_path[:-4] + '.txt', 'w') as f_txt:
        #         f_txt.writelines(txt_items)
        for txt in txt_items:
            text_rects.append(txt.split(','))
        # print("text_rects is:", text_rects)

    return text_rects


def predict(img_path):
    threshold = float(cfg.pixel_threshold)
    # print(img_path, threshold)
    model_path = r'F:\ocr_demo1\detect\saved_model/mb3_512_model_epoch_535.pth'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EAST().to(device)
    model.load_state_dict(torch.load(model_path))
    model.eval()

    text_rects = detect(img_path, model, device,threshold)

    return text_rects
