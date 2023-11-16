from PIL import Image
import numpy as np
from skimage.morphology import skeletonize
from scipy.ndimage import binary_dilation
from gwd_numpy import *
import torch
from medpy.metric.binary import dc, hd95
from scipy.ndimage import rotate
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from UNET.cldice_metric import *

from skimage import io, data
import math

# 建立灰度图旋转函数
def image_rotate(img, degrees=0, x0=0, y0=0, resize=False):
    """
    :param img:
    :param degrees:
    :param x0: 这个是图像的宽
    :param y0: 这个是图像的高
    :param resize:
    :return:
    """
    # 建立旋转需要的全部矩阵
    # 将角度转为弧度和正弦余弦值
    radians = math.radians(degrees)
    sinA = math.sin(radians)
    cosA = math.cos(radians)
    # 矩阵坐标转笛卡尔坐标并旋转再转回矩阵坐标
    mat2dika2mat = np.mat([[1, 0, 0], [0, -1, 0], [-x0, y0, 1]]) * \
                   np.mat([[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]]) * \
                   np.mat([[1, 0, 0], [0, -1, 0], [x0, y0, 1]])
    # 矩阵坐标转笛卡尔坐标并旋转再转回矩阵坐标(这是用于反向转回的矩阵)
    mat2dika2mat_r = np.mat([[1, 0, 0], [0, -1, 0], [-x0, y0, 1]]) * \
                     np.mat([[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]]).I * \
                     np.mat([[1, 0, 0], [0, -1, 0], [x0, y0, 1]])

    # 给旋转后的图像大小一个初始值
    img_h, img_w = img.shape[:2]
    if len(img.shape) == 2:
        img_d = 0
    else:
        img_d = img.shape[2]
    # 首先求出旋转之后图像的大小
    if resize:
        # 将四个顶点分别进行计算旋转后的位置，以此确定旋转后图像多大
        # 注意：设定好的矩阵的运算中，横着向右代表矩阵坐标的横轴正方形，竖着向下代表矩阵坐标的竖轴正方向
        tops = np.mat([[0, 0, 1],
                       [0, img_h - 1, 1],
                       [img_w - 1, 0, 1],
                       [img_w - 1, img_h - 1, 1]])
        # 四个点分别进行旋转
        for i in range(tops.shape[0]):
            tops[i] = tops[i] * mat2dika2mat
        # 取出行和列的最大最小值
        max_hw = tops.max(axis=0)
        min_hw = tops.min(axis=0)
        max_w, max_h = max_hw[0, 0], max_hw[0, 1]
        min_w, min_h = min_hw[0, 0], min_hw[0, 1]
        # 设置旋转后图像的大小
        img_h, img_w = abs(int(max_h - min_h)), abs(int(max_w - min_w))
        # 重新建立新旋转中心的运算矩阵
        # 新的旋转中心找到之后，只需要替换旋转时从笛卡尔坐标系转换为矩阵坐标系那个公式中的旋转中心坐标
        x0_ = x0 * img_w // img.shape[1]
        y0_ = y0 * img_h // img.shape[0]
        # 矩阵坐标转笛卡尔坐标并旋转再转回矩阵坐标
        mat2dika2mat = np.mat([[1, 0, 0], [0, -1, 0], [-x0, y0, 1]]) * \
                       np.mat([[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]]) * \
                       np.mat([[1, 0, 0], [0, -1, 0], [x0_, y0_, 1]])
        # 矩阵坐标转笛卡尔坐标并旋转再转回矩阵坐标(这是用于反向转回的矩阵)
        mat2dika2mat_r = np.mat([[1, 0, 0], [0, -1, 0], [-x0_, y0_, 1]]) * \
                         np.mat([[cosA, -sinA, 0], [sinA, cosA, 0], [0, 0, 1]]).I * \
                         np.mat([[1, 0, 0], [0, -1, 0], [x0, y0, 1]])
    # 建立空白图像
    if img_d:
        img_new = np.zeros((img_h , img_w , img_d), dtype='uint8')#原本hw都+1,被我删掉了
    else:
        img_new = np.zeros((img_h , img_w ), dtype='uint8')#原本hw都+1,被我删掉了
    # 循环旋转每一个像素点
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            temp = np.mat([j, i, 1]) * mat2dika2mat
            x, y, t = temp[0, 0], temp[0, 1], temp[0, 2]
            x = int(x)
            y = int(y)
            if x > img_new.shape[1] - 1 or x < 0 or y > img_new.shape[0] - 1 or y < 0:
                continue
            img_new[y, x] = img[i, j]
    # 对全部的值进行双线性插值运算
    for i in range(img_new.shape[0]):
        for j in range(img_new.shape[1]):
            # if img[i, j] != 0:
            #     continue
            temp = np.mat([j, i, 1]) * mat2dika2mat_r
            x, y, t = temp[0, 0], temp[0, 1], temp[0, 2]
            if x > img.shape[1] - 1 or x < 0 or y > img.shape[0] - 1 or y < 0:
                continue
            n = int(x)
            m = int(y)
            u = x - n
            v = y - m
            if n >= img.shape[1] - 1:
                n = img.shape[1] - 2
            if m >= img.shape[0] - 1:
                m = img.shape[0] - 2
            img_new[i, j] = (1 - v) * (1 - u) * img[m, n] + (1 - v) * u * img[m, n + 1] + v * (1 - u) * img[
                m + 1, n] + v * u * img[m + 1, n + 1]
    # 返回旋转完成的图像
    return img_new

def picture_to_patch(tensor):
    patch_size =256
    batch_size, channels, height, width = tensor.size()
    patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)# patches的维度为（batch_size，channels，16，128，128）
    return patches

def get_gwd_from_patch(y_true, y_pred):
    gwd = 0
    gwd_loss = GraphWassersteinDistance()
    pred_patch = picture_to_patch(torch.from_numpy(y_pred).cuda().unsqueeze(0).unsqueeze(1)).squeeze()
    gt_patch = picture_to_patch(torch.from_numpy(y_true).cuda().unsqueeze(0).unsqueeze(1)).squeeze()
    for i in range(pred_patch.size(0)):
        gwd += gwd_loss(pred_patch[i,...].to(torch.int),gt_patch[i,...].to(torch.int))

    return gwd/ pred_patch.size(0)

# 定义扰动函数，模拟断裂血管
def introduce_vessel_disruption(label_array, disruption_probability=0.1):
    # disruption_probability表示断裂的概率
    disrupted_label_array = np.copy(label_array)
    for i in range(label_array.shape[0]):
        for j in range(label_array.shape[1]):
            if label_array[i, j] == 1:  # 血管区域的像素值通常为1
                # 以disruption_probability的概率将血管像素值置为0，模拟断裂
                if np.random.rand() < disruption_probability:
                    disrupted_label_array[i, j] = 0
    return disrupted_label_array

def shift_vessel(label_array,shift_step=1):
    offset_array = np.roll(label_array, shift=shift_step, axis=1)
    return offset_array


def apply_random_mask(image, mask_size, num_patches):
    masked_image = np.copy(image)
    height, width = image.shape

    for _ in range(num_patches):
        # 随机选择patch的左上角坐标
        top_left_x = np.random.randint(0, width - mask_size)
        top_left_y = np.random.randint(0, height - mask_size)

        # 将选定的patch区域遮掩（设置为0）
        masked_image[top_left_y:top_left_y + mask_size, top_left_x:top_left_x + mask_size] = 0

    return masked_image

def rotate_vessel(array_data,angle):

    # 将数组逆时针旋转5度，指定旋转中心点
    rotated_array = rotate(array_data, angle=angle, axes=(1, 0), reshape=False)

    # 打印旋转后的数组
    return rotated_array

def processing(original_label_path = "./new_data/test/mask/06_test_0.png"):
    original_label = Image.open(original_label_path)
    original_label_array = np.array(original_label)
    skeleton = skeletonize(original_label_array)
    dilated_skeleton = binary_dilation(skeleton)
    p = Image.fromarray(skeleton)
    p.save("./new_data/disturb/original_skeleton.png")
    dp = Image.fromarray(dilated_skeleton)
    dp.save("./new_data/disturb/dilated_skeleton.png")
    for i in range(10):# 断裂血管
        disrupted_label_array = introduce_vessel_disruption(skeleton, disruption_probability=(i+1)/11)
        disrupted_label = Image.fromarray(disrupted_label_array)
        dilated_label = binary_dilation(disrupted_label)
        dilated_skeleton_img = Image.fromarray(dilated_label)
        disrupted_label_path = "./new_data/disturb/disrupted/disrupted_label_"+str(i)+".png"
        disrupted_label.save(disrupted_label_path)

        disrupted_label_path = "./new_data/disturb/disrupted/disrupted_dilated_label_" + str(i) + ".png"
        dilated_skeleton_img.save(disrupted_label_path)
    #
    # for i in range(20):# 偏移血管
    #     shift_label_array = shift_vessel(skeleton, shift_step=i*2)
    #     shift_label = Image.fromarray(shift_label_array)
    #     dilated_label = binary_dilation(shift_label)
    #     dilated_skeleton_img = Image.fromarray(dilated_label)
    #     shift_label_path = "./new_data/disturb/shift/shifted_label_"+str(i)+".png"
    #     shift_label.save(shift_label_path)
    #
    #     shift_label_path = "./new_data/disturb/shift/shift_dilated_label_" + str(i) + ".png"
    #     dilated_skeleton_img.save(shift_label_path)
    #
    # mask_size = 32
    # mask_label = skeleton
    # for i in range(20):# 遮蔽血管
    #     mask_label = apply_random_mask(mask_label, mask_size=mask_size,num_patches=i)
    #     shift_label = Image.fromarray(mask_label)
    #     dilated_label = binary_dilation(shift_label)
    #     dilated_skeleton_img = Image.fromarray(dilated_label)
    #     shift_label_path = "./new_data/disturb/masked/masked_label_"+str(i)+".png"
    #     shift_label.save(shift_label_path)
    #
    #     shift_label_path = "./new_data/disturb/masked/masked_dilated_label_" + str(i) + ".png"
    #     dilated_skeleton_img.save(shift_label_path)

    # image = skeleton * 255
    # for i in range(19):  # 旋转血管
    #     shift_label_array = image_rotate(image, (i-9), image.shape[1] // 2, image.shape[0] // 2, False)
    #     shift_label = Image.fromarray(shift_label_array>0)
    #
    #     dilated_label = binary_dilation(shift_label)
    #     dilated_skeleton_img = Image.fromarray(dilated_label)
    #     shift_label_path = "./new_data/disturb/rotate/rotate_label_" + str(i) + ".png"
    #     shift_label.save(shift_label_path)
    #     shift_label_path = "./new_data/disturb/rotate/rotate_dilated_label_" + str(i) + ".png"
    #     dilated_skeleton_img.save(shift_label_path)

if __name__ == '__main__':
    gwd = GraphWassersteinDistance()
    original_label_path = "./new_data/test/mask/06_test_0.png"

    # processing(original_label_path)

    original_label = Image.open("./new_data/disturb/original_skeleton.png")
    skeleton = np.array(original_label)
    dilated_skeleton =Image.open("./new_data/disturb/dilated_skeleton.png")
    dilated_skeleton = np.array(dilated_skeleton)

    # print('断裂')
    # for i in range(10):# 断裂血管
    #     disrupted_label_path = "./new_data/disturb/disrupted/disrupted_label_"+str(i)+".png"
    #     disrupted_label_array = np.array(Image.open(disrupted_label_path))
    #     disrupted_label_path = "./new_data/disturb/disrupted/disrupted_dilated_label_" + str(i) + ".png"
    #     dilated_label = np.array(Image.open(disrupted_label_path))
    #
    #     print(f"{get_gwd_from_patch(disrupted_label_array,skeleton):1.4f}")
        # print(f"{hd95(disrupted_label_array,skeleton):1.4f}")
        # print(f"{dc(dilated_skeleton,dilated_label):1.4f}" )
        # print(f"{clDice_dilated(dilated_label,disrupted_label_array,dilated_skeleton,skeleton):1.4f}")
        # s = skeleton.reshape(-1)
        # p = disrupted_label_array.reshape(-1)
        # print(f"{recall_score(p,s):1.4f}")#s,p断裂，p,s过度连接

    # print('偏移')
    # for i in range(20):# 偏移血管
    #     shift_label_path = "./new_data/disturb/shift/shifted_label_"+str(i)+".png"
    #     shift_label_array =  np.array(Image.open(shift_label_path))
    #     shift_label_path = "./new_data/disturb/shift/shift_dilated_label_" + str(i) + ".png"
    #     dilated_label = np.array(Image.open(shift_label_path))

        # print(f"{get_gwd_from_patch(shift_label_array,skeleton):1.4f}")
        # print(f"{hd95(dilated_label,dilated_skeleton):1.4f}")
        # print(f"{dc(shift_label_array, skeleton):1.4f}" )
        # print(f"{clDice_dilated(dilated_label, shift_label_array, dilated_skeleton, skeleton):1.4f}")
        # s = dilated_skeleton.astype(np.uint8).reshape(-1)
        # p = dilated_label.astype(np.uint8).reshape(-1)
        # print(f"{recall_score(s,p):1.4f}")#s,p偏移

    print('遮蔽')
    for i in range(20):# 遮蔽血管
        shift_label_path = "./new_data/disturb/masked/masked_label_"+str(i)+".png"
        mask_label = np.array(Image.open(shift_label_path))
        shift_label_path = "./new_data/disturb/masked/masked_dilated_label_" + str(i) + ".png"
        dilated_label = np.array(Image.open(shift_label_path))
        print(f"{get_gwd_from_patch(mask_label,skeleton):1.4f}")
    #     print(f"{hd95(mask_label,skeleton):1.4f}")
        # print(f"{dc(mask_label,skeleton):1.4f}" )
        # print(f"{clDice_dilated(dilated_label, mask_label,dilated_skeleton, skeleton):1.4f}")
        # s = skeleton.astype(np.uint8).reshape(-1)
        # p = mask_label.astype(np.uint8).reshape(-1)
        # print(f"{jaccard_score(s,p):1.4f}")#p,s过多；s,p遮蔽

    # print('旋转')
    # for i in range(19):  # 旋转血管
    #     shift_label_path = "./new_data/disturb/rotate/rotate_label_" + str(i) + ".png"
    #     shift_label_array = np.array(Image.open(shift_label_path))
    #     shift_label_path = "./new_data/disturb/rotate/rotate_dilated_label_" + str(i) + ".png"
    #     dilated_label =  np.array(Image.open(shift_label_path))
    #     # print(f"{get_gwd_from_patch(shift_label_array,skeleton):1.4f}")
    #     # print(f"{hd95(dilated_label,dilated_skeleton):1.4f}")
    #     print(f"{dc(shift_label_array, skeleton):1.4f}")
    #     print(f"{clDice_dilated(dilated_label, shift_label_array, dilated_skeleton, skeleton):1.4f}")
    #     s = skeleton.astype(np.uint8).reshape(-1)
    #     p = shift_label_array.astype(np.uint8).reshape(-1)
    #     print(f"{jaccard_score(p,s):1.4f}")




