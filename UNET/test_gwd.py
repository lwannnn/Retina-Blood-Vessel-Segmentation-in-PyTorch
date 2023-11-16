import os, time
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
from operator import add
import numpy as np
from glob import glob
import cv2
from tqdm import tqdm
import imageio
import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
from GWDistance_2d import GraphWassersteinDistance
from model import build_unet
from FCN import FCN
from utils import create_dir, seeding
from medpy.metric.binary import dc, hd95
from cldice_metric import *

def picture_to_patch(tensor):
    patch_size =256
    batch_size, channels, height, width = tensor.size()
    patches = tensor.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    patches = patches.contiguous().view(batch_size, channels, -1, patch_size, patch_size)# patches的维度为（batch_size，channels，16，128，128）
    return patches

def calculate_gwd(y_true, y_pred):
    gwd = 0
    gwd_loss = GraphWassersteinDistance()
    pred_patch = picture_to_patch(y_pred)
    gt_patch = picture_to_patch(y_true)
    for i in range(pred_patch.size(2)):
        gwd += gwd_loss(pred_patch[:,:,i,...],gt_patch[:,:,i,...])


    print(gwd / pred_patch.size(2))
    return gwd/ pred_patch.size(2)


def calculate_metrics(y_true, y_pred):
    gwd = calculate_gwd(y_true, y_pred)
    hausdorff_distance95 = hd95(((y_pred>0.5)*1).cpu().numpy(), y_true.cpu().numpy())
    cldice_ = clDice(((y_pred.squeeze()>0.5)*1).cpu().numpy(), y_true.squeeze().cpu().numpy())
    dice = dc(((y_pred > 0.5) * 1).cpu().numpy(), y_true.cpu().numpy())
    """ Ground truth """
    y_true = y_true.cpu().numpy()
    y_true = y_true > 0.5
    y_true = y_true.astype(np.uint8)
    y_true = y_true.reshape(-1)

    """ Prediction """
    y_pred = y_pred.cpu().numpy()
    y_pred = y_pred > 0.5
    y_pred = y_pred.astype(np.uint8)
    y_pred = y_pred.reshape(-1)

    score_jaccard = jaccard_score(y_true, y_pred)
    score_f1 = f1_score(y_true, y_pred)
    score_recall = recall_score(y_true, y_pred)
    score_precision = precision_score(y_true, y_pred)
    score_acc = accuracy_score(y_true, y_pred)

    return [score_jaccard, score_f1, score_recall, score_precision, score_acc , gwd,hausdorff_distance95,dice,cldice_]

def mask_parse(mask):
    mask = np.expand_dims(mask, axis=-1)    ## (512, 512, 1)
    mask = np.concatenate([mask, mask, mask], axis=-1)  ## (512, 512, 3)
    return mask

if __name__ == "__main__":
    """ Seeding """
    seeding(42)

    """ Folders """
    create_dir("results")

    """ Load dataset """
    test_x = sorted(glob("../new_data/test/image/*"))
    test_y = sorted(glob("../new_data/test/mask/*"))

    """ Hyperparameters """
    H = 512
    W = 512
    size = (W, H)
    checkpoint_path = "files/202310231618.pth"

    """ Load the checkpoint """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = build_unet()
    # model = FCN()
    model = model.to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    metrics_score = [0.0, 0.0, 0.0, 0.0, 0.0 ,0.0,0.0,0.0,0.0]
    time_taken = []

    for i, (x, y) in tqdm(enumerate(zip(test_x, test_y)), total=len(test_x)):
        """ Extract the name """
        name = x.split("/")[-1].split(".")[0]
        print(name)
        """ Reading image """
        image = cv2.imread(x, cv2.IMREAD_COLOR) ## (512, 512, 3)
        ## image = cv2.resize(image, size)
        x = np.transpose(image, (2, 0, 1))      ## (3, 512, 512)
        x = x/255.0
        x = np.expand_dims(x, axis=0)           ## (1, 3, 512, 512)
        x = x.astype(np.float32)
        x = torch.from_numpy(x)
        x = x.to(device)

        """ Reading mask """
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)  ## (512, 512)
        ## mask = cv2.resize(mask, size)
        y = np.expand_dims(mask, axis=0)            ## (1, 512, 512)
        y = y/255.0
        y = np.expand_dims(y, axis=0)               ## (1, 1, 512, 512)
        y = y.astype(np.float32)
        y = torch.from_numpy(y)
        y = y.to(device)

        with torch.no_grad():

            """ Prediction and Calculating FPS """
            start_time = time.time()
            pred_y = model(x)
            pred_y = torch.sigmoid(pred_y)
            total_time = time.time() - start_time
            time_taken.append(total_time)


            score = calculate_metrics(y, pred_y)
            metrics_score = list(map(add, metrics_score, score))
            pred_y = pred_y[0].cpu().numpy()        ## (1, 512, 512)
            pred_y = np.squeeze(pred_y, axis=0)     ## (512, 512)
            pred_y = pred_y > 0.5
            pred_y = np.array(pred_y, dtype=np.uint8)

        """ Saving masks """
        ori_mask = mask_parse(mask)
        pred_y = mask_parse(pred_y)
        line = np.ones((size[1], 10, 3)) * 128

        cat_images = np.concatenate(
            [image, line, ori_mask, line, pred_y * 255], axis=1
        )
        cv2.imwrite(f"results/{name}.png", cat_images)

    jaccard = metrics_score[0]/len(test_x)
    f1 = metrics_score[1]/len(test_x)
    recall = metrics_score[2]/len(test_x)
    precision = metrics_score[3]/len(test_x)
    acc = metrics_score[4]/len(test_x)
    gwd_score = metrics_score[5] / len(test_x)
    hd95_score = metrics_score[6] / len(test_x)
    dice_score = metrics_score[7] / len(test_x)
    cldice_score =metrics_score[8] / len(test_x)
    print(f"Jaccard: {jaccard:1.4f} - F1: {f1:1.4f} - Recall: {recall:1.4f} - Precision: {precision:1.4f} - Acc: {acc:1.4f} - Gwd : {gwd_score:1.4f} - hd95 :{hd95_score:1.4f} - dice:{dice_score:1.4f}"
          f"cldice_score: {cldice_score:1.4f}" )

    fps = 1/np.mean(time_taken)
    print("FPS: ", fps)
