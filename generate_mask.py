from contextlib import contextmanager
from io import StringIO
from streamlit.report_thread import REPORT_CONTEXT_ATTR_NAME
from threading import current_thread
import streamlit as st
import sys
from time import sleep
import cv2
from PIL import Image
import io
import numpy as np
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as T
sys.path.append('./indad')
from indad.new_data import MVTecDataset, StreamingDataset
from indad.models import SPADE, PaDiM, PatchCore
from indad.new_data import IMAGENET_MEAN, IMAGENET_STD
from extract.find_contour import *

N_IMAGE_GALLERY = 4
N_PREDICTIONS = 2
METHODS = ["SPADE", "PaDiM", "PatchCore"]
BACKBONES = ["efficientnet_b0", "tf_mobilenetv3_small_100"]

# keep the two smallest datasets
mvtec_classes = ["hazelnut_reduced", "transistor_reduced"]

def tensor_to_img(x, normalize=False):
    if normalize:
        x *= IMAGENET_STD.unsqueeze(-1).unsqueeze(-1)
        x += IMAGENET_MEAN.unsqueeze(-1).unsqueeze(-1)
    x =  x.clip(0.,1.).permute(1,2,0).detach().numpy()
    return x

def pred_to_img(x, range):
    range_min, range_max = range
    x -= range_min
    if (range_max - range_min) > 0:
        x /= (range_max - range_min)
    return tensor_to_img(x)

def show_pred(sample, score, fmap, range , index, original_image_path):
    sample_img = tensor_to_img(sample, normalize=True)
    fmap_img = pred_to_img(fmap, range)

    # overlay
    plt.imshow(sample_img)
    plt.imshow(fmap_img, cmap="jet", alpha=0.5)
    plt.axis('off')
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0, transparent=True)
    buf.seek(0)
    overlay_img = Image.open(buf)
    ls = original_image_path.split('/')
    # print(ls)
    dot_split = ls[-1].split('.')
    if "normal" in ls:
        save_path = f"road_heatmap/normal/normal_{dot_split[0]}.png"
    else:
        save_path = f"road_heatmap/abnormal/abnormal_{dot_split[0]}.png"
    
    overlay_img.save(save_path)
    # actual display
    cols = st.columns(3)
    cols[0].subheader("Test sample")
    cols[0].image(sample_img)
    cols[1].subheader("Anomaly map")
    cols[1].image(fmap_img)
    cols[2].subheader("Overlay")
    cols[2].image(overlay_img)

def get_sample_images(dataset, n):
    n_data = len(dataset)
    ans = []
    if n < n_data:
        indexes = np.random.choice(n_data, n, replace=False)
    else:
        indexes = list(range(n_data))
    for index in indexes:
        sample, _ = dataset[index]
        ans.append(tensor_to_img(sample, normalize=True))
    return ans  
def pytoimg(tensor):
    print(tensor)
    transform = T.ToPILImage()
    img = transform(tensor)
    img.show()

def main():

    app_mvtec_dataset = 'road'
    app_backbone = 'efficientnet_b0'
        # LOAD DATA  
    train_dataset, test_dataset = MVTecDataset(app_mvtec_dataset).get_datasets()
            
        # LOAD MODEL       
    model = PatchCore(
                    f_coreset=.01, 
                    backbone_name=app_backbone,
                    coreset_eps=.95,
                )
        # TRAINING
        # --------
    model.fit(DataLoader(train_dataset))

        # TESTING
        # -------  
    print(len(test_dataset))
    for index in range(len(test_dataset)):
        sample,original_image_path,*_ = test_dataset[index]
        img_lvl_anom_score, pxl_lvl_anom_score = model.predict(sample.unsqueeze(0))
        cpl = pxl_lvl_anom_score
        cpl = cpl.numpy()
        print(img_lvl_anom_score)
        # print(type(cpl))
        # print(cpl.shape)
        # print(cpl)
        ls = original_image_path.split('/')
        # print(ls)
        dot_split = ls[-1].split('.')
        new_img = np.zeros(shape = (224,224))
        for i in range(224):
            for j in range(224):
                if cpl[0][j][i] > 33:
                    new_img[j][i] = 255
        
        mask_path = f"road_res/mask{dot_split[0]}.png"
        
        cv2.imwrite(mask_path , new_img)
        score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
        color_range = score_range
        # print(original_image_path)
        show_pred(sample , img_lvl_anom_score , pxl_lvl_anom_score , color_range , index, original_image_path)
        crop_images(mask_path , original_image_path , index)
    # print(type(sample))
    
    print("Success!")
    # print(sample.dtype)
    # pytoimg(sample)
    # print(img_lvl_anom_score)
    # sample_img = (sample.cpu().numpy())
    # print(sample.size())
    # pytoimg(sample)
    # print("Sample type " , (sample_img.shape))
    # smg = Image.fromarray(sample_img , 'RGB')
    # smg.save('test_sample.png')
    # print("sample size " , sample.size())
    # print("type is" , type(pxl_lvl_anom_score))
    # print("Len is " , (pxl_lvl_anom_score.size()))
    # cpl = pxl_lvl_anom_score
    # cpl = cpl.numpy()
    # print(type(cpl))
    # print(cpl.shape)
    # print(cpl)
    # new_img = np.zeros(shape = (224,224))
    # for i in range(224):
    #     for j in range(224):
    #         if cpl[0][j][i] > img_lvl_anom_score:
    #             new_img[j][i] = 255
    
    # cv2.imwrite('new_img_rev1.png' , cpl  )
    
    # score_range = pxl_lvl_anom_score.min(), pxl_lvl_anom_score.max()
    # show_pred(sample, img_lvl_anom_score, pxl_lvl_anom_score, color_range)

    
if __name__ == "__main__":
    main()
