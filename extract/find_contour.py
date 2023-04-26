import cv2
import numpy as np
import os
from torchvision import transforms
from torch import tensor
from torchvision.datasets import ImageFolder
from torchvision import transforms
from torch.utils.data import DataLoader
from PIL import Image
IMAGENET_MEAN = tensor([.485, .456, .406])
IMAGENET_STD = tensor([.229, .224, .225])

def get_if_in_roi(box, target_x, target_y):
    in_roi = False
    img_gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    ret, threshold_image = cv2.threshold(
        img_gray, 1, 255, cv2.THRESH_BINARY)[1]

    if threshold_image[target_y][target_x] > 1:
        in_roi = True
    else:
        in_roi = False
    return in_roi


def get_contour_outer(box):
    gray = cv2.cvtColor(box, cv2.COLOR_BGR2GRAY)
    ret, img_bin = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours_outers, hierarchy = cv2.findContours(
        img_bin.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # contours_outers = imutils.grab_contours(contours_outer)

    return contours_outers


def get_contour_centor(contour_outer):
    moments = cv2.moments(contour_outer)
    centor_x = int(moments["m10"] / moments["m00"])
    centor_y = int(moments["m01"] / moments["m00"])
    return centor_x, centor_y


def get_scaled_contour(contour, scale):
    centor_x, centor_y = get_contour_centor(contour)

    centor_norm = contour - [centor_x, centor_y]
    centor_scaled = centor_norm * scale
    centor_scaled = centor_scaled + [centor_x, centor_y]
    centor_scaled = centor_scaled.astype(np.int32)

    return centor_scaled

def get_normal_contour(img):
    # img = cv2.imread(path)
    imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(imgray, 127, 255, 0)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    return contours
    cv2.drawContours(img, contours, -1, (0,255,0), 3)
    cv2.imshow('contour' , img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# def get_contours(path):
def crop_images(mask_path , original_image_path , index):
    transform=transforms.Compose([
                transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),
                # transforms.ToTensor(),
                # transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ])
    mask = cv2.imread(mask_path)
    # original_image_path = '/Users/srijan/Desktop/ind_knn_ad/datasets/road/test/defect/img1250.jpg' 
    original_image = Image.open(original_image_path)
    original_image = transform(original_image)
    # transform = transforms.ToPILImage()
    # original_image = transform(original_image)
    print(type(original_image))
    # original_image.save('/home/srijan/crop/ind_knn_ad/road_res/org.png')
    # print(type(original_image) , original_image.shape)
    img = mask
    print("Mask shape" , img.shape)
    scale = 1.5
    # all_contours = get_scaled_contour(img, scale)
    all_contours = get_normal_contour(img)
    # print(len(all_contours))
    result_image = img.copy()
    # contour_outers = get_contour_outer(contours)
    # print((contours))
    if len(all_contours) == 0:
        return
    contours = all_contours[0]
    # area = cv2.contourArea(contours)
    # print("Area of contour is" , area)
    # centor_x, centor_y = get_contour_centor(contours)
    # print(centor_x , centor_y)
    
    # cv2.drawContours(result_image, [get_scaled_contour(contours, scale)], -1, (255, 255, 0, 100), 2)
    x,y,w,h = cv2.boundingRect(contours)
    original_image = np.array(original_image)
    org_path = f"org_img/org{index}.png"
    
    cv2.rectangle(original_image,(x,y),(x+w,y+h),(255,0,0),3)
    cv2.imwrite(org_path , original_image)
    cropped_image = original_image[y:y+h,x:x+w]
    cropped_folder_path = f"cropped/crop{index}.png"
    cv2.imwrite(cropped_folder_path , cropped_image)

# crop_images('/home/srijan/crop/ind_knn_ad/road_res/mask0.png' , '/home/srijan/crop/ind_knn_ad/datasets/road/test/defect/img699.jpg' , 0)
