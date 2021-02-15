import numpy as np
import os
import pydicom
from PIL import Image

# DICOM windowing 파라미터
WINDOW_CENTER = 40 #80
WINDOW_WIDTH = 80 #200

def load_data(dcm_path, save_path):

    # dcm 로드
    dataset = pydicom.read_file(dcm_path)
    
    img = dataset.pixel_array
       
    # windowing 파라미터 획득
    _, _, intercept, slope = get_windowing(dataset)
    
    # rescale 수행
    img = (img * slope + intercept)
    
    # windowing 수행
    img = windowing(img, WINDOW_CENTER, WINDOW_WIDTH)*255.        
    
    if len(img.shape)==2:
        img = np.stack((img,)*3, axis=-1)
    
    img = Image.fromarray(img.astype(np.uint8))
    img.save(save_path)

# DICOM windowing 파라미터 획득 함수
def get_windowing(data):
    dicom_fields = [data[('0028', '1050')].value,  # window center
                    data[('0028', '1051')].value,  # window width
                    data[('0028', '1052')].value,  # intercept
                    data[('0028', '1053')].value]  # slope
    return [get_first_of_dicom_field_as_int(x) for x in dicom_fields]

def get_first_of_dicom_field_as_int(x):
    # get x[0] as in int is x is a 'pydicom.multival.MultiValue', otherwise get int(x)
    if type(x) == pydicom.multival.MultiValue:
        return int(x[0])
    else:
        return int(x)

# DICOM windowing 수행 함수
def windowing(img, window_center, window_width, rescale=True):
    img_min = window_center - window_width // 2
    img_max = window_center + window_width // 2
    img[img < img_min] = img_min
    img[img > img_max] = img_max

    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - img_min) / (img_max - img_min)

    return img

def all_channels_windowing(img, window_center, window_width, rescale=True):
    grey_img = windowing(img, window_center, window_width, rescale) * 3.0
    all_chan_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    all_chan_img[:, :, 2] = np.clip(grey_img, 0.0, 1.0)
    all_chan_img[:, :, 0] = np.clip(grey_img - 1.0, 0.0, 1.0)
    all_chan_img[:, :, 1] = np.clip(grey_img - 2.0, 0.0, 1.0)
    
    return all_chan_img

def gradient_windowing(img, window_center, window_width, rescale=True):
    grey_img = windowing(img, window_center, window_width, rescale)
    rainbow_img = np.zeros((grey_img.shape[0], grey_img.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4 * grey_img - 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4 * grey_img * (grey_img <=0.75), 0,1) + np.clip((-4*grey_img + 4) * (grey_img > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4 * grey_img + 2, 0, 1.0) * (grey_img > 0) * (grey_img <= 1.0)
    
    return rainbow_img    

def bsb_windowing(img, window_center, window_width, rescale=True):
    brain_img = windowing(img, 40, 80, rescale)
    subdural_img = windowing(img, 80, 200, rescale)
    bone_img = windowing(img, 600, 2000, rescale)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    
    return bsb_img    

def gradient_bsb_windowing(img, window_center, window_width, rescale=True):
    brain_img = windowing(img, 40, 80, rescale)
    subdural_img = windowing(img, 80, 200)
    bone_img = windowing(img, 600, 2000)
    combo = (brain_img*0.3 + subdural_img*0.5 + bone_img*0.2)
    
    rainbow_img = np.zeros((combo.shape[0], combo.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4 * combo - 2, 0, 1.0) * (combo > 0) * (combo <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4 * combo * (combo <=0.75), 0,1) + np.clip((-4*combo + 4) * (combo > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4 * combo + 2, 0, 1.0) * (combo > 0) * (combo <= 1.0)
    
    return rainbow_img    

def sigmoid_windowing(img, window_center, window_width, U=1.0, eps=(1.0 / 255.0), rescale=True):
    ue = np.log((U / eps) - 1.0)
    W = (2 / window_width) * ue
    b = ((-2 * window_center) / window_width) * ue
    z = W * img + b
    img = U / (1 + np.power(np.e, -1.0 * z))
    
    if rescale:
        # Extra rescaling to 0-1, not in the original notebook
        img = (img - np.min(img)) / (np.max(img) - np.min(img))
    
    return img

def sigmoid_bsb_windowing(img, window_center, window_width, rescale=True):
    brain_img = sigmoid_windowing(img, 40, 80, rescale)
    subdural_img = sigmoid_windowing(img, 80, 200, rescale)
    bone_img = sigmoid_windowing(img, 600, 2000, rescale)
    
    bsb_img = np.zeros((brain_img.shape[0], brain_img.shape[1], 3))
    bsb_img[:, :, 0] = brain_img
    bsb_img[:, :, 1] = subdural_img
    bsb_img[:, :, 2] = bone_img
    
    return bsb_img

def sigmoid_gradient_bsb_windowing(img, window_center, window_width, rescale=True):
    brain_img = sigmoid_windowing(img, 40, 80, rescale)
    subdural_img = sigmoid_windowing(img, 80, 200, rescale)
    bone_img = sigmoid_windowing(img, 600, 2000, rescale)
    combo = (brain_img*0.35 + subdural_img*0.5 + bone_img*0.15)
    combo_norm = (combo - np.min(combo)) / (np.max(combo) - np.min(combo))    

    rainbow_img = np.zeros((combo_norm.shape[0], combo_norm.shape[1], 3))
    rainbow_img[:, :, 0] = np.clip(4*combo_norm - 2, 0, 1.0) * (combo_norm > 0.01) * (combo_norm <= 1.0)
    rainbow_img[:, :, 1] =  np.clip(4*combo_norm * (combo_norm <=0.75), 0,1) + np.clip((-4*combo_norm + 4) * (combo_norm > 0.75), 0, 1)
    rainbow_img[:, :, 2] = np.clip(-4*combo_norm + 2, 0, 1.0) * (combo_norm > 0.01) * (combo_norm <= 1.0)
    
    return rainbow_img
    
def main():
    rootdir = "D:/hemorrhage_dcm/"
    save_path = "D:/hemorrhage_png/"
    for parent, dirnames, filenames in os.walk(rootdir):
        for filename in filenames:
            if '.dcm' in filename.lower():
                dicom_path = os.path.join(parent, filename)
                #name = dicom_path.replace('\\', '_')[3:]   
                print (filename)
                load_data(dicom_path, save_path+filename+".png")

if __name__ == "__main__":
    main()