import os
import torch
import numpy as np
import pydicom
from dcm import get_windowing

from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import cv2
from skimage.measure import label, regionprops
import PIL.Image as Image

class AISDataset(Dataset):
    def __init__(self, dicom_dir, bboxes, windowing, window_center, window_width, transform=None):
        self.dicom_path = dicom_dir
        self.bboxes = bboxes
        self.transform = transform
        self.windowing = windowing
        self.window_center = window_center
        self.window_width = window_width
        
    def __getitem__(self, index):
        # Load Image
        dicom_fn = self.dicom_path[index]
        bboxes = self.bboxes[index]
        windowing = self.windowing
        wc = self.window_center
        ww = self.window_width
        
        dataset = pydicom.read_file(dicom_fn)
        img = dataset.pixel_array
           
        # windowing 파라미터 획득
        _, _, intercept, slope = get_windowing(dataset)
        
        # rescale 수행
        img = (img * slope + intercept)
        
        # windowing 수행
        img = windowing(img, wc, ww)        
        
        img = img.astype(np.float32)

        if len(img.shape)==2:
            img = np.stack((img,)*3, axis=-1)
        
        sample = {'img': img, 'annot': bboxes}
        
        if self.transform:
            sample = self.transform(sample)
        return sample
                                              
    def __len__(self):
        return len(self.dicom_path)

def get_bbox(mask, cl_num):
    annotations = np.zeros((1, 5))
    # some images appear to miss annotations
    if mask.max() == 0:
        return annotations
    for cl in range(0, cl_num):
        mask_cl = np.where(mask==cl+1,1,0)
        mask_cl = label(mask_cl)
        props = regionprops(mask_cl)
        
        for i in range(0,len(props)):
            yl, xl, yr, xr = [int(b) for b in props[i].bbox]
            annotation = np.array([[xl,yl,xr,yr,cl]])
            annotations = np.append(annotations, annotation, axis=0)
        
    return annotations[1:]

class HmDataset(Dataset):
    def __init__(self, image_dir, mask_dir, len_cls, transform=None):
        self.image_path = image_dir
        self.mask_path = mask_dir
        self.transform = transform
        self.len_cls = len_cls
        
    def __getitem__(self, index):
        # Load Image
        img_fn = self.image_path[index]
        mask_fn = self.mask_path[index]
        
        img = cv2.imread(img_fn)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img.astype(np.float32) / 255.
        mask = np.array(Image.open(mask_fn))
        
        annotations = get_bbox(mask, self.len_cls)
                
        sample = {'img': img, 'annot': annotations}
        if self.transform:
            sample = self.transform(sample)
        return sample
                                              
    def __len__(self):
        return len(self.image_path)

class CocoDataset(Dataset):
    def __init__(self, root_dir, set='train2017', transform=None):

        self.root_dir = root_dir
        self.set_name = set
        self.transform = transform

        self.coco = COCO(os.path.join(self.root_dir, 'annotations', 'instances_' + self.set_name + '.json'))
        self.image_ids = self.coco.getImgIds()

        self.load_classes()

    def load_classes(self):

        # load class names (name -> label)
        categories = self.coco.loadCats(self.coco.getCatIds())
        categories.sort(key=lambda x: x['id'])

        self.classes = {}
        for c in categories:
            self.classes[c['name']] = len(self.classes)

        # also load the reverse (label -> name)
        self.labels = {}
        for key, value in self.classes.items():
            self.labels[value] = key

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):

        img = self.load_image(idx)
        annot = self.load_annotations(idx)
        sample = {'img': img, 'annot': annot}
        if self.transform:
            sample = self.transform(sample)
        return sample

    def load_image(self, image_index):
        image_info = self.coco.loadImgs(self.image_ids[image_index])[0]
        path = os.path.join(self.root_dir, self.set_name, image_info['file_name'])
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        return img.astype(np.float32) / 255.

    def load_annotations(self, image_index):
        # get ground truth annotations
        annotations_ids = self.coco.getAnnIds(imgIds=self.image_ids[image_index], iscrowd=False)
        annotations = np.zeros((0, 5))

        # some images appear to miss annotations
        if len(annotations_ids) == 0:
            return annotations

        # parse annotations
        coco_annotations = self.coco.loadAnns(annotations_ids)
        for idx, a in enumerate(coco_annotations):

            # some annotations have basically no width / height, skip them
            if a['bbox'][2] < 1 or a['bbox'][3] < 1:
                continue

            annotation = np.zeros((1, 5))
            annotation[0, :4] = a['bbox']
            annotation[0, 4] = a['category_id'] - 1
            annotations = np.append(annotations, annotation, axis=0)

        # transform from [x, y, w, h] to [x1, y1, x2, y2]
        annotations[:, 2] = annotations[:, 0] + annotations[:, 2]
        annotations[:, 3] = annotations[:, 1] + annotations[:, 3]

        return annotations


def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    imgs = torch.from_numpy(np.stack(imgs, axis=0))

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:

        annot_padded = torch.ones((len(annots), max_num_annots, 5)) * -1

        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = torch.ones((len(annots), 1, 5)) * -1

    imgs = imgs.permute(0, 3, 1, 2)

    return {'img': imgs, 'annot': annot_padded, 'scale': scales}


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""
    
    def __init__(self, img_size=512):
        self.img_size = img_size

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        height, width, _ = image.shape
        if height > width:
            scale = self.img_size / height
            resized_height = self.img_size
            resized_width = int(width * scale)
        else:
            scale = self.img_size / width
            resized_height = int(height * scale)
            resized_width = self.img_size

        image = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)

        new_image = np.zeros((self.img_size, self.img_size, 3))
        new_image[0:resized_height, 0:resized_width] = image

        annots[:, :4] *= scale

        return {'img': torch.from_numpy(new_image).to(torch.float32), 'annot': torch.from_numpy(annots), 'scale': scale}


class Flip(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            new_bbox = annots.copy()
            
            flip_image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            new_bbox[:, 0] = cols - x2
            new_bbox[:, 2] = cols - x_tmp

            sample = {'img': flip_image, 'annot': new_bbox}

        return sample

class Rotate(object):

    def __init__(self, angle=10):
        self.angle = angle
        
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        angle = np.random.randint(-self.angle, self.angle)
        rotation_angle = angle*np.pi/180
        rot_matrix = np.array(
                    [[np.cos(rotation_angle), -np.sin(rotation_angle)], [np.sin(rotation_angle), np.cos(rotation_angle)]])        
        
        height, width, channel = image.shape
        temp_image = image
        matrix = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        image = cv2.warpAffine(image, matrix, (width, height))
        for i in range(height):
            for j in range(width):
                if image[i,j,0]==0 and image[i,j,1]==0 and image[i,j,2]==0:
                    image[i,j]=temp_image[i,j]
        
        new_bbox = np.zeros((0, 5))

        for i in range(len(annots)):
            xlu, ylu, xrd, yrd = annots[i][0], annots[i][1], annots[i][2], annots[i][3]
            xld, yld, xru, yru = annots[i][0], annots[i][3], annots[i][2], annots[i][1]
            
            nc_xlu, nc_ylu = np.matmul(rot_matrix, np.array((xlu-(width/2), -ylu+(height/2))))
            nc_xrd, nc_yrd = np.matmul(rot_matrix, np.array((xrd-(width/2), -yrd+(height/2))))
            nc_xld, nc_yld = np.matmul(rot_matrix, np.array((xld-(width/2), -yld+(height/2))))
            nc_xru, nc_yru = np.matmul(rot_matrix, np.array((xru-(width/2), -yru+(height/2))))
            
            r_xlu, r_ylu, r_xrd, r_yrd = nc_xlu+(width/2), -nc_ylu+(height/2), nc_xrd+(width/2), -nc_yrd+(height/2)
            r_xld, r_yld, r_xru, r_yru = nc_xld+(width/2), -nc_yld+(height/2), nc_xru+(width/2), -nc_yru+(height/2)
            
            n_xlu, n_ylu, n_xrd, n_yrd = min(r_xlu,r_xrd, r_xld, r_xru), min(r_ylu, r_yrd, r_yld, r_yru),\
            max(r_xlu,r_xrd, r_xld, r_xru), max(r_ylu, r_yrd, r_yld, r_yru)

            n_xlu = max(0, n_xlu)
            n_ylu = max(0, n_ylu)
            n_xrd = min(width-1, n_xrd)
            n_yrd = min(height-1, n_yrd)

            annotation = np.array([[n_xlu, n_ylu, n_xrd, n_yrd, annots[i][4]]])
        
            new_bbox = np.append(new_bbox, annotation, axis=0)     
        
        sample = {'img': image, 'annot': new_bbox}

        return sample

class Zoom(object):

    def __init__(self, zoom=(0.4,1.6)):
        self.zoom = zoom
        
    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']
        zoom = np.random.randint(self.zoom[0]*10, (self.zoom[1])*10)/10

        height, width, channel = image.shape
        r_height, r_width = int(height*zoom), int(width*zoom)
        resize_image = cv2.resize(image, (r_height, r_width))
        
        zoom_image = np.full((height,width,channel), (image[:,:,0].min(),image[:,:,1].min(),image[:,:,2].min()), dtype=np.float64)
        margin_height = abs(int((r_height-height)/2))
        margin_width = abs(int((r_width-width)/2))
        
        if zoom <= 1:
            zoom_image[margin_height:margin_height+r_height, margin_width:margin_width+r_width] = resize_image
        else:
            zoom_image = resize_image[margin_height:margin_height+height, margin_width:margin_width+width]
            
        new_bbox = np.zeros((0, 5))
        
        for i in range(len(annots)):
            xl, yl, xr, yr = annots[i][0], annots[i][1], annots[i][2], annots[i][3]
            nc_xl, nc_yl, nc_xr, nc_yr = xl-(width/2), -yl+(height/2), xr-(width/2), -yr+(height/2)
            znc_xl, znc_yl, znc_xr, znc_yr = int(nc_xl*zoom), int(nc_yl*zoom), int(nc_xr*zoom), int(nc_yr*zoom)
                                                                                                                          
            n_xl, n_yl, n_xr, n_yr = znc_xl+(width/2), -znc_yl+(height/2), znc_xr+(width/2), -znc_yr+(height/2)
            
            n_xl = max(0, n_xl)
            n_yl = max(0, n_yl)
            n_xr = min(width-1, n_xr)
            n_yr = min(height-1, n_yr)
            
            annotation = np.array([[n_xl, n_yl, n_xr, n_yr, annots[i][4]]])
            
            new_bbox = np.append(new_bbox, annotation, axis=0)       
        
        sample = {'img': zoom_image, 'annot': new_bbox}

        return sample
   
class Normalizer(object):

    def __init__(self, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
        self.mean = np.array([[mean]])
        self.std = np.array([[std]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}
