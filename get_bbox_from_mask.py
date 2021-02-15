import numpy as np
import cv2
from glob import glob
import yaml
import os
from skimage.measure import label, regionprops
import csv

project_name = 'hemorrhage_binary'

class Params:
    def __init__(self, project_file):
        self.params = yaml.safe_load(open(project_file).read())

    def __getattr__(self, item):
        return self.params.get(item, None)

params = Params('./projects/'+project_name+'.yml')

def get_bbox(mask, cl_num):
    annotations = np.zeros((1, 5))
    for cl in range(1, cl_num+1):
        mask_cl = np.where(mask==cl,1,0)
        mask_cl = label(mask_cl)
        props = regionprops(mask_cl)
        
        for i in range(0,len(props)):
            yl, xl, yr, xr = [int(b) for b in props[i].bbox]
            annotation = np.array([[xl,yl,xr,yr,cl]])
            annotations = np.append(annotations, annotation, axis=0)
        
    return annotations

data_path = '../Hemorrhage_dataset/hemorrhage_binary_n200/'
image_paths = glob(data_path + 'image/*.png')
image_list = os.listdir(data_path + 'image')
mask_paths = [image_paths[i][:45] + 'mask/' + image_list[i][:-8] + '_label.png' for i in range(len(image_paths))]

bbox_list = []
for mask_path in mask_paths:
    
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    bbox = get_bbox(mask, len(params.obj_list))
    print(bbox)
    bbox_list.append(bbox)

csvfile=open('./bbox_hm.csv','w', newline="")
csvwriter = csv.writer(csvfile)

for row in bbox_list:
    csvwriter.writerow(row)
csvfile.close()