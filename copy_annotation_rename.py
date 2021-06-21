import cv2 as cv2
import os
import numpy as np
import glob

from pathlib import Path
import xml.etree.cElementTree as ET
from PIL import Image


def xml_file_rename(path,path_to_final_annotation):
    frmnumber = 828
    dirs = os.listdir(path)
    for frmno,item in enumerate(dirs):
        if os.path.isfile(f"{path}{(frmno+1)*60}.xml"):
            tree = ET.parse(f"{path}{(frmno+1)*60}.xml")
            root = tree.getroot()
            frmnumber = frmnumber+1
            root.find('filename').text = f"{frmnumber}.jpg"
            tree.write(path_to_final_annotation+f"{frmnumber}.xml")

path_to_final_annotation  = 'path_to/final_dataset_annotation/'
PATH_TO_ANNOTATION = "path_to/annotations/"

xml_file_rename(PATH_TO_ANNOTATION,path_to_final_annotation)
        
