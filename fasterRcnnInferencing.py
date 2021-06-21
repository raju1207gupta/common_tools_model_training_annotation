import cv2 as cv2
import time
import os
import numpy as np

from pathlib import Path
import xml.etree.cElementTree as ET
from PIL import Image

def create_labimg_xml(image_path, annotation_list):
    
    image_path = Path(image_path)
    img = np.array(Image.open(image_path).convert('RGB'))

    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = str(image_path.parent.name)
    ET.SubElement(annotation, 'filename').text = str(image_path.name)
    ET.SubElement(annotation, 'path').text = str(image_path)

    source = ET.SubElement(annotation, 'source')
    ET.SubElement(source, 'database').text = 'Unknown'

    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str (img.shape[1])
    ET.SubElement(size, 'height').text = str(img.shape[0])
    ET.SubElement(size, 'depth').text = str(img.shape[2])

    ET.SubElement(annotation, 'segmented').text = '0'

    for annot in annotation_list:
        tmp_annot = annot.split(',')
        cords, label = tmp_annot[0:-1], tmp_annot[-1]
        xmin, ymin, xmax, ymax = cords[0], cords[1], cords[2], cords[3]

        object = ET.SubElement(annotation, 'object')
        ET.SubElement(object, 'name').text = label
        ET.SubElement(object, 'pose').text = 'Unspecified'
        ET.SubElement(object, 'truncated').text = '0'
        ET.SubElement(object, 'difficult').text = '0'

        bndbox = ET.SubElement(object, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    tree = ET.ElementTree(annotation)
    xml_file_name = f"{image_path.parent}\\annotations\\{(image_path.name.split('.')[0]+'.xml')}"
    tree.write(xml_file_name)

def img_segmentation(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # reshape the image to a 2D array of pixels and 3 color values (RGB)
    pixel_values = image.reshape((-1, 3))
    # convert to float
    pixel_values = np.float32(pixel_values)
    # define stopping criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    # number of clusters (K)
    k = 5
    _, labels, (centers) = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # convert back to 8 bit values
    centers = np.uint8(centers)
    # flatten the labels array
    labels = labels.flatten()
    # convert all pixels to the color of the centroids
    segmented_image = centers[labels.flatten()]
    # reshape back to the original image dimension
    segmented_image = segmented_image.reshape(image.shape)
    # disable only the cluster number 2 (turn the pixel into black)
    masked_image = np.copy(image)
    # convert to the shape of a vector of pixel values
    masked_image = masked_image.reshape((-1, 3))
    # color (i.e cluster) to disable
    cluster = 5
    masked_image[labels == cluster] = [0, 0, 0]
    masked_image[labels == 2] = [0, 0, 0]
    masked_image[labels == 3] = [255, 0, 0]
    # convert back to original shape
    masked_image = masked_image.reshape(image.shape)
    return masked_image

path  = "D:/deeplearning learn/Social distancing/face-mask-detector/dataset_GMR/classification/without_mask/"
cv2Net = cv2.dnn.readNetFromTensorflow("D:/deeplearning learn/OrionEdgeSocialDistancingAPI/ModelGraph/PPE_Detection_FrozenGraph1/frozen_inference_graph_old.pb", "D:/deeplearning learn/OrionEdgeSocialDistancingAPI/ModelGraph/PPE_Detection_FrozenGraph1/PPE_detection.pbtxt")
# cv2Net = cv2.dnn.readNetFromTensorflow('D://deeplearning learn//Social distancing//blockdetection//covid-19//frozen_inference_graph.pb', 'D://deeplearning learn//Social distancing//blockdetection//covid-19//fasterRcnnmappbtxtfile.pbtxt')
#cv2Net = cv2.dnn.readNetFromTensorflow('D:/deeplearning learn/OrionEdgeSocialDistancingAPI/ModelGraph/Mask_Detection_FrozenGraph/frozen_inference_graph3.pb', 'covid-19/mask_detection_pipeline_config/mask_detection_faster_rcnn_inception_v2_coco_9_6_20.pbtxt')
#cv2Net = cv2.dnn.readNetFromTensorflow('D:/deeplearning learn/OrionEdgeSocialDistancingAPI/ModelGraph/Mask_Detection_FrozenGraph/frozen_inference_graph2.pb', 'covid-19/mask_detection_pipeline_config/mask_detection_faster_rcnn_inception_v2_coco_17_6_20.pbtxt')
#cv2Net = cv2.dnn.readNetFromTensorflow('D:/deeplearning learn/OrionEdgeSocialDistancingAPI/ModelGraph/Mask_Detection_FrozenGraph/frozen_inference_graph1.pb', 'covid-19/mask_detection_pipeline_config/mask_detection_frozen_inference_graph20_5_20.pbtxt')
#cv2Net = cv2.dnn.readNetFromModelOptimizer('C:/Users/Raju/Documents/Intel/OpenVINO/openvino_models/ir/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.xml', 'C:/Users/Raju/Documents/Intel/OpenVINO/openvino_models/ir/intel/vehicle-license-plate-detection-barrier-0106/FP32/vehicle-license-plate-detection-barrier-0106.bin')
# cap = cv2.VideoCapture("C:/Users/Raju/Videos/Captures/lucastv.mp4")
#cap = cv2.VideoCapture("D:/deeplearning learn/Social distancing/blockdetection/output.mp4")
#cap = cv2.VideoCapture("D:/deeplearning learn/pyreaserch/social-distance-detector/marico.mp4")
#cap = cv2.VideoCapture("D:/deeplearning learn/marico_demo/HUL3.mp4")
#cap = cv2.VideoCapture("rtsp://admin:HUL@2020@103.89.58.106:554/cam/realmonitor?channel=6&subtype=0")
#cap = cv2.VideoCapture("rtsp://admin:G#i@l@2019@10.66.0.116:554/cam/realmonitor?channel=10&subtype=0")
cap = cv2.VideoCapture("rtsp://admin:G#i@l@2019@10.66.0.191:554/cam/realmonitor?channel=10&subtype=0")
# cap = cv2.VideoCapture(0)
#cap = cv2.VideoCapture("D:/deeplearning learn/pose_extractor_build/ActionAI/image_dir/all/sample.mp4")
timeStamp=time.time()
fpsFilt = 0
frmno = 1088
frmnumber = 0
while cap.isOpened() :
    anotation_list = []
    ret, img = cap.read()
    frmnumber = frmnumber+1
    #img = cv2.imread('000023.jpg')
    
    img = cv2.resize(img,(1080,720),interpolation=cv2.INTER_CUBIC)
    #seg_frame = img_segmentation(img)
    rows = img.shape[1]
    cols = img.shape[0]
    if frmnumber%60==0:
        cv2Net.setInput(cv2.dnn.blobFromImage(img, size=(1080*2,720*2), swapRB=True, crop=False))
        #cv2Net.setInput(cv2.dnn.blobFromImage(img,scalefactor=2, size=(1080,720), swapRB=True, crop=False))
        cv2Out = cv2Net.forward()
        centroids = []
        boxes = []
        confidences = []
        classIDs = []
        for detection in cv2Out[0,0,:,:]:
            score = float(detection[2])
            class_id = int(detection[1])
            #class_id = int(np.argmax(score))
            confidence = score
            if score > 0.55 and (class_id == 0 or class_id == 1):
            #if score > 0.85 :
                left = int(detection[3] * rows)
                top = int(detection[4] * cols)
                right = int(detection[5] * rows)
                bottom = int(detection[6] * cols)
                # top = int(detection[3] * cols)
                # left = int(detection[4] * rows)
                # bottom = int(detection[5] * cols)
                # right = int(detection[6] * rows)
    
                #cv2.rotate(img,rotateCode=90)   
                #cv2.imwrite(os.path.join(path,f'{str(frmno)}.jpg'),img[left:right, top:bottom])
                #cv2.imwrite(os.path.join(path,f'{str(frmno)}.jpg'),img)
                #cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                # if class_id == 0 :
                #     cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                #     cv2.putText(img, 'No mask', (int(left), int(top - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                # #if class_id == 1:
                # else:
                #     cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                #     cv2.putText(img, 'with mask', (int(left), int(top - 10)),cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if class_id == 0 :
                    #cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 255, 0), thickness=2)
                    #cv2.putText(img, 'with_helmet', (int(left), int(top - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                    class_name="with_helmet"
                    coords = str(f"{left},{top},{right},{bottom},with_helmet")
                elif class_id == 1 :
                    class_name="without_helmet"
                    #cv2.rectangle(img, (int(left), int(top)), (int(right), int(bottom)), (0, 0, 255), thickness=2)
                    #cv2.putText(img, 'without_helmet', (int(left), int(top - 10)),cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    coords = str(f"{left},{top},{right},{bottom},without_helmet")
                anotation_list.append(coords)
                
                
                # update our list of bounding box coordinates,
                # centroids, and confidences
                classIDs.append(class_name)
                boxes.append([left, top, int(right), int(bottom)])
                confidences.append(float(confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.30, 0.3)
        # ensure at least one detection exists
        if len(idxs) > 0 :
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                frmno = frmno+1
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                # cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                #cv2.rectangle(img, (x, y), (w, h), (0,0,255), 2)
                # (startX, startY) = (max(0, int(x-(w-x)/2)), max(0, int(y-(h-y)/2)))
                # (endX, endY) = (min(rows - 1, int(w+(w-x)/2)), min(cols - 1, int(h+(h-y)/2)))
                # save_Frame=img[startY:endY, startX:endX]
                #save_Frame=seg_frame[y:h, x:w]
                save_Frame=img[y:h, x:w]
                # save_Frame=img[y:int(y+(h-y)/4), int(x+(w-x)/5):int(w-(w-x)/5)]
                # save_Frame = cv2.resize(save_Frame,(300,600),fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
                
                # seg_frame = img_segmentation(save_Frame)
                # coords = str(f"{int(x+(w-x)/5)},{y},{int(w-(w-x)/5)},{int(y+(h-y)/5)},with_helmet")
                # anotation_list.append(coords)
                
               
                try:
                    save_Frame= cv2.resize(save_Frame,(299,299))
                    cv2.imwrite(os.path.join(path,f'{str(frmno)}.jpg'),save_Frame)
                    
                except:
                    continue
                # try:
                #     pathh = path+classIDs[i]+'/'
                #     save_Frame=img[y:h, x:w]
                #     save_Frame = cv2.resize(save_Frame,(299,299),fx=4,fy=4,interpolation=cv2.INTER_CUBIC)
                #     cv2.imwrite(os.path.join(pathh,f'{str(frmno)}.jpg'),save_Frame)
                    
                # except:
                #     continue

        dt=time.time()-timeStamp
        timeStamp=time.time()
        fps=1/dt
        fpsFilt=.9*fpsFilt + .1*fps
        #print(str(round(fps,1))+' fps')
        #cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow('img', img)
        #cv2.imwrite(os.path.join(path,f'{str(frmnumber)}.jpg'),img)
        #create_labimg_xml(os.path.join(path,f'{str(frmnumber)}.jpg'), anotation_list)
        cv2.resizeWindow('Frame',800,600)

    key = cv2.waitKey(1) & 0xFF

    # Press `q` to exit
    if key == ord("q"):
        break

# Clean
cap.release()
cv2.destroyAllWindows()