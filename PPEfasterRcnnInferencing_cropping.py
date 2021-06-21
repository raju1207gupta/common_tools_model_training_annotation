import cv2 as cv2
import time
import os
import numpy as np

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

def image_color_detection(img):
    img = cv2.resize(img,(200,200))
    detection_color = ''
    imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    print(f'pixel value : {imggray[100,20]}')
    if imggray[100,20] in range(160,190):
        detection_color = 'green'
    if imggray[100,20] in range(45,102):
        detection_color = 'yellow'
    if imggray[100,20] in range(105,130):
        detection_color = 'white'
    if imggray[100,20] in range(200,250):
        detection_color = 'red'
    return detection_color



path  = 'path_to/classification/mask_crops/'

cv2Net = cv2.dnn.readNetFromTensorflow("path_to/frozen_inference_graph_old.pb", "path_to/PPE_detection.pbtxt")
image_path = "path_to/saved_images_new/"

cap = cv2.VideoCapture(0)

timeStamp=time.time()
fpsFilt = 0
frmno = 1200
#while cap.isOpened():
for item in os.listdir(image_path):
    if os.path.isfile(image_path+item):
        img = cv2.imread(image_path+item)
        #ret, img = cap.read()
        
        #img = cv2.imread('000023.jpg')
        
        img = cv2.resize(img,(1280,720))
        #seg_frame = img_segmentation(img)
        rows = img.shape[1]
        cols = img.shape[0]
        cv2Net.setInput(cv2.dnn.blobFromImage(img, size=(1080,720), swapRB=True, crop=False))
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
            if score > 0.60 and (class_id == 0 or class_id == 1):
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
                

                # update our list of bounding box coordinates,
                # centroids, and confidences
                boxes.append([left, top, int(right), int(bottom)])
                confidences.append(float(confidence))
        idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.35, 0.35)
        # ensure at least one detection exists
        if len(idxs) > 0:
            # loop over the indexes we are keeping
            for i in idxs.flatten():
                frmno = frmno+1
                # extract the bounding box coordinates
                (x, y) = (boxes[i][0], boxes[i][1])
                (w, h) = (boxes[i][2], boxes[i][3])
                #cv2.rectangle(img, (x, y), (x + w, y + h), (0,0,255), 2)
                
                # (startX, startY) = (max(0, int(x-(w-x)/2)), max(0, int(y-(h-y)/2)))
                # (endX, endY) = (min(rows - 1, int(w+(w-x)/2)), min(cols - 1, int(h+(h-y)/2)))
                #save_Frame=img[startY:endY, startX:endX]
                save_Frame=img[y:h, x:w]
                #save_Frame=seg_frame[y:h, x:w]
                # seg_frame = img_segmentation(save_Frame)
                #color_detected = image_color_detection(save_Frame)
                #cv2.rectangle(img, (x, y), (w, h), (0,0,255), 2)
                #cv2.putText(img, color_detected, (x, y - 10),cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0,255,0), 2)
                try:
                    cv2.imwrite(os.path.join(path,f'{str(frmno)}.jpg'),save_Frame)
                except:
                    continue

        dt=time.time()-timeStamp
        timeStamp=time.time()
        fps=1/dt
        fpsFilt=.9*fpsFilt + .1*fps
        #print(str(round(fps,1))+' fps')
        cv2.putText(img,str(round(fpsFilt,1))+' fps',(0,30),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2)
        cv2.imshow('img', img)
        cv2.resizeWindow('Frame',800,600)
        cv2.waitKey(0)

        key = cv2.waitKey(1) & 0xFF

        # Press `q` to exit
        if key == ord("q"):
            break

# Clean
#cap.release()
cv2.destroyAllWindows()