import cv2
import torch
import pandas as pd
import itertools

person_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

person_model.classes = 0
person_model.conf = 0.5

def distancing(peoplelist, img, dist_thres_lim=(100,150)):
    violationcounter = 0
    selectedlist = []
    already_red = dict() # dictionary to store if a plotted rectangle has already been labelled as high risk
    centers = []
    for i in peoplelist:
        centers.append(((int(i[2])+int(i[0]))//2,(int(i[3])+int(i[1]))//2))
    for j in centers:
        already_red[j] = 0
    x_combs = list(itertools.product(peoplelist,2))
    radius = 10
    thickness = 2
    for x in x_combs:
        xyxy1, xyxy2 = x[0],x[1]
        cntr1 = ((int(xyxy1[2])+int(xyxy1[0]))//2,(int(xyxy1[3])+int(xyxy1[1]))//2)
        cntr2 = ((int(xyxy2[2])+int(xyxy2[0]))//2,(int(xyxy2[3])+int(xyxy2[1]))//2)
        dist = ((cntr2[0]-cntr1[0])**2 + (cntr2[1]-cntr1[1])**2)**0.5
        if dist < dist_thres_lim[0]:
            violationcounter += 1
            color = (0, 0, 255)
            lcolor = (0, 255, 0)
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, lcolor, thickness)
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
            cv2.rectangle(img, (int(xyxy1[0]),int(xyxy1[1])), (int(xyxy1[2]),int(xyxy1[3])), color, thickness)
            cv2.rectangle(img, (int(xyxy2[0]),int(xyxy2[1])), (int(xyxy2[2]),int(xyxy2[3])), color, thickness)
            if xyxy1 not in selectedlist:
                selectedlist.append(xyxy1)
            if xyxy2 not in selectedlist:
                selectedlist.append(xyxy2)
    return img, violationcounter, selectedlist


def socialdistance_detection(im):
    """
    Check if an image frame from a camera stream contains a person in proximity to a machine.
    
    :param frame: Image frame
    :type frame: numpy.ndarray
    :return: Dictionary with the following keys:
        - base_image: Original image
        - processed_image: Image with detected objects
        - violation: Boolean indicating if there is a person in proximity to a machine
        - amount: Number of detected violations
        - candidates: List of all people
        - selected: NULL for now
    :rtype: dict
    """
    peoplelist = []
    full = person_model(im, size=1280).pandas().xyxy[0]
    for i in range(0,len(full)):
        if full['name'][i] == 'person':
            peoplelist.append([full['xmin'][i],full['ymin'][i],full['xmax'][i],full['ymax'][i]])
    if len(peoplelist) == 0:
        dict = {'base_image': im,
        'processed_image': im,
        'violation': False,
        'amount': 0,
        'candidates': 'NULL',
        'selected': 'NULL',
        }
        return dict
    else:
        imc = im.copy()
        imc,violationcounter, selected = distancing(peoplelist,im)
        if violationcounter == 0:
            violationboolean = False
        else:
            violationboolean = True
        dict = {'base_image': im,
        'processed_image': imc,
        'violation': violationboolean,
        'amount': violationcounter,
        'candidates': peoplelist,
        'selected': selected,
        }
        return dict
