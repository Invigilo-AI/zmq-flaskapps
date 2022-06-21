import cv2
import torch
import pandas as pd
import itertools

machine_model = torch.hub.load('ultralytics/yolov5', 'custom', './dangerousmachine.pt')
person_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

person_model.classes = 0,1,2,3,5,7,8
person_model.conf = 0.5

def distancing(peoplelist,machinelist, img, dist_thres_lim=(100,150)):
    violationcounter = 0
    already_red = dict() # dictionary to store if a plotted rectangle has already been labelled as high risk
    centers = []
    for i in peoplelist:
        centers.append(((int(i[2])+int(i[0]))//2,(int(i[3])+int(i[1]))//2))
    for j in centers:
        already_red[j] = 0
    x_combs = list(itertools.product(peoplelist,machinelist))
    radius = 10
    thickness = 5
    for x in x_combs:
        xyxy1, xyxy2 = x[0],x[1]
        cntr1 = ((int(xyxy1[2])+int(xyxy1[0]))//2,(int(xyxy1[3])+int(xyxy1[1]))//2)
        cntr2 = ((int(xyxy2[2])+int(xyxy2[0]))//2,(int(xyxy2[3])+int(xyxy2[1]))//2)
        dist = ((cntr2[0]-cntr1[0])**2 + (cntr2[1]-cntr1[1])**2)**0.5

        if dist < dist_thres_lim[0]:
            violationcounter += 1
            color = (0, 0, 255)
            already_red[cntr1] = 1
            already_red[cntr2] = 1
            cv2.line(img, cntr1, cntr2, color, thickness)
            cv2.circle(img, cntr1, radius, color, -1)
            cv2.circle(img, cntr2, radius, color, -1)
    return img, violationcounter


def proximity_detection(im):
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
    machinelist = []
    violist = []
    people_frame = person_model(im, size=1280).pandas().xyxy[0]
    machine_frame = machine_model(im).pandas().xyxy[0]
    full = pd.concat([people_frame, machine_frame], ignore_index=True)
    for i in range(0,len(full)):
        if full['name'][i] == 'person':
            peoplelist.append([full['xmin'][i],full['ymin'][i],full['xmax'][i],full['ymax'][i]])
        else:
            machinelist.append([full['xmin'][i],full['ymin'][i],full['xmax'][i],full['ymax'][i]])
    if len(peoplelist) == 0:
        dict = {'base_image': im,
        'processed_image': im,
        'violation': False,
        'amount': 0,
        'candidates': 'NULL',
        'selected': 'NULL',
        }
        return dict
    elif len(machinelist) == 0:
        dict = {'base_image': im,
        'processed_image': im,
        'violation': False,
        'amount': 0,
        'candidates': peoplelist,
        'selected': 'NULL',
        }
        return dict
    else:
        imc = im.copy()
        imc,violationcounter = distancing(peoplelist,machinelist,im)
        if violationcounter == 0:
            violationboolean = False
        else:
            violationboolean = True
        dict = {'base_image': im,
        'processed_image': imc,
        'violation': violationboolean,
        'amount': violationcounter,
        'candidates': peoplelist,
        'selected': 'NULL',
        }
        return dict