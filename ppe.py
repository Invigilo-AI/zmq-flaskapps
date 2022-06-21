import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'custom', './ppev2.pt')

def ppe_detection(im):
    """
    Checks for PPE detection in the frame
    
    :param frame: Image frame
    :type frame: numpy.ndarray
    :return: Dictionary with the following keys:
        - base_image: Original image
        - processed_image: Image with detected objects
        - violation: Boolean indicating if there is a ppe violation
        - amount: Number of violations
        - candidates: List of all detection
        - selected: List of selected objects that perform the violation
    :rtype: dict
    """
    listolist = []
    violist = []
    results = model(im)
    data = results.pandas().xyxy[0]
    amt = len(data)
    if amt==0:
        dict = {'base_image': im, 
    'processed_image': im,
    'violation': False,
    'amount': amt,
    'candidates': 'NULL',
    'selected': 'NULL',
    }
        return dict
    else:
        isg = im.copy()
        for i in range(0,amt):
            xmin = int(data['xmin'][i])
            ymin = int(data['ymin'][i])
            xmax = int(data['xmax'][i])
            ymax = int(data['ymax'][i])
            listolist.append([xmin,ymin,xmax,ymax])
            if data['class'][i] != 1:
                color = (0,255,0)
            else:
                color = (0,0,255)
                violist.append([xmin,ymin,xmax,ymax])
            isg = cv2.rectangle(isg, (xmin, ymin), (xmax, ymax), color, 2)
        amt = len(violist)
        if amt==0:
            violationbool = False
        else:
            violationbool = True
        dict = {'base_image': im, 
        'processed_image': isg,
        'violation': violationbool,
        'amount': amt,
        'candidates': listolist,
        'selected': violist,
        }
        return dict