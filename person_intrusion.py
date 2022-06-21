import cv2
import torch

model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
model.classes = 0

def intrusion_detection(im):
    """
    Check for person intrusion in an entire frame <geolocation to be added>
    
    :param frame: Image frame
    :type frame: numpy.ndarray
    :return: Dictionary with the following keys:
        - base_image: Original image
        - processed_image: Image with detected objects
        - violation: Boolean indicating if there is a person intrusion
        - amount: Number of detected objects
        - candidates: List of all objects
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
            violist.append([xmin,ymin,xmax,ymax])
            isg = cv2.rectangle(isg, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        dict = {'base_image': im, 
        'processed_image': isg,
        'violation': True,
        'amount': amt,
        'candidates': listolist,
        'selected': violist,
        }
        return dict