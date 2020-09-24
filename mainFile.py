#for loading video and displaying the output
# copied run_video.py initially
import argparse
import logging
import time
import math
import csv

import cv2
import numpy as np

from tf_pose.estimator import TfPoseEstimator
from tf_pose.networks import get_graph_path, model_wh

logger = logging.getLogger('TfPoseEstimator-Video')
logger.setLevel(logging.DEBUG)
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)
formatter = logging.Formatter('[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)

fps_time = 0

def choosePose(choosedNumber): 
    switcher = { 
        1: "warriorII", 
        2: "cat", 
        3: "chair",
        4: "triangle",  
    } 
    return switcher.get(choosedNumber, "nothing") 

def angle(flat,pt1,pt2,pt3):
    angle_calc=0
    a = (flat[pt1*2],flat[(pt1*2)+1])
    b = (flat[pt2*2],flat[(pt2*2)+1]) # b is midpoint
    c = (flat[pt3*2],flat[(pt3*2)+1])
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    angle_calc = ang + 360 if ang < 0 else ang
    angle_calc = 360-angle_calc if angle_calc > 215 else angle_calc
    return angle_calc


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='tf-pose-estimation Video')
    parser.add_argument('--video', type=str, default='')
    parser.add_argument('--resolution', type=str, default='432x368', help='network input resolution. default=432x368')
    parser.add_argument('--model', type=str, default='mobilenet_thin', help='cmu / mobilenet_thin / mobilenet_v2_large / mobilenet_v2_small')
    parser.add_argument('--show-process', type=bool, default=False,
                        help='for debug purpose, if enabled, speed for inference is dropped.')
    parser.add_argument('--showBG', type=bool, default=True, help='False to show skeleton only.')
    args = parser.parse_args()

    logger.debug('initialization %s : %s' % (args.model, get_graph_path(args.model)))
    w, h = model_wh(args.resolution)
    e = TfPoseEstimator(get_graph_path(args.model), target_size=(w, h))
    cap = cv2.VideoCapture(args.video)

    # MAIN MENU
    print("_____________________________________________________________________________________________")
    accurate_angle_list = []
    choosedNumber=int(input("\tEnter Pose to perform:-\n\t\t1.WarriorII Pose\n\t\t2.Cat Pose\n\t\t3.Chair Pose\n\t\t4.Triangle Pose\n\t"))
    selectedPose = choosePose(choosedNumber)
    selectedPose=selectedPose.replace(" ", "")
    print("**************** Selected Pose:-",selectedPose,"Pose ****************")

    with open('./sepData/fetchSet.csv', 'r') as inputCSV:
        for row in csv.reader(inputCSV):
            if row[8] == selectedPose: # and row[1] == "0" and row[2] == "0" and row[3] == "0" and row[4] == "0" and row[5] == "0" and row[6] == "0" and row[7] == "0" and row[8] == "0" and row[9] == "0":
                accurate_angle_list=[int(row[0]),int(row[1]),int(row[2]),int(row[3]),int(row[4]),int(row[5]),int(row[6]),int(row[7])]
                break
        # writer.writerow(row)
    print("****************List:-",accurate_angle_list,"****************")
    print("_____________________________________________________________________________________________")
    # rshoulder=117, lshoulder=100, relbow=183,	lelbow=185, rhip=209, lhip=55, rknee=168, lknee=180 --> triangle pose
    # Hip and NECK deleted for POSE issues
    angle_name_list=["Rshoulder","Lshoulder","Relbow","Lelbow","Rhip","Lhip","Rknee","Lknee"]
    # accurate_angle_list = [117,100,183,185,209,55,168,178]
    angle_coordinates=[[8,2,3],[6,5,11],[2,3,4],[7,6,5],[9,8,2],[12,11,5],[8,9,10],[13,12,11]]
    pos_onScreen=[100,1000]
    correctionValue = 13

    if cap.isOpened() is False:
        print("Error opening video stream or file")
    global flat
    flat = [0.0 for i in range(36)]
    while cap.isOpened():
        ret_val, image = cap.read()

        # humans = e.inference(image)
        humans = e.inference(image, resize_to_default=(w > 0 and h > 0), upsample_size=4.0)
        if not args.showBG:
            image = np.zeros(image.shape)
        # print(humans)
        image,flat = TfPoseEstimator.draw_humans(image, humans, imgcopy=False)
        # print("________________________________")

		# PURE CODE
        correctAngleCount = 0
        for itr in range(8):
            angleObtained = angle(flat , angle_coordinates[itr][0],angle_coordinates[itr][1],angle_coordinates[itr][2])

            if(angleObtained<accurate_angle_list[itr]-correctionValue):
                status="more"
            elif(accurate_angle_list[itr]+correctionValue<angleObtained):
                status="less"
            else:
                status="OK"
                correctAngleCount+=1
            cv2.putText(image, angle_name_list[itr]+":- %s" % (status), (pos_onScreen[itr%2], (itr+1)*70),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        posture="CORRECT" if correctAngleCount>6 else "WRONG"
        cv2.putText(image, "Posture:- %s" % (posture), (590, 80),  cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)

        cv2.putText(image, "FPS: %f" % (1.0 / (time.time() - fps_time)), (590, 40),  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.imshow('tf-pose-estimation result', image)
        fps_time = time.time()
        
        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()
logger.debug('finished+')
