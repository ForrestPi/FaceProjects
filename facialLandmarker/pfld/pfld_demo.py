import argparse
import time

import numpy as np
import torch
import torchvision
from torchvision import transforms
import cv2

from models.pfld import PFLDInference, AuxiliaryNet
from mtcnn.detector import detect_faces, show_bboxes



class PFLD(object):
    def __init__(self,model_path):
        checkpoint = torch.load(model_path)
        plfd_backbone = PFLDInference().cuda()
        plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
        plfd_backbone.eval()
        self.plfd_backbone = plfd_backbone.cuda()
        
        self.transform = transforms.Compose([transforms.ToTensor()])
    
    def lanmark(self,image):
        input = cv2.resize(image, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        
        input = self.transform(input).unsqueeze(0).cuda()
        _, landmarks = self.plfd_backbone(input)
        pre_landmark = landmarks[0]
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2)
        return pre_landmark
        
    def process(self,img,bounding_boxes):
        height, width = img.shape[:2]
        for box in bounding_boxes:
            #score = box[4]
            x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            size = int(max([w, h])*1.1)
            cx = x1 + w//2
            cy = y1 + h//2
            x1 = cx - size//2
            x2 = x1 + size
            y1 = cy - size//2
            y2 = y1 + size

            dx = max(0, -x1)
            dy = max(0, -y1)
            x1 = max(0, x1)
            y1 = max(0, y1)

            edx = max(0, x2 - width)
            edy = max(0, y2 - height)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            cropped = img[y1:y2, x1:x2]
            if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
                cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
            
            cropped = cv2.resize(cropped, (112, 112))

            input = cv2.resize(cropped, (112, 112))
            input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
            
            input = self.transform(input).unsqueeze(0).cuda()
            _, landmarks = self.plfd_backbone(input)
            pre_landmark = landmarks[0]
            pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size]
        return pre_landmark
    
    def transfer(self,landmark):
        eye_left=np.mean(landmark[60:68], axis=0)
        eye_right=np.mean(landmark[68:76], axis=0)
        nose_tip=np.mean(landmark[55:60], axis=0)
        points=[]
        points.append([eye_left[0],eye_left[1]])
        points.append([eye_right[0],eye_right[1]])
        points.append([nose_tip[0],nose_tip[1]])
        return points
    

def test_pfld(args):
    pfld = PFLD(args.model_path)
    image = cv2.imread(args.image_path)
    lanmark = pfld.lanmark(image)
    print(lanmark*112)    
    

def transfer(landmark):
    eye_left=np.mean(landmark[60:68], axis=0)
    eye_right=np.mean(landmark[68:76], axis=0)
    nose_tip=np.mean(landmark[55:60], axis=0)
    points=[]
    points.append([eye_left[0],eye_left[1]])
    points.append([eye_right[0],eye_right[1]])
    points.append([nose_tip[0],nose_tip[1]])
    return points
    
def main(args):
    checkpoint = torch.load(args.model_path)
    plfd_backbone = PFLDInference().cuda()
    plfd_backbone.load_state_dict(checkpoint['plfd_backbone'])
    plfd_backbone.eval()
    plfd_backbone = plfd_backbone.cuda()
    transform = transforms.Compose([transforms.ToTensor()])

    img = cv2.imread(args.image_path)

    height, width = img.shape[:2]

    bounding_boxes, landmarks = detect_faces(img)
    for box in bounding_boxes:
        #score = box[4]
        x1, y1, x2, y2 = (box[:4]+0.5).astype(np.int32)
        w = x2 - x1 + 1
        h = y2 - y1 + 1

        size = int(max([w, h])*1.1)
        cx = x1 + w//2
        cy = y1 + h//2
        x1 = cx - size//2
        x2 = x1 + size
        y1 = cy - size//2
        y2 = y1 + size

        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)

        edx = max(0, x2 - width)
        edy = max(0, y2 - height)
        x2 = min(width, x2)
        y2 = min(height, y2)
        
        
        cropped = img[y1:y2, x1:x2]
        if (dx > 0 or dy > 0 or edx > 0 or edy > 0):
            cropped = cv2.copyMakeBorder(cropped, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)
        cv2.imwrite("cropped.jpg",cropped)
        print("cropped shape:",cropped.shape)#cropped shape: (668, 668, 3)
        cropped = cv2.resize(cropped, (112, 112))

        input = cv2.resize(cropped, (112, 112))
        input = cv2.cvtColor(input, cv2.COLOR_BGR2RGB)
        
        input = transform(input).unsqueeze(0).cuda()
        _, landmarks = plfd_backbone(input)
        pre_landmark = landmarks[0]
        print("size:",size)#size: 668
        pre_landmark = pre_landmark.cpu().detach().numpy().reshape(-1, 2) * [size, size]
        for idx,(x, y) in enumerate(pre_landmark.astype(np.int32)):
            cv2.circle(img, (x1 + x, y1 + y), 1, (0, 0, 255))
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, str(idx+1), (x1 + x, y1 + y), font, 0.4, (0, 0, 255), 1,cv2.LINE_AA)
        
        points = transfer(pre_landmark)
        for point in points:
            cv2.circle(img, (int(round(x1 + point[0])), int(round(y1 + point[1]))), 5, (255, 0, 0),2)
        
    cv2.imwrite("result.jpg",img)
    #cv2.imshow('0', img)
    #cv2.waitKey(0)
            



def parse_args():
    parser = argparse.ArgumentParser(description='Testing')
    parser.add_argument(
        '--model_path',
        default="./checkpoint/snapshot/checkpoint.pth.tar",
        type=str)
    parser.add_argument(
        '--image_path',
        default="./single_face.jpg",
        type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    main(args)
    