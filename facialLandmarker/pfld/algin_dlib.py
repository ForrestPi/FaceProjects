import numpy as np
import cv2
import dlib
import math
import imutils



def warp_affine(image, points, scale=1.0):
    dis1,dis2 = getDis(points[2][0],points[2][1],points[0][0],points[0][1],points[1][0],points[1][1])
    eye_center = ((points[0][0] + points[1][0]) / 2, (points[0][1] + points[1][1]) / 2)
    dy = points[1][1] - points[0][1]
    dx = points[1][0] - points[0][0]
    center=(points[2][0],points[2][1])
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx) #获取旋转角度angle = cv2.fastAtan2((y2 - y1), (x2 - x1))
    print("angle:",angle)
    rot = cv2.getRotationMatrix2D(center, angle, scale=scale) # 获取旋转矩阵
    rot_img = cv2.warpAffine(image, rot, dsize=(image.shape[1], image.shape[0]))
    delta_width = dis2*1
    delta_height1 = dis1*3
    delta_height2 = dis1*2
    x1 = max(round(center[0]-delta_width),0)
    y1 = max(round(center[1]-delta_height1),0)
    x2 = min(x1+round(delta_width*2),rot_img.shape[1])
    y2 = min(round(y1+delta_height1+delta_height2),rot_img.shape[0])
    return rot_img,(x1,y1,x2,y2)

def detect(image):
    detector = dlib.get_frontal_face_detector()
    # 取灰度
    img_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 人脸数rects
    rects = detector(img_gray, 0)
    return rects


def landmark(image,rects):
    predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    landmarksList=[]
    for i in range(len(rects)):
        landmarks = np.matrix([[p.x, p.y] for p in predictor(image,rects[i]).parts()])
        landmarksList.append(landmarks)
    return landmarksList

def vis_landmark(landmarksList,image):
    img = image.copy()
    for idx, point in enumerate(landmarks):
        # 68点的坐标
        pos = (point[0, 0], point[0, 1])
        print(idx,pos)

        # 利用cv2.circle给每个特征点画一个圈，共68个
        cv2.circle(img, pos, 5, color=(0, 255, 0))
        # 利用cv2.putText输出1-68
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(img, str(idx+1), pos, font, 0.8, (0, 0, 255), 1,cv2.LINE_AA)
    return img


def dlib_points(landmark):
    eye_left=np.mean(np.asarray(landmark[36:42]), axis=0)
    eye_right=np.mean(np.asarray(landmark[42:48]), axis=0)
    nose_tip = np.asarray(landmark[30])
    nose_tip = np.squeeze(nose_tip)
    points=[]
    points.append([eye_left[0],eye_left[1]])
    points.append([eye_right[0],eye_right[1]])
    points.append([nose_tip[0],nose_tip[1]])
    return points


def getDis(pointX,pointY,lineX1,lineY1,lineX2,lineY2):
    a=lineY2-lineY1
    b=lineX1-lineX2
    c=lineX2*lineY1-lineX1*lineY2
    dis1=(math.fabs(a*pointX+b*pointY+c))/(math.pow(a*a+b*b,0.5))
    dis2=math.sqrt(a*a+b*b)
    return dis1,dis2


def crop_face(image, points):
    dis1,dis2 = getDis(points[2][0],points[2][1],points[0][0],points[0][1],points[1][0],points[1][1])
    dy = points[1][1] - points[0][1]
    dx = points[1][0] - points[0][0]
    center=(points[2][0],points[2][1])
    print("center:",center)
    cv2.circle(image,center,radius =3,
           color = (0,0,255), thickness = 2)
    print(dis1,dis2)
    # 计算旋转角度
    angle = cv2.fastAtan2(dy, dx) #获取旋转角度angle = cv2.fastAtan2((y2 - y1), (x2 - x1))

    delta_width = dis2*1
    delta_height1 = dis1*3
    delta_height2 = dis1*2
    x1 = max(round(center[0]-delta_width),0)
    y1 = max(round(center[1]-delta_height1),0)
    x2 = min(x1+round(delta_width*2),image.shape[1])
    y2 = min(round(y1+delta_height1+delta_height2),image.shape[0])

    polygon = np.array([(x1,y1),
                        (x2,y1),
                        (x2,y2),
                        (x1,y2),],np.int32)
    print("polygon:",polygon)
    #cv2.circle(image,(int(center[0]-delta_width),int(center[1]-delta_height)),radius =3,
    #       color = (0,0,255), thickness = 2)
    #cv2.circle(image,(int(center[0]+delta_width),int(center[1]+delta_height)),radius =3,
    #       color = (0,0,255), thickness = 2)
    # magic that makes sense if one understands numpy arrays
    poly = np.reshape(polygon,(4,1,2))
    cv2.polylines(image, [poly],1, (0,0,255))
    M = cv2.getRotationMatrix2D(center,360-angle,1) # M.shape =  (2, 3)
    rotatedpolygon = cv2.transform(poly,M)
    print("rotatedpolygon:",rotatedpolygon.shape)
    cv2.polylines(image,[rotatedpolygon],True,(255,255,255))
    cv2.circle(image,(int(rotatedpolygon[0][0][0]),int(rotatedpolygon[0][0][1])),radius =4,color = (255,0,0), thickness = 2)
    cv2.circle(image,(int(rotatedpolygon[1][0][0]),int(rotatedpolygon[1][0][1])),radius =4,color = (255,0,0), thickness = 2)
    cv2.circle(image,(int(rotatedpolygon[2][0][0]),int(rotatedpolygon[2][0][1])),radius =4,color = (255,0,0), thickness = 2)
    cv2.circle(image,(int(rotatedpolygon[3][0][0]),int(rotatedpolygon[3][0][1])),radius =4,color = (255,0,0), thickness = 2)
    print("rotatedpolygon:",rotatedpolygon)
    x,y,w,h = cv2.boundingRect(rotatedpolygon)
    print(x,y,w,h)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),2)
    cv2.imwrite("crop_face.png",image)
    return (x,y,x+w,y+h)
    
 
if __name__ == "__main__":
    img_path = "/home/yiling/code/frontFace/002.jpg"
    image = cv2.imread(img_path,1)
    image = imutils.rotate_bound(image, 350)
    rects = detect(image)
    if len(rects)==0:
        print("found 0 faces")
        exit(-1)
    landmarksList=landmark(image,rects)
    if len(landmarksList)==0:
        print("found no landmark")
        exit(-1)
    for landmark in landmarksList:
        points=dlib_points(landmark)
        rot_img,(x1,y1,x2,y2)=warp_affine(image, points, scale=1.0)
        cv2.imwrite("rot_img.png",rot_img)
        crop_face(image, points)