from keras import backend as K
import time
from multiprocessing.dummy import Pool
K.set_image_data_format('channels_first')
import cv2
import os
import glob
import numpy as np
from numpy import genfromtxt
import tensorflow as tf
from fr_utils import *
from inception_blocks_v2 import *
import win32com.client as wincl

PADDING = 50
ready_to_detect_identity = True #全局变量，用来确保：如果程序正在识别另一个人的时候，不再识别新的人
windows10_voice_interface = wincl.Dispatch("SAPI.SpVoice") #调用win10的文字转语音功能，发出声音

FRmodel = faceRecoModel(input_shape=(3, 96, 96))

#triplet_loss损失函数计算loss，比softmax效果好很多，可以最小化类内距离，最大化类间距离
def triplet_loss(y_true, y_pred, alpha = 0.3):
    """
    Implementation of the triplet loss as defined by formula (3)

    FaceNet使用了一种名为Triplet Loss的独特方法来计算Loss。Triplet loss最小化了一个anchor与一个positive之间的距离，
    这些图像包含相同的标识，并最大化了anchor与negative的图像之间的距离，这些图像包含不同的身份

    Arguments:
    y_pred -- python list containing three objects:
            anchor -- the encodings for the anchor images, of shape (None, 128)
            positive -- the encodings for the positive images, of shape (None, 128)
            negative -- the encodings for the negative images, of shape (None, 128)
    
    Returns:
    loss -- real number, value of the loss
    """
    #以下代码就是实现了Triplet Loss的数学公式，具体可搜索公式

    anchor, positive, negative = y_pred[0], y_pred[1], y_pred[2]
    
    # Step 1: Compute the (encoding) distance between the anchor and the positive, you will need to sum over axis=-1
    pos_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, positive)), axis=-1)
    # Step 2: Compute the (encoding) distance between the anchor and the negative, you will need to sum over axis=-1
    neg_dist = tf.reduce_sum(tf.square(tf.subtract(anchor, negative)), axis=-1)
    # Step 3: subtract the two previous distances and add alpha.
    basic_loss = tf.add(tf.subtract(pos_dist, neg_dist), alpha)
    # Step 4: Take the maximum of basic_loss and 0.0. Sum over the training examples.
    loss = tf.reduce_sum(tf.maximum(basic_loss, 0.0))
    
    return loss

FRmodel.compile(optimizer = 'adam', loss = triplet_loss, metrics = ['accuracy']) #设置网络参数，梯度优化器，loss函数等等
load_weights_from_FaceNet(FRmodel) #加载模型权重

#生成人脸识别数据库，也就是这个数据库中的人脸都是注册过的
#只需要把带有人脸的图像放入images文件夹即可实现人脸注册
def prepare_database():
    database = {}

    # load all the images of individuals to recognize into the database
    for file in glob.glob("images/*"):
        identity = os.path.splitext(os.path.basename(file))[0] #使用图像文件名作为唯一id
        database[identity] = img_path_to_encoding(file, FRmodel)#将图像传入到模型中提取特征，返回特征值向量
    return database


#打开网络摄像头读取图像，识别人脸
def webcam_face_recognizer(database):
    """
    Runs a loop that extracts images from the computer's webcam and determines whether or not
    it contains the face of a person in our database.

    If it contains a face, an audio message will be played welcoming the user.
    If not, the program will process the next frame from the webcam
    """
    global ready_to_detect_identity

    cv2.namedWindow("preview")
    vc = cv2.VideoCapture(0)

    #使用opencv中的人脸检测模块
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    
    while vc.isOpened():
        _, frame = vc.read()
        img = frame

        # We do not want to detect a new identity while the program is in the process of identifying another person
        #如果程序正在识别另一个人的时候，不再识别新的人
        if ready_to_detect_identity:
            img = process_frame(img, frame, face_cascade)#进行人脸检测和识别
        
        key = cv2.waitKey(100)
        cv2.imshow("preview", img)

        if key == 27: # exit on ESC
            break
    cv2.destroyWindow("preview")

#进行具体的人脸检测和识别
def process_frame(img, frame, face_cascade):
    """
    Determine whether the current frame contains the faces of people from our database
    """
    global ready_to_detect_identity
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) #检测人脸区域

    # Loop through all the faces detected and determine whether or not they are in the database
    #将检测到的人脸区域抽取特征后和数据库中的人脸特征进行比对
    identities = []
    for (x, y, w, h) in faces:
        x1 = x-PADDING
        y1 = y-PADDING
        x2 = x+w+PADDING
        y2 = y+h+PADDING

        img = cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)

        identity = find_identity(frame, x1, y1, x2, y2)#将当前人脸和数据库中的每一个注册人脸进行比对

        #如果是相似度大于给定的阈值
        if identity is not None:
            identities.append(identity)

    #如果人脸一致
    if identities != []:
        cv2.imwrite('example.png', img)

        ready_to_detect_identity = False #表示识别到用户了，暂时不进行人脸识别
        pool = Pool(processes=1) #线程池
        # We run this as a separate process so that the camera feedback does not freeze
        pool.apply_async(welcome_users, [identities]) #将识别之后的语音提示单独放入一个线程中，否则会导致主线程阻塞
    return img


#人脸比对，将当前人脸和数据库中一一比对
def find_identity(frame, x1, y1, x2, y2):
    """
    Determine whether the face contained within the bounding box exists in our database

    x1,y1_____________
    |                 |
    |                 |
    |_________________x2,y2

    """
    height, width, channels = frame.shape
    # The padding is necessary since the OpenCV face detector creates the bounding box around the face and not the head
    part_image = frame[max(0, y1):min(height, y2), max(0, x1):min(width, x2)]  #截取人脸区域
    
    return who_is_it(part_image, database, FRmodel)

#使用模型提取当前人脸的特征值，并和数据库中的一一比对
def who_is_it(image, database, model):
    """
    Implements face recognition for the happy house by finding who is the person on the image_path image.
    
    Arguments:
    image_path -- path to an image
    database -- database containing image encodings along with the name of the person on the image
    model -- your Inception model instance in Keras
    
    Returns:
    min_dist -- the minimum distance between image_path encoding and the encodings from the database
    identity -- string, the name prediction for the person on image_path
    """
    encoding = img_to_encoding(image, model) #提取人脸特征值
    
    min_dist = 100
    identity = None
    
    # Loop over the database dictionary's names and encodings.
    #将当前人脸特征值和数据库中的一一比对
    for (name, db_enc) in database.items():
        
        # Compute L2 distance between the target "encoding" and the current "emb" from the database.
        dist = np.linalg.norm(db_enc - encoding) #使用L2距离计算人脸相似度距离

        print('distance for %s is %s' %(name, dist))

        # If this distance is less than the min_dist, then set min_dist to dist, and identity to name
        #依次比对，只保留相似度最高
        if dist < min_dist:
            min_dist = dist
            identity = name

    #如果相似度距离大于阈值则判定为不一致，否则判定为同一人
    if min_dist > 0.82:
        return None
    else:
        return str(identity)

#对识别到的用户说出语音消息
def welcome_users(identities):
    """ Outputs a welcome audio message to the users """
    global ready_to_detect_identity
    welcome_message = 'Welcome '

    if len(identities) == 1:
        welcome_message += '%s, have a nice day.' % identities[0]
    else:
        for identity_id in range(len(identities)-1):
            welcome_message += '%s, ' % identities[identity_id]
        welcome_message += 'and %s, ' % identities[-1]
        welcome_message += 'have a nice day!'

    windows10_voice_interface.Speak(welcome_message)

    # Allow the program to start detecting identities again
    ready_to_detect_identity = True #语音播报完成，则表示当前识别过程完成了，可以继续识别其他用户了

if __name__ == "__main__":
    database = prepare_database() #得到人脸注册数据库
    webcam_face_recognizer(database) #开启摄像头，开始人脸识别

# ### References:
# 
# - Florian Schroff, Dmitry Kalenichenko, James Philbin (2015). [FaceNet: A Unified Embedding for Face Recognition and Clustering](https://arxiv.org/pdf/1503.03832.pdf)
# - Yaniv Taigman, Ming Yang, Marc'Aurelio Ranzato, Lior Wolf (2014). [DeepFace: Closing the gap to human-level performance in face verification](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf) 
# - The pretrained model we use is inspired by Victor Sy Wang's implementation and was loaded using his code: https://github.com/iwantooxxoox/Keras-OpenFace.
# - Our implementation also took a lot of inspiration from the official FaceNet github repository: https://github.com/davidsandberg/facenet 
# 
