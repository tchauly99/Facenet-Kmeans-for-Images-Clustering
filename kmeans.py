import cv2
import sys
import os
import time
import tensorflow as tf
import numpy
from align.detect_face import *
from face_contrib import *
from facenet import *
from align.align_mtcnn import  *

def find_embedding(data_dir, models):
    #find image paths from the dir given
    image_paths = os.listdir(data_dir)
    paths_list = [os.path.join(data_dir, image_path) for image_path in image_paths]
    paths=numpy.array(paths_list)
    #call the function for finding faces in an image
    recog=Recognition(models,'models/your_model.pkl')
    #init lists
    vectors = []
    bb = []
    dis=[]
    without_face=[]
    for i in range(0,len(paths)):
        #get image fromm the path
        img=cv2.imread(paths[i])
        #get faces from the img
        faces=recog.identify(img)
        #if there is no face detected
        #record the index of it in the list for later use
        if len(faces) == 0:
            print("can't detect a face")
            print(paths[i])
            without_face.append(i)
            continue
        #if there are more than one face
        #only record one with the biggest bounding box
        elif len(faces) > 1:
            for j in range(0, len(faces)):
                bounding_box=faces[j].bounding_box.astype(int)
                point1=numpy.array((bounding_box[0],bounding_box[1]))
                point2=numpy.array((bounding_box[2],bounding_box[3]))
                #find the distance from its top left point to its bottom right point
                dis.append(numpy.linalg.norm(point1-point2))
            max_value=max(dis)
            max_index=dis.index(max_value)
            dis=[]
            faces[0]=faces[max_index]
        #get the bounding box and the embedding vector
        # to the list previously defined
        face_bb = faces[0].bounding_box.astype(int)
        embedding=faces[0].embedding
        bb.append(face_bb)
        vectors.append(embedding)
    #delete the image with no face detected from the paths list
    for i in without_face:
        del paths_list[i]
    #change the lists to arrays
    paths=numpy.array(paths_list)
    bound=numpy.array(bb)
    data=numpy.array(vectors)
    print(bound.shape, type(bound))
    print(data.shape, type(data))
    #get embedding vectors as data for K-means clustering
    data = numpy.float32(data)
    # maximum number of iterations is 10
    # Required accuracy is 1.0
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    #with 3 clusters expected
    #maximum number of iterations is 10
    #Required accuracy is 1.0
    # get the labels
    ret, label, center = cv2.kmeans(data, 3, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    #draw bounding boxes and appropriate labels on images
    #for i in range(center.shape[0]):
     #   p=vectors.index(center[i,:])
      #  img = cv2.imread(paths_list[p])
       # cv2.imshow('center', img)
        #cv2.rectangle(img, (bound[p, 0], bound[p, 1]), (bound[p, 2], bound[p, 3]), (0, 0, 255), 2)
        #cv2.putText(img, str(label[p, 0]), (bound[p, 0], bound[p, 3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
         #           (0, 255, 0), 2, 2)

    for i in range(0, len(paths)):
        img = cv2.imread(paths[i])
        cv2.rectangle(img, (bound[i,0], bound[i,1]), (bound[i,2], bound[i,3]), (0,0,255), 2)
        cv2.putText(img , str(label[i,0]), (bound[i,0], bound[i,3] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0,255,0), 2, 2)
        cv2.imshow('done',img)
        #press "n" to next
        #press "q" to quit the program
        while (1):
            if cv2.waitKey(0) & 0xFF == ord('n'):
                quit=0
                break
                cv2.destroyAllWindows()
            elif cv2.waitKey(0) & 0xFF == ord('q'):
                quit=1
                break
        if quit==1:
            break
            cv2.destroyAllWindows()
#align_mtcnn('your_face1', 'face_align1')
#call the function with dir to images and facenet models
find_embedding('your_face1/', 'models')


