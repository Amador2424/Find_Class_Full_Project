import cv2

from mahotas.features import haralick

import mahotas.features as ft 
from BiT import bio_taxo

import numpy as np

import glob


import os

from typing import List

from os import listdir

from skimage.feature import graycomatrix, graycoprops # scikit-image

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

 

 
#création des fonctions
def haralick_with_mean(data):
    all_statistics = ft.haralick(data).mean(0)
    return all_statistics

def fct_BiT(data):

    return bio_taxo(data)


def glcm(data):
    glcm = graycomatrix(data, [2], [0], 256, symmetric=True, normed=True)
    diss = graycoprops(glcm, 'dissimilarity')[0,0]
    cont = graycoprops(glcm, 'contrast')[0,0]
    corr = graycoprops(glcm, 'correlation')[0,0]
    ener = graycoprops(glcm, 'energy')[0,0]
    homo = graycoprops(glcm, 'homogeneity')[0,0]    
    all_statistics = [diss, cont, corr, ener, homo]
    return all_statistics


def convert_class(cls_name ):
    if cls_name =="01_TUMOR": return "TUM"
    elif cls_name=="02_STROMA" : return "STROM"
    elif cls_name=="03_COMPLEX" : return "COM"
    elif cls_name=="04_LYMPHO" : return "LYMP"
    elif cls_name=="05_DEBRIS" : return "DEB"
    elif cls_name=="06_MUCOSA" : return "MUC"
    elif cls_name=="07_ADIPOSE" : return "ADIP"
    
def convert_class2(cls_name ):
    return 0 if cls_name=='Saudavel' else 1

def save_to_csv(features,output_path):

    df = pd.DataFrame(features)

    df.to_csv(output_path, index=False)

   

def process_Img(img_drt,subfolders):

    features=[]

    count =0

    for type in subfolders:

        img_paths = glob.glob(os.path.join(img_drt+type, "*"))

       

        for img_path in img_paths:

            img=cv2.imread(img_path)

            B,G,R = cv2.split(img)

            RGB_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            R_gray = R

            G_gray = G

            B_gray = B

            #print(type,img_paths)


           # print(RGB_feat)

            R_feat = (fct_BiT(R_gray))

            #print(RGB_feat)

            G_feat = (glcm(G_gray))

            #print(RGB_feat)

            B_feat =( haralick_with_mean(B_gray))

            #print(B_feat)
            if img_drt=="REFUGE":
                cls = ([convert_class2(type)])
            else : 
                cls = ([convert_class(type)])

            feature = np.hstack([R_feat , G_feat , B_feat , cls]) 

            #print(feature)

            features.append(feature)

            count+=1

            print(f'Feature Extrait {count}-->{type}')

       

        #print(type, img_paths)

    return features

def normalize(features, output_path):
    X = [row[:-1] for row in features]  # Remove the last column (class)
    Y = [row[-1] for row in features]  # Class labels

    scaler = MinMaxScaler()
    XN = scaler.fit_transform(X)

    normalized_data = np.hstack((XN, np.array(Y).reshape(-1, 1)))
    normalized_df = pd.DataFrame(normalized_data)

    normalized_df.to_csv(output_path, index=False)



#TRAINING 


def main():

    crc_path = r"C:\Users\amado\anaconda3\cours\IA2\envIA2\Salzburg/"

    crc_dir: List[str] = listdir(crc_path)
    crc_output_path = './am.csv'
    crc_features = process_Img(crc_path, crc_dir)
   # save_to_csv(crc_features, crc_output_path)
    
    normalize(crc_features, './Normalized_CRC.csv')  # Normalize and save
    
    rfg_path = r"C:\Users\amado\anaconda3\cours\IA2\LABO1\REFUGE/"
    rfg_dir: List[str] = listdir(rfg_path)
    rfg_output_path = './REFUGE.csv'
    rfg_features = process_Img(rfg_path, rfg_dir)
   # save_to_csv(rfg_features, rfg_output_path)
    
    normalize(rfg_features, './Normalized_REFUGE.csv')  # Normalize and save
   

if __name__ =="__main__":

    main()

   