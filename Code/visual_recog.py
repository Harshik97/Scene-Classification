import os, math, multiprocessing
from os.path import join
from copy import copy

import numpy as np
from PIL import Image
from opts import get_opts
import visual_words
from visual_words import *

def get_feature_from_wordmap(opts, wordmap):
    '''
    Compute histogram of visual words.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist: numpy.ndarray of shape (K)
    '''

    K = opts.K

    map_shape=wordmap.shape
    hist_data=np.reshape(wordmap,(1,map_shape[0]*map_shape[1]))
    bins=np.linspace(0,K,num=K+1,endpoint=True)
    histogram,_=np.histogram(hist_data,bins=bins,density=True)
    histogram=np.reshape(histogram,(1,K))
    return histogram

def get_feature_from_wordmap_SPM(opts, wordmap):
    '''
    Compute histogram of visual words using spatial pyramid matching.

    [input]
    * opts      : options
    * wordmap   : numpy.ndarray of shape (H,W)

    [output]
    * hist_all: numpy.ndarray of shape (K*(4^L-1)/3)
    '''
        
    K = opts.K
    L = opts.L

    map_size=wordmap.shape
    weights=np.zeros((1,L+1))
    weights[0][0]=2**(-L)
    weights[0][1]=2**(-L)
    for l in range(2,L+1):
        weights[0][l]=2**(l-L-1)

    layer=[i for i in range(L+1)]
    count=0
    for i in range(len(weights)):

        weight=weights[0][len(weights)-i-1]
        l=layer[len(layer)-i-1]
        factor=2**l
        new_h=int(map_size[0]/factor)
        new_w=int(map_size[1]/factor)


        for j in range(factor):

            for k in range(factor):


                new_map=wordmap[new_h*j:new_h*(j+1),new_w*k:new_w*(k+1)]

                histogram=get_feature_from_wordmap(opts,new_map)
                if count==0:

                    weighted_hist=histogram*weight
                else:
                    weighted_hist=np.concatenate((weighted_hist,histogram*weight),axis=1)
                count+=1

    return weighted_hist




def get_image_feature(opts, img_path, dictionary):
    '''
    Extracts the spatial pyramid matching feature.

    [input]
    * opts      : options
    * img_path  : path of image file to read
    * dictionary: numpy.ndarray of shape (K, 3F)


    [output]
    * feature: numpy.ndarray of shape (K)
    '''


    img = Image.open(join(opts.data_dir,img_path))
    img = np.array(img).astype(np.float32) / 255
    wordmap=get_visual_words(opts,img,dictionary)
    SPM=get_feature_from_wordmap_SPM(opts,wordmap)
    return SPM

def build_recognition_system(opts, n_worker=1):
    '''
    Creates a trained recognition system by generating training features from all training images.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [saved]
    * features: numpy.ndarray of shape (N,M)
    * labels: numpy.ndarray of shape (N)
    * dictionary: numpy.ndarray of shape (K,3F)
    * SPM_layer_num: number of spatial pyramid layers
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir
    SPM_layer_num = opts.L

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    train_labels = np.loadtxt(join(data_dir, 'train_labels.txt'), np.int32)
    dictionary = np.load(join(out_dir, 'dictionary.npy'))


    start=0
    for file in train_files:
        feature=get_image_feature(opts,file,dictionary)
        if start==0:
            features=feature
        else:
            features=np.concatenate((features,feature),axis=0)

        start+=1


    np.savez_compressed(join(out_dir, 'trained_system.npz'),
         features=features,
         labels=train_labels,
         dictionary=dictionary,
         SPM_layer_num=SPM_layer_num,
    )

def distance_to_set(word_hist, histograms):
    '''
    Compute similarity between a histogram of visual words with all training image histograms.

    [input]
    * word_hist: numpy.ndarray of shape (K)
    * histograms: numpy.ndarray of shape (N,K)

    [output]
    * sim: numpy.ndarray of shape (N)
    '''


    total_samples=histograms.shape
    most_similar=np.zeros((1,total_samples[0]))
    one=np.ones((1,total_samples[0]))

    minima=np.minimum(word_hist,histograms)
    similarity=np.sum(minima,axis=1)
    distance=one-similarity

    return distance


    
def evaluate_recognition_system(opts, n_worker=1):
    '''
    Evaluates the recognition system for all test images and returns the confusion matrix.

    [input]
    * opts        : options
    * n_worker  : number of workers to process in parallel

    [output]
    * conf: numpy.ndarray of shape (8,8)
    * accuracy: accuracy of the evaluated system
    '''

    data_dir = opts.data_dir
    out_dir = opts.out_dir

    trained_system = np.load(join(out_dir, 'trained_system.npz'))
    dictionary = trained_system['dictionary']
    histograms=trained_system['features']
    # using the stored options in the trained system instead of opts.py
    test_opts = copy(opts)
    test_opts.K = dictionary.shape[0]
    test_opts.L = trained_system['SPM_layer_num']
    train_labels=trained_system['labels']
    test_files = open(join(data_dir, 'test_files.txt')).read().splitlines()
    test_labels = np.loadtxt(join(data_dir, 'test_labels.txt'), np.int32)


    correct=0

    c_matrix = np.zeros((8, 8))
    index=0


    for file in test_files:
        test_hist=get_image_feature(opts,file,dictionary)

        test_label=test_labels[index]
        index+=1
        distance=distance_to_set(test_hist,histograms)
        matched_train_label=np.where(distance==np.amin(distance))
        predicted_label=train_labels[matched_train_label[1]]


        c_matrix[test_label,predicted_label[0]]+=1

    accuracy=np.trace(c_matrix)/np.sum(c_matrix).sum()
    return c_matrix,accuracy*100




