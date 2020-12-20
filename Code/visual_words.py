import os, multiprocessing
from os.path import join, isfile

import numpy as np
import sklearn
from sklearn.cluster import KMeans
from scipy.spatial import distance
from numpy import load
from PIL import Image
import scipy.ndimage
import skimage.color
from util import *

def extract_filter_responses(opts, img):
    '''
    Extracts the filter responses for the given image.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    [output]
    * filter_responses: numpy.ndarray of shape (H,W,3F)
    '''
    
    filter_scales = opts.filter_scales

    img_size=img.shape

    if len(img_size)==2:
        img=img.reshape((img_size[0],img_size[1],1))
        img_size=img.shape
    if img_size[2]==1:
        a=np.concatenate((img,img),axis=2)
        img=np.concatenate((a,img),axis=2)
    img=skimage.color.rgb2lab(img)
    color=img_size[2]
    real_img=img
    F=4*len(filter_scales)
    filter_img=np.zeros([img_size[0],img_size[1],3*F])
    for i in range(img_size[2]):
        for j in range(len(filter_scales)):
            filter_img[:,:,color*4*j+i]=scipy.ndimage.gaussian_filter(img[:,:,i],sigma =filter_scales[j])
            filter_img[:,:,color*4*j+3+i]=scipy.ndimage.gaussian_laplace(img[:,:,i],sigma =filter_scales[j])
            filter_img[:,:,color*4*j+6+i] = scipy.ndimage.gaussian_filter(img[:, :, i], sigma=filter_scales[j],order=[1, 0])
            filter_img[:,:,color*4*j+9+i]=scipy.ndimage.gaussian_filter(img[:,:,i], sigma=filter_scales[j],order=[0,1])

    return filter_img

def compute_dictionary_one_image(args,response):
    '''
    Extracts a random subset of filter responses of an image and save it to disk
    This is a worker function called by compute_dictionary

    Your are free to make your own interface based on how you implement compute_dictionary
    '''


    response_size=response.shape
    flat_response=np.reshape(response,(response_size[0]*response_size[1],response_size[2]))
    alpha_samples = np.random.randint(response_size[0]* response_size[1], size=args.alpha)
    sampled_response = flat_response[alpha_samples,:]

    return sampled_response

def compute_dictionary(opts, n_worker=1):
    '''
    Creates the dictionary of visual words by clustering using k-means.

    [input]
    * opts         : options
    * n_worker     : number of workers to process in parallel
    
    [saved]
    * dictionary : numpy.ndarray of shape (K,3F)
    '''

    data_dir = opts.data_dir
    feat_dir = opts.feat_dir
    out_dir = opts.out_dir
    K = opts.K

    train_files = open(join(data_dir, 'train_files.txt')).read().splitlines()
    alpha=opts.alpha

    check=0
    for file in train_files:
        img=Image.open(join(data_dir,file))
        img = np.array(img).astype(np.float32) / 255
        filter_response=extract_filter_responses(opts,img)
        one_dictionary=compute_dictionary_one_image(opts,filter_response)
        if check==0:
            dictionary=one_dictionary
        else:
            dictionary = np.concatenate((dictionary,one_dictionary), axis=0)
        check+=1

    np.save(join(out_dir,'new_dictionary.npy'), dictionary)
    training_data = np.load(join(out_dir, 'new_dictionary.npy'))
    kmeans = sklearn.cluster.KMeans(n_clusters=K).fit(training_data)
    visual_words = kmeans.cluster_centers_
    np.save(join(out_dir, 'dictionary.npy'), visual_words)




def get_visual_words(opts, img, dictionary):
    '''
    Compute visual words mapping for the given img using the dictionary of visual words.

    [input]
    * opts    : options
    * img    : numpy.ndarray of shape (H,W) or (H,W,3)
    
    [output]
    * wordmap: numpy.ndarray of shape (H,W)
    '''
    

    filter_img=extract_filter_responses(opts,img)
    filter_size=filter_img.shape

    wordmap=np.zeros([filter_size[0]*filter_size[1]])
    flat_filter_img=np.reshape(filter_img,(filter_size[0]*filter_size[1],filter_size[2]))
    dist=distance.cdist(flat_filter_img,dictionary)
    for i in range(filter_size[0]*filter_size[1]):

        wordmap[i]=np.argmin(dist[i][:],axis=0)


    wordmap=np.reshape(wordmap,(filter_size[0],filter_size[1]))

    return wordmap
