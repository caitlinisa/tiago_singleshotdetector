import h5py
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
import scipy
from sklearn.cluster import DBSCAN


class Dataloader:
    def __init__(self):
        self.directory_img_data = "/home/cornel/Data/images.mat"
        self.directory_label_data = "/home/cornel/Data/labels.mat"
        self.directory_names_data = "/home/cornel/Data/names.mat"


    def load_files(self):
        images = scipy.io.loadmat(self.directory_img_data)
        label = scipy.io.loadmat(self.directory_label_data)
        names = scipy.io.loadmat(self.directory_names_data)
        return images,label,names
    
    def convert_to_numpy(self):
        images,labels,names= self.load_files()
        datim = list(images['images'])
        #print (names)
        np_image = np.array(datim, dtype = np.uint8)
        datlab = list(labels['labels'])
        np_labels = np.array(datlab, dtype = np.uint8)
        datname = list(names['names'])
        np_names = np.array(datname, dtype = np.object)
        print(np_image.shape,np_labels.shape,np_names.shape)
        print(np_names)
        return np_image,np_labels,np_names
    
    def get_bounding_box(self):
        pass

    #Cluster the labels so that the points can be assigned to a certain cluster
    #This is needed so that one can extract the bounding boxes
    def get_cluster(self,np_labels)
        (h,w,l) = np_labels.shape
        for index in range(l):
            image_label = np_labels[:,:,index]
            db = DBSCAN(eps = 0.3, min_samples= 10).fit(image_label)
            core_samples_mask = np.zeros_like(db.labels_, dtype= bool)
            core_samples_mask[db.core_samples_indices_] = True
            labels = db.labels_
            n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
            

    
    def visualize_images(self,np_image,index):
        plt.imshow(np_image[:,:,:,index])
        plt.show() 

    def visualize_labels(self,np_labels,index):
        plt.imshow(np_labels[:,:,index],interpolation='none')
        plt.show()
        


dl = Dataloader()
np_image,np_labels,np_names = dl.convert_to_numpy()
dl.visualize_images(np_image,0)
dl.visualize_labels(np_labels,0)
#print (label[0])
