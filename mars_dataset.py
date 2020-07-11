import torch
from PIL import Image
from collections import OrderedDict
import os
import random

class Mars(torch.utils.data.Dataset):
    def __init__(self, root, train=True, transform=None, seed=10):
        self.triplets = []
        self.labels = []
        self.transform = transform
        
        images = {}
        
        for split in os.listdir(root):
            split_dir = '{}/{}'.format(root, split)
            
            # training set
            if train and split == 'bbox_train':
                for index, classname in enumerate(os.listdir(split_dir)):
                    class_dir = '{}/{}'.format(split_dir, classname)
                    images[index] = []

                    for filename in os.listdir(class_dir):
                        file_path = '{}/{}'.format(class_dir, filename)
                        images[index].append(file_path)

            # test set
            elif not train and split == 'bbox_test':
                for dirname in os.listdir(split_dir):
                    set_dir = '{}/{}'.format(split_dir, dirname)

                    for filename in os.listdir(set_dir):
                        file_path = '{}/{}'.format(set_dir, filename)
                        classname = (int(filename[filename.find('T') + 1 : filename.find('F')]))

                        if not classname in images:
                            images[classname] = []

                        images[classname].append(file_path)

        
        random.seed(seed)
        keys = list(images.keys())

        # split the images into triplets
        for classname in images:

            count = 0
            triplets_out = []
            labels_out = []

            for filename in images[classname]:
                # choose random class for negative
                pos_class = classname
                neg_class = pos_class

                if len(keys) > 1:
                    while neg_class == pos_class:
                        neg_class = random.choice(keys)
                
                # choose random pos/neg examples
                anchor = filename
                pos = filename
                neg = random.choice(images[neg_class])

                if len(images[pos_class]) > 1:
                    while pos == filename:
                        pos = random.choice(images[pos_class])
            
                triplets_out.append((anchor, pos, neg))
                labels_out.append((pos_class, pos_class, neg_class))
                count += 1

                if count == 4:
                    self.triplets.append(triplets_out)
                    self.labels.append(labels_out)

                    count = 0
                    triplets_out = []
                    labels_out = []



    def __len__(self):
        return len(self.triplets)


    def __getitem__(self, index):
        triplets_out = []
        labels_out = []

        for triplet in self.triplets[index]:
            anchor, pos, neg = triplet
            triplet = (Image.open(anchor), Image.open(pos), Image.open(neg))
            
            if self.transform is not None:
                triplet = [self.transform(img) for img in triplet]
            
            triplets_out.append(triplet)
            
        
        for label in self.labels[index]:
            labels_out.append(label)

        return triplets_out, labels_out

if __name__ == '__main__':
    mars = Mars('./.data')
    print(len(mars))
    print(mars[0][1])
    print(mars[1][1])
    print(mars[2][1])
    print(mars[3][1])
    print(mars[4][1])
    mars[0][0][0][0].show()