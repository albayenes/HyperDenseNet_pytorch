



from os.path import isfile, join
import os
import numpy as np
from sampling import reconstruct_volume
from sampling import my_reconstruct_volume
from sampling import load_data_trainG
from sampling import load_data_test

import torch
import torch.nn as nn
from HyperDenseNet import *
# from medpy.metric.binary import dc,hd
import argparse

import pdb
from torch.autograd import Variable
from progressBar import printProgressBar
import nibabel as nib

from AbideDataReader import AbideDataset

DEVICE = torch.device("cuda")


def runTraining(opts):
    print('' * 41)
    print('~' * 50)
    print('~~~~~~~~~~~~~~~~~  PARAMETERS ~~~~~~~~~~~~~~~~')
    print('~' * 50)
    print('  - Number of image modalities: {}'.format(opts.numModal))
    print('  - Number of classes: {}'.format(opts.numClasses))
    print('  - Directory to load images: {}'.format(opts.root_dir))
    for i in range(len(opts.modality_dirs)):
        print('  - Modality {}: {}'.format(i+1,opts.modality_dirs[i]))
    print('  - Directory to save results: {}'.format(opts.save_dir))
    print('  - To model will be saved as : {}'.format(opts.modelName))
    print('-' * 41)
    print('  - Number of epochs: {}'.format(opts.numClasses))
    print('  - Batch size: {}'.format(opts.batchSize))
    print('  - Number of samples per epoch: {}'.format(opts.numSamplesEpoch))
    print('  - Learning rate: {}'.format(opts.l_rate))
    print('' * 41)

    print('-' * 41)
    print('~~~~~~~~  Starting the training... ~~~~~~')
    print('-' * 41)
    print('' * 40)

    samplesPerEpoch = opts.numSamplesEpoch
    batch_size = opts.batchSize

    lr = opts.l_rate
    epoch = opts.numEpochs
    
    root_dir = opts.root_dir
    model_name = opts.modelName

    if not (len(opts.modality_dirs)== opts.numModal): raise AssertionError

    moda_1 = root_dir
    moda_2 = root_dir

    if (opts.numModal == 3):
        moda_3 = root_dir + 'Training/' + opts.modality_dirs[2]

    moda_g = root_dir

    print(' --- Getting image names.....')
    print(' - Training Set: -')
    if os.path.exists(moda_1):
        imageNames_train = [f for f in os.listdir(moda_1) if isfile(join(moda_1, f, 'mri', 'T1.nii.gz'))][0:1]
        imageNames_train.sort()
        print(' ------- Images found ------')
        for i in range(len(imageNames_train)):
            print(' - {}'.format(imageNames_train[i])) 
    else:
        raise Exception(' - {} does not exist'.format(moda_1))

    img_data_folder = '/media/albayenes/vpa-med4.data/Users/albayenes/sub_brain_segmentation/dataset/abide/HyperDensetNetDataset'
    train_set = AbideDataset(img_data_folder)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=1, num_workers=0, shuffle=True)

    valid_set = AbideDataset(img_data_folder)
    valid_loader = torch.utils.data.DataLoader(valid_set, batch_size=1, num_workers=0, shuffle=True)


    num_classes = 30
    # print("~~~~~~~~~~~ Creating the model ~~~~~~~~~~")
    # num_classes = opts.numClasses
    #
    # Define HyperDenseNet
    # To-Do. Get as input the config settings to create different networks
    if (opts.numModal == 2):
        hdNet = HyperDenseNet_2Mod(num_classes)
    if (opts.numModal == 3):
        hdNet = HyperDenseNet(num_classes)
    #

    '''try:
        hdNet = torch.load(os.path.join(model_name, "Best_" + model_name + ".pkl"))
        print("--------model restored--------")
    except:
        print("--------model not restored--------")
        pass'''

    softMax = nn.Softmax(dim=1)
    CE_loss = nn.CrossEntropyLoss().to(DEVICE)
    

    hdNet.to(DEVICE)
    softMax.to(DEVICE)
    CE_loss.to(DEVICE)

    # To-DO: Check that optimizer is the same (and same values) as the Theano implementation
    optimizer = torch.optim.Adam(hdNet.parameters(), lr=lr, betas=(0.9, 0.999))
    
    print(" ~~~~~~~~~~~ Starting the training ~~~~~~~~~~")
    numBatches = int(samplesPerEpoch/batch_size)
    dscAll = []
    for e_i in range(epoch):
        hdNet.train()
        
        lossEpoch = []

        for i, (patches_modal_1, patches_modal_2, patches_modal_g) in enumerate(train_loader):
            optimizer.zero_grad()

            patches_modal_1 = patches_modal_1.to(DEVICE).float()
            patches_modal_2 = patches_modal_2.to(DEVICE).float()
            patches_modal_g = patches_modal_g.to(DEVICE).long()

            segmentation_prediction = hdNet(patches_modal_1, patches_modal_2)
            
            predClass_y = softMax(segmentation_prediction)

            # To adapt CE to 3D
            # LOGITS:
            # segmentation_prediction = segmentation_prediction.permute(0,2,3,4,1).contiguous()
            # segmentation_prediction = segmentation_prediction.view(segmentation_prediction.numel() // num_classes, num_classes)
            
            CE_loss_batch = CE_loss(predClass_y, patches_modal_g)
            
            loss = CE_loss_batch
            loss.backward()
            
            optimizer.step()
            lossEpoch.append(CE_loss_batch.cpu().data.numpy())


            print(loss)

        if not os.path.exists(model_name):
            os.makedirs(model_name)

        np.save(os.path.join(model_name, model_name + '_loss.npy'), dscAll)

        print(' Epoch: {}, loss: {}'.format(e_i,np.mean(lossEpoch)))


        if (100+e_i%20)==0:
             lr = lr/2
             print(' Learning rate decreased to : {}'.format(lr))
             for param_group in optimizer.param_groups:
                param_group['lr'] = lr
                        
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root_dir', type=str, default='./Data/MRBrainS/DataNii/', help='directory containing the train and val folders')
    parser.add_argument('--modality_dirs', nargs='+', default=['T1','T2_FLAIR'], help='subdirectories containing the multiple modalities')
    parser.add_argument('--save_dir', type=str, default='./Results/', help='directory ot save results')
    parser.add_argument('--modelName', type=str, default='HyperDenseNet_2Mod', help='name of the model')
    parser.add_argument('--numModal', type=int, default=2, help='Number of image modalities')
    parser.add_argument('--numClasses', type=int, default=4, help='Number of classes (Including background)')
    parser.add_argument('--numSamplesEpoch', type=int, default=1000, help='Number of samples per epoch')
    parser.add_argument('--numEpochs', type=int, default=500, help='Number of epochs')
    parser.add_argument('--batchSize', type=int, default=10, help='Batch size')
    parser.add_argument('--l_rate', type=float, default=0.0002, help='Learning rate')

    opts = parser.parse_args()
    print(opts)
    
    runTraining(opts)
