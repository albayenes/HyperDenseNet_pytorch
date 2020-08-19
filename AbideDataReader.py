import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import random


class AbideDataset(Dataset):
    def __init__(self, img_folder, num_of_patches=128, training=True):
        self.training = training
        self.num_of_patches = num_of_patches
        self.img_folder = img_folder
        self.subj_list = os.listdir(self.img_folder)
        random.seed(9001)
        nums = [x for x in range(len(self.subj_list))]
        random.shuffle(nums)

        if self.training:
            self.training_subj_list = [self.subj_list[index] for index in nums]
        else:
            self.valid_subj_list = [self.subj_list[index] for index in nums]


    def __len__(self):
        if self.training:
            return len(self.training_subj_list)
        else:
            return len(self.valid_subj_list)


    def __getitem__(self, item):
        if self.training:
            self.img_modality_1 = os.path.join(self.img_folder, self.training_subj_list[item], 'mri', 'T1.nii.gz')
            self.img_modality_2 = os.path.join(self.img_folder, self.training_subj_list[item], 'mri', 'T1.nii.gz')
            self.img_segmentation = os.path.join(self.img_folder, self.training_subj_list[item], 'mri', 'aseg.nii.gz')
        else:
            self.img_modality_1 = os.path.join(self.img_folder, self.training_subj_list[item], 'mri', 'T1.nii.gz')
            self.img_modality_2 = os.path.join(self.img_folder, self.training_subj_list[item], 'mri', 'T1.nii.gz')
            self.img_segmentation = os.path.join(self.img_folder, self.training_subj_list[item], 'mri', 'aseg.nii.gz')

        imageData_1 = nib.load(self.img_modality_1).get_data()
        imageData_2 = nib.load(self.img_modality_2).get_data()
        imageData_g = nib.load(self.img_segmentation).get_data()

        num_classes = len(np.unique(imageData_g))
        print(num_classes)
        new_imageData_g = np.zeros(imageData_g.shape)
        for i, l in enumerate(np.unique(imageData_g)):
            new_imageData_g[imageData_g == l] = i

        imageData_g = new_imageData_g

        patch_shape = (27, 27, 27)


        patchesList_modal_1 = []
        patchesList_modal_2 = []
        patchesList_modal_g = []
        for p in range(self.num_of_patches):
            x = np.random.randint(patch_shape[0] + imageData_1.shape[0] - patch_shape[0])
            y = np.random.randint(patch_shape[1] + imageData_1.shape[1] - patch_shape[1])
            z = np.random.randint(patch_shape[2] + imageData_1.shape[2] - patch_shape[2])

            patchesList_modal_1.append(imageData_1[x:x + patch_shape[0],
                               y:y + patch_shape[1],
                               z:z + patch_shape[2]])

            patchesList_modal_2.append(imageData_2[x:x + patch_shape[0],
                                       y:y + patch_shape[1],
                                       z:z + patch_shape[2]])

            patchesList_modal_g.append(imageData_g[x:x + patch_shape[0],
                                       y:y + patch_shape[1],
                                       z:z + patch_shape[2]])

        patches_modal_1 = np.concatenate(patchesList_modal_1, axis=0).reshape((len(patchesList_modal_1), 1) + patch_shape)
        patches_modal_2 = np.concatenate(patchesList_modal_1, axis=0).reshape((len(patchesList_modal_1), 1) + patch_shape)
        patches_modal_g = np.concatenate(patchesList_modal_g, axis=0).reshape((len(patchesList_modal_1), 1) + patch_shape)

        return patches_modal_1, patches_modal_2, patches_modal_g