import torch
from torch.utils.data import Dataset
import nibabel as nib
import numpy as np
import os
import random


class AbideDataset(Dataset):
    def __init__(self, img_folder, num_of_patches=16, training=True):
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

        # num_classes = len(np.unique(imageData_g))
        # print(num_classes)
        # new_imageData_g = np.zeros(imageData_g.shape)
        # for i, l in enumerate(np.unique(imageData_g)):
        #     new_imageData_g[imageData_g == l] = i
        #
        # imageData_g = new_imageData_g

        half_patch_shape = (13, 13, 13)
        patch_shape = (27, 27, 27)

        half_label_patch_shape = (4, 4, 4)
        label_patch_shape = (9, 9, 9)

        patchesList_modal_1 = np.zeros((self.num_of_patches, 1) + (patch_shape))
        patchesList_modal_2 = np.zeros((self.num_of_patches, 1) + (patch_shape))
        patchesList_modal_g = np.zeros((self.num_of_patches, 1) + (patch_shape))
        for p in range(self.num_of_patches):
            x = np.random.randint(half_patch_shape[0], imageData_1.shape[0] - half_patch_shape[0] - 1)
            y = np.random.randint(half_patch_shape[1], imageData_1.shape[1] - half_patch_shape[1] - 1)
            z = np.random.randint(half_patch_shape[2], imageData_1.shape[2] - half_patch_shape[2] - 1)

            patchesList_modal_1[p, ...] = imageData_1[x - half_patch_shape[0]:x + half_patch_shape[0] + 1,
                                                   y - half_patch_shape[1]:y + half_patch_shape[1] + 1,
                                                   z - half_patch_shape[2]:z + half_patch_shape[2] + 1]

            patchesList_modal_2[p, ...] = imageData_2[x - half_patch_shape[0]:x + half_patch_shape[0] + 1,
                                       y - half_patch_shape[1]:y + half_patch_shape[1] + 1,
                                       z - half_patch_shape[2]:z + half_patch_shape[2] + 1]

            patchesList_modal_g[p, ...] = imageData_g[x - half_label_patch_shape[0]:x + half_label_patch_shape[0] + 1,
                                       y - half_label_patch_shape[1]:y + half_label_patch_shape[1] + 1,
                                       z - half_label_patch_shape[2]:z + half_label_patch_shape[2] + 1]


        return patchesList_modal_1, patchesList_modal_2, patchesList_modal_g