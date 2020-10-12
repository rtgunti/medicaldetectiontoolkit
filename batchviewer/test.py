# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 12:30:15 2020

@author: rtgun
"""
from batchviewer import view_batch
import numpy as np
import nibabel as nib

data = nib.load("d010_pre0_dat.nii.gz")
data = data.get_fdata()
print(data.shape)

data = data[None]

data1 = np.transpose(data, axes=(0, 3, 1, 2))
print(data1.shape)

data2 = np.transpose(data, axes=(0, 3, 2, 1))
print(data2.shape)

data3 = np.rot90(data1, k=2, axes=(-2, -1))
print(data3.shape)

data4 = np.rot90(data2, k=2, axes=(-1, -2))
print(data4.shape)

data_f = np.concatenate((data1, data2, data3, data4))

print(data_f.shape)
view_batch(data1, width=500, height=500)
