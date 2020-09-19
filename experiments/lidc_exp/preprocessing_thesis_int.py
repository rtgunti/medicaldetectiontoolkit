#!/usr/bin/env python
# Copyright 2018 Division of Medical Image Computing, German Cancer Research Center (DKFZ).
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

'''
This preprocessing script loads nrrd files obtained by the data conversion tool: https://github.com/MIC-DKFZ/LIDC-IDRI-processing/tree/v1.0.1
After applying preprocessing, images are saved as numpy arrays and the meta information for the corresponding patient is stored
as a line in the dataframe saved as info_df.pickle.
'''

import os
import numpy as np
from multiprocessing import Pool
import pandas as pd
import numpy.testing as npt
from skimage.transform import resize
import subprocess
import pickle
import nibabel

import configs
cf = configs.configs()

def resample_array(img, mask, src_spacing, target_spacing):

    src_spacing = np.round(src_spacing, 3)
    target_shape = [int(img.shape[ix] * src_spacing[ix] / target_spacing[ix]) for ix in range(len(img.shape))]
    for i in range(len(target_shape)):
        try:
            assert target_shape[i] > 0
        except:
            raise AssertionError("AssertionError:", src_imgs.shape, src_spacing, target_spacing)
    img = img.astype(float)
    resampled_img = resize(img, target_shape, order=1, mode='edge').astype('float32')
    resampled_mask = resize(mask, target_shape, order=0, mode='edge').astype('float32')

    return resampled_img, resampled_mask


def pp_patient(inputs):

    ix, (dat, seg) = inputs
    pid = str(dat.split('_')[0]) + '_' + str(dat.split('_')[1])
    
    img = nibabel.load(os.path.join(cf.raw_data_dir, dat))
    mask = nibabel.load(os.path.join(cf.raw_seg_dir, seg))
    
    img_arr = img.get_fdata()
    mask_arr = mask.get_fdata()
    
    img_arr = np.rot90(img_arr)
    mask_arr = np.rot90(mask_arr)
    
    img_arr = np.fliplr(img_arr)   #only for thesis
    mask_arr = np.fliplr(mask_arr)
    
    img_arr *= np.clip(mask_arr, 0, 1)
    
    mask_arr = np.clip(mask_arr, 0, 2)
    mask_arr[mask_arr == 1] = 0
    mask_arr[mask_arr == 2] = 1

    img_arr, mask_arr = resample_array(img_arr, mask_arr, (1.0, 1.0, 2.5), cf.target_spacing)
    
    print('Processing {}'.format(pid), img_arr.shape, mask_arr.shape)
    img_arr = (img_arr - np.mean(img_arr)) / np.std(img_arr).astype(np.float16)
    final_rois = mask_arr
    mal_labels = [0]

#     fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, 0])]
    fg_slices = [ii for ii in np.unique(np.argwhere(final_rois != 0)[:, -1])]
    print(fg_slices)
    mal_labels = np.array(mal_labels)

    np.save(os.path.join(cf.pp_dir_int, '{}_rois.npy'.format(pid)), final_rois)
    np.save(os.path.join(cf.pp_dir_int, '{}_img.npy'.format(pid)), img_arr)

    with open(os.path.join(cf.pp_dir_int, 'meta_info_{}.pickle'.format(pid)), 'wb') as handle:
        meta_info_dict = {'pid': pid, 'class_target': mal_labels, 'fg_slices': fg_slices}
        pickle.dump(meta_info_dict, handle)

def aggregate_meta_info(exp_dir):

    files = sorted([os.path.join(exp_dir, f) for f in os.listdir(exp_dir) if 'meta_info' in f])
    df = pd.DataFrame(columns=['pid', 'class_target', 'fg_slices'])
    for f in files:
        with open(f, 'rb') as handle:
            df.loc[len(df)] = pickle.load(handle)

    df.to_pickle(os.path.join(exp_dir, 'info_df.pickle'))
    print ("aggregated meta info to df with length", len(df))


if __name__ == "__main__":

    data_paths = sorted([path for path in os.listdir(cf.raw_data_dir) if 'int1' in path])
    seg_paths = sorted([path for path in os.listdir(cf.raw_seg_dir) if 'int1' in path])
    paths = [p for p in zip(data_paths, seg_paths)]
#     paths = paths[:4]

    if not os.path.exists(cf.pp_dir_int):
        os.mkdir(cf.pp_dir_int)

    pool = Pool(processes=8)
    p1 = pool.map(pp_patient, enumerate(paths), chunksize=1)
    pool.close()
    pool.join()
    # for i in enumerate(paths):
    #     pp_patient(i)

    aggregate_meta_info(cf.pp_dir_int)
    subprocess.call('cp {} {}'.format(os.path.join(cf.pp_dir_int, 'info_df.pickle'), os.path.join(cf.pp_dir_int, 'info_df_bk.pickle')), shell=True)