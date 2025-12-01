import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import os
import numpy as np
from tqdm import tqdm
import OpenEXR
import Imath
import matplotlib.pyplot as plt
import cv2
import imageio

def lookat(eye, target, up):
    def normalize(v):
        norm = np.linalg.norm(v)
        if norm == 0: 
            return v
        return v / norm
    mz = normalize( (eye[0]-target[0], eye[1]-target[1], eye[2]-target[2]) ) # inverse line of sight
    mx = normalize( np.cross( up, mz ) )
    my = normalize( np.cross( mz, mx ) )
    tx = -np.dot( mx, eye )
    ty = -np.dot( my, eye )
    tz = -np.dot( mz, eye )   
    return np.array([mx[0], mx[1], mx[2], tx, 
                     my[0], my[1], my[2], ty,
                     mz[0], mz[1], mz[2], tz,
                     0.0,   0.0,   0.0,   1.0], dtype=np.float32).reshape((4, 4))

def perspective(fov, aspect, near, far):
    frustumDepth = far - near
    oneOverDepth = 1 / frustumDepth

    result = np.zeros((4, 4), dtype=np.float32)
    result[1][1] = -1 / np.tan(0.5 * fov)
    result[0][0] = -result[1][1] / aspect
    result[2][2] = -far * oneOverDepth
    result[2][3] = (-far * near) * oneOverDepth
    result[3][2] = -1
    result[3][3] = 0
    return result

coefs = {
    "classroom": 10.0,
    "living-room-2": 5.4,
    "dining-room": 11.0,
    "living-room": 5.2,
    "bedroom": 5.2,
    "kitchen": 6.2,
    "staircase": 8.0,
    'living-room-3': 4.8,
    'bathroom': 4.0,
    'bathroom2': 4.0,
    'deadtree': 6.0,
    'window': 10.0,
    'cornell_box_bunny': 4.0,
    'emerald_square': 20.0
}

# for testing
coefs["classroom1"] = coefs["classroom"]
coefs["classroom2"] = coefs["classroom"]
coefs["staircase1"] = coefs["staircase"]
coefs["staircase2"] = coefs["staircase"]
coefs["living-room-31"] = coefs["living-room-3"]
coefs["bedroom1"] = coefs["bedroom"]
coefs["dining-room1"] = coefs["dining-room"]

def read_exr_channel(file_path, channel_name, scale):
    exr_file = OpenEXR.InputFile(file_path)
    header = exr_file.header()
    dw = header['dataWindow']
    width = dw.max.x - dw.min.x + 1
    height = dw.max.y - dw.min.y + 1
    FLOAT = Imath.PixelType(Imath.PixelType.HALF)
    channel_data = exr_file.channel(channel_name, FLOAT)
    channel_data_np = np.frombuffer(channel_data, dtype=np.float16)
    channel_data_np = channel_data_np.astype(np.float32)
    channel_data_np.shape = (height, width)
    return channel_data_np / scale

def read_exr_data(folder_path, exr_file, channels, scale=1.0):
    full_exr_file = os.path.join(folder_path, f"{exr_file}.exr")
    data = [read_exr_channel(full_exr_file, ch, scale) for ch in channels]
    return np.stack(data, axis=0)

class NSSMDataset(Dataset):
    def __init__(self, feature_folder, gt_folder, gt_index_multiplier, use_temporal=False, use_msm=False, use_pcf=False,
                 extra_inputs=None, video_order=False, video_length=10, data_range=None, temporal_group_num=1,
                 cv_div_d=False):
        super().__init__()
        self.feature_folder = feature_folder
        self.gt_folder = gt_folder
        self.gt_index_multiplier = gt_index_multiplier
        #self.scenes = scenes
        #self.folder_names = []
        #for scene in self.scenes:
        #    scene_dir = os.path.join(base_directory, scene)
        #    # sort according to number
        #    temp = sorted([int(f) for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f))])
        #    if data_range is not None:
        #        l, r = data_range
        #        assert l >= 0 and l < r and r <= len(temp)
        #        temp = temp[l:r]
        #    self.folder_names.extend([os.path.join(scene_dir, str(f)) for f in temp])
        #self.len = len(self.folder_names) * 6
        #self.odd_only = odd_only
        #if odd_only:
        #    assert temporal_group_num == 2
        #    assert not video_order
        #    self.len = self.len // 2
        # use gt folder to determine length (file num in gt folder = length)
        self.len = 0
        self.data_indices = []
        for entry in os.listdir(gt_folder):
            full_path = os.path.join(gt_folder, entry)
            # only count .exr files that match the pattern
            if os.path.isfile(full_path) and entry.startswith('Mogwai.ShadowDenoiser.output') and entry.endswith('.exr'):
                # extract the number before .exr
                num_of_data_str = entry[len('Mogwai.ShadowDenoiser.output.'):-len('.exr')]
                num_of_data = int(num_of_data_str) / gt_index_multiplier
                self.data_indices.append(num_of_data)
        self.data_indices = sorted(self.data_indices)
        if data_range is not None:
                l, r = data_range
                assert l >= 0 and l < r and r <= len(self.data_indices)
                self.data_indices = self.data_indices[l:r]
        self.len = len(self.data_indices)

        #self.penumbra_clamp = penumbra_clamp
        self.use_msm = use_msm
        self.use_pcf = use_pcf
        self.extra_inputs = [] if extra_inputs is None else extra_inputs
        #self.video_order = video_order # whether output according to videos' order: for one scene, output 6 video with different light size
        #self.video_length = video_length
        #self.use_temporal = use_temporal
        #if use_temporal:
        #    self.temporal_group_num = temporal_group_num
        #    assert temporal_group_num in [2]
        #    assert not video_order
        self.cv_div_d = cv_div_d
        #self.penumbra_width_choice = penumbra_width_choice # not using this
        print(f"Loading Dataset with scenes {self.scenes}, in total {self.len} data")
    
    def __len__(self):
        return self.len

    def parse_run_info(self, folder_path, scale=1.0):
        run_info_path = os.path.join(folder_path, "run_info.txt")
        run_info = {}
        if os.path.exists(run_info_path):
            with open(run_info_path, 'r') as file:
                for line in file:
                    if line.startswith("Light Position:"):
                        run_info['light_position'] = np.array(list(map(float, line.split(":")[1].strip().strip('[]').split(',')))) / scale
                    elif line.startswith("Camera Position:"):
                        run_info['camera_position'] = np.array(list(map(float, line.split(":")[1].strip().strip('[]').split(',')))) / scale
                    elif line.startswith("Camera Direction:"):
                        run_info['camera_direction'] = np.array(list(map(float, line.split(":")[1].strip().strip('[]').split(','))))
        return run_info
        
    def mygetitem(self, idx, is_perturb): 
        #folder_path = self.folder_names[folder_idx]        
        #scene_id = int(os.path.split(folder_path)[-1])
        #scene_name = os.path.split(os.path.split(folder_path)[-2])[-1]

        #coef = coefs[scene_name]
        coef = coefs["emerald_square"]

        # TODO: add normal and worldPos if needed in future
        #normW = read_exr_data(folder_path, 'Mogwai.GBuffer.normW.15', ['R', 'G', 'B'])
        #posW = read_exr_data(folder_path, 'Mogwai.GBuffer.posW.15', ['R', 'G', 'B'], coef)

        data_idx = self.data_indices[idx]

        distRtoB = read_exr_data(self.feature_folder, f'Mogwai.NSSMFeaturePass.distRtoB.{data_idx}', ['R'], coef)
        distVtoR = read_exr_data(self.feature_folder, f'Mogwai.NSSMFeaturePass.distVtoR.{data_idx}', ['R'], coef)
        stdDev = read_exr_data(self.feature_folder, f'Mogwai.NSSMFeaturePass.projectedShadowDepthStd.{data_idx}', ['R','G'], coef)
        shadowMap = read_exr_data(self.feature_folder, f'Mogwai.NSSMFeaturePass.shadowMask.{data_idx}', ['R'])
        ce = read_exr_data(self.feature_folder, f'Mogwai.NSSMFeaturePass.ce.{data_idx}', ['R'])
        cv = read_exr_data(self.feature_folder, f'Mogwai.NSSMFeaturePass.cv.{data_idx}', ['R'])

        gt_index = data_idx * self.gt_index_multiplier
        gt = read_exr_data(self.gt_folder, f'Mogwai.ShadowDenoiser.output.{gt_index}', ['R'])

        #info = self.parse_run_info(folder_path, coef)

        mask = distRtoB != 0 # we don't care about pixels that are not in shadow receiver
        H, W = gt.shape[-2:]

        # replace kpnsm penumbra width with Shadow depth divergence map
        # this is already normalized
        shadowDvg = read_exr_data(self.feature_folder, f'Mogwai.NSSMFeaturePass.projectedDivergenceMap.{data_idx}', ['R'])

        res = {
            "scene": 0,  # dummy scene id
            "scene_id": 0,
            "id": data_idx,
            #"posW": posW,
            #"normW": normW,
            "distVtoR": distVtoR,
            "distRtoB": distRtoB,
            "shadowMap": shadowMap,
            "ce": ce,
            "cv": cv / (distVtoR + 1e-10) if self.cv_div_d else cv,
            #"info": info,
            "gt": gt,
            "mask": mask,
            "stdDev": stdDev, # this has two channels
            "shadowDvg": shadowDvg,
        }
            
        # for exporting colored results
        #if "mscolor" in self.extra_inputs:
        #    mscolor = read_exr_data(folder_path, 'Mogwai.AccumulatePassColor.output.90.exr' , ['R', 'G', 'B'])
        #    res["mscolor"] = mscolor

        return res

    def __getitem__(self, idx):
        return self.mygetitem(idx, False)

if __name__ == "__main__":
    
# --------start penumbra width--------
    ### calculate 95 percent large
    scenes = ["classroom", "kitchen", "living-room", "staircase", "living-room-2"]
    dataset = NSSMDataset('/data/', '/data/mogwai_gt_renders', gt_index_multiplier=10, use_temporal=False,
                          data_range=[0, 100], use_msm=False, use_temporal=False, temporal_group_num=2)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=16)

    hist = np.zeros((len(1), 1000), dtype=np.float32)
    for i, batch_data in tqdm(enumerate(dataloader)):
        # print(batch_data["id"], batch_data["perturb_id"])
        for k in batch_data.keys():
            if isinstance(batch_data[k], torch.Tensor):
                assert not torch.isnan(batch_data[k].min())
                # assert batch_data[k].min() > -1.05
                # assert not torch.isnan(batch_data[k].max())
                # assert batch_data[k].max() < 1.05
        image = batch_data["shadowDvg"] * 1000
        image_hist, _ = np.histogram(image, bins=1000, range=(0, 1000))
        hist[0] += image_hist

    for i, scene in enumerate(scenes):
        cdf = np.cumsum(hist[i]) / np.sum(hist[i])

        threshold_index = np.searchsorted(cdf, 0.95)
        print(f"scene {scene} 95 percent corresponds to", threshold_index)
        threshold_index = np.searchsorted(cdf, 0.98)
        print(f"scene {scene} 98 percent corresponds to", threshold_index)



