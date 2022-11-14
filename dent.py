import os
import torch
import numpy as np
from torchvision import transforms
from model.VFIT_B import UNet_3D_3D

os.environ["CUDA_VISIBLE_DEVICES"]='1'

device = torch.device('cuda')

from model.VFIT_B import UNet_3D_3D

def PercentileRescaler(Arr):

    minval=np.percentile(Arr, 0, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)
    maxval=np.percentile(Arr, 100, axis=None, out=None, overwrite_input=False, interpolation='linear', keepdims=False)

    if minval==maxval:
        print("Zero Detected")
    Arr=(Arr-minval)/(maxval-minval)
    Arr=np.clip(Arr, 0.0, 1.0)

    return Arr, minval, maxval

def RestoreRescaler(Arr, minval, maxval):
    arr= Arr*(maxval-minval)+(minval)
    arr = np.clip(arr, minval, maxval)
    return arr

load_from = './checkpoints_large_dataset_2022_10_03/model_best.pth'

model = UNet_3D_3D(n_inputs=4, joinType="concat")
model = torch.nn.DataParallel(model).to(device)

model_dict = model.state_dict()
model.load_state_dict(torch.load(load_from)["state_dict"] , strict=True)
model.eval();

def apply_model(slice_arr):
    """ Interpolate new frames given `slice_arr` if size (T, nx, ny), where T is the total number of cardiac phases.
    
    """
    
    T = transforms.Compose([transforms.ToTensor()])

    with torch.no_grad():

        interpolated = None 
        for center_phase_idx in range(len(slice_arr)):

            if center_phase_idx < len(slice_arr) - 1:
                sample_ids = [center_phase_idx-2, center_phase_idx-1, center_phase_idx, center_phase_idx+1]
            else:
                sample_ids = [center_phase_idx-2, center_phase_idx-1, center_phase_idx, 0]

            samples, minval, maxval = PercentileRescaler(slice_arr[np.array(sample_ids)]) # normalize 16-bit array to 0-1

            samples = (255*np.repeat(samples[...,None], 3, -1)).astype('uint8') 

            images = [T(samples[tk])[None] for tk in range(4)]
            images = [img.cuda() for img in images]

            torch.cuda.synchronize()
            
            out = model(images)
            out = np.clip(out.detach().cpu().numpy(), 0, 1)
            out = RestoreRescaler(out, minval, maxval)

            if interpolated is None:
                interpolated = out[:,0]
            else:
                interpolated = np.concatenate((interpolated, out[:,0]))
            torch.cuda.synchronize()
            
    return interpolated

