import numpy as np
import torch
import torchvision.transforms as T

def preprocess(state):
    
    # convert state to numpy array and then to torch tensor
    frame = torch.from_numpy(np.array(state).astype(np.float32))
    
    # reshape so that grayscaling is possible
    reshaped_frame = frame.reshape(4, 3, 210, 160)
    
    # grayscale image
    gray_frame = T.Grayscale()(reshaped_frame)
    
    # reshape image so network can process it
    reshaped_gray_frame = gray_frame.reshape(1, 4, 210, 160)
    
    # downscale image to 84x84
    small_gray_frame = T.Resize((84, 84))(reshaped_gray_frame)
    
    return small_gray_frame

def preprocess_two(state):
    
    # add additional dimension to numpy array
    frame_plus_dim = np.expand_dims(state, 0)
    
    # convert state to numpy array and then to torch tensor
    return torch.from_numpy(np.array(frame_plus_dim).astype(np.float32))