import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from torch.utils.data import Dataset
from torchvision import transforms
import glob

from PIL import Image

class SampleData(Dataset):
    def __init__(self, path, rfps = 10, ofps = 30):
        self.path = path
        self.rfps = rfps
        self.ofps = ofps

        self.pil_to_tensor = transforms.ToTensor()
        
        self.frame_files = glob.glob(f'{self.path}/frame*.jpg')
        self.num_frames = len(self.frame_files)

    def __len__(self):
        return int(self.num_frames * self.rfps / self.ofps)

    def __getitem__(self, index):
        frame_idx = int(index * self.ofps / self.rfps)
        img = Image.open(f'{self.path}/frame{frame_idx}.jpg')
        return self.pil_to_tensor(img)
    
if __name__ == "__main__":
    sample = SampleData("data/images/1")
    
    print(f"Dataset length: {len(sample)}")
    if len(sample) > 0:
        frame_tensor = sample[0]
        print(f"Frame shape: {frame_tensor.shape}")
        print(f"Frame tensor type: {frame_tensor.dtype}")
    else:
        print("No frames available in dataset")