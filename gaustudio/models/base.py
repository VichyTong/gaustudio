import numpy as np
from typing import Dict
from plyfile import PlyData, PlyElement
import torch.nn as nn
import torch

class BasePointCloud(nn.Module):
    def __repr__(self):
        properties = self.config["attributes"].keys()
        return f"{self.__class__.__name__}(num_points={self.num_points}, properties={properties})"

    def __init__(self, config, device=None) -> None:
        super().__init__()
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.config = {**self.default_conf, **config}
        self.setup(device)
        self.setup_functions()

    @torch.no_grad()
    def to(self, device):
        self.device = device
        for elem in self.config["attributes"]:
            if elem == 'xyz':
                self._xyz = self._xyz.to(device)
            elif elem == 'opacity':
                self._opacity = self._opacity.to(device)
            else:
                setattr(self, '_'+elem, getattr(self, '_'+elem).to(device))
        return self
    
    @property
    def get_center_and_size(self):
        import numpy as np
        _xyz = self._xyz.cpu().numpy()
        lower_bound = np.percentile(_xyz, 5, axis=0)
        upper_bound = np.percentile(_xyz, 95, axis=0)
        center = (lower_bound + upper_bound) / 2
        size = upper_bound - lower_bound
        return center, size
    
    @property
    def get_center(self):
        min_xyz, _ = torch.min(self._xyz, dim=0)
        max_xyz, _ = torch.max(self._xyz, dim=0)
        return (min_xyz + max_xyz) / 2
    
    def setup(self, device,  num_points = 0):
        self.device = device
        self.num_points = num_points
        for elem in self.config["attributes"]:
            dummy_data = torch.empty(num_points, device=device)
            setattr(self, '_'+elem, dummy_data)
    
    def setup_functions(self):
        pass
    
    def update(self, **args):
        for elem in self.config["attributes"]:
            if elem in args:
                setattr(self, '_'+elem, args[elem])
        self.num_points = self._xyz.shape[0]
        
    def create_from_attribute(self, **args):
        for elem in args:
            tensor_value = torch.as_tensor(args[elem])
            setattr(self, '_' + elem, tensor_value)
            self.num_points = tensor_value.shape[0]
        self.config["attributes"] = list(args.keys())
        
    def calculate_center(self):
        if not hasattr(self, '_xyz'):
            raise ValueError("XYZ coordinates are not loaded. Make sure to load the PLY file and include 'xyz' in the attributes.")
        
        # Get the xyz coordinates
        xyz = self._xyz.cpu().numpy()  # Convert the tensor to a numpy array for easier manipulation
        
        # Calculate the minimum and maximum coordinates along each axis
        min_coords = xyz.min(axis=0)
        max_coords = xyz.max(axis=0)
        
        # Create the bounding box
        bounding_box = {
            'min_x': min_coords[0],
            'min_y': min_coords[1],
            'min_z': min_coords[2],
            'max_x': max_coords[0],
            'max_y': max_coords[1],
            'max_z': max_coords[2],
        }
        
        center = [
            (min_coords[0] + max_coords[0]) / 2,
            (min_coords[1] + max_coords[1]) / 2,
            (min_coords[2] + max_coords[2]) / 2,
        ]
        return center
    
    def load(self, ply_path: str):
        plydata = PlyData.read(ply_path)  
        self.num_points = plydata['vertex'].count

        for elem in self.config["attributes"]:
            if elem == 'xyz':
                xyz = np.stack((plydata.elements[0]['x'], 
                                     plydata.elements[0]['y'],
                                     plydata.elements[0]['z']), axis=1)
                self._xyz = torch.from_numpy(xyz).float().to(self.device)
                
            elif elem == 'opacity':
                opacity = plydata.elements[0]['opacity'][..., np.newaxis]
                self._opacity = torch.from_numpy(opacity).float().to(self.device)
                  
            elif elem == 'rgb':
                rgb = np.stack((plydata.elements[0]['red'],
                                plydata.elements[0]['green'],
                                plydata.elements[0]['blue']), axis=1)
                self._rgb = torch.from_numpy(rgb).float().to(self.device) / 255
            else:
                names = [n.name for n in plydata.elements[0].properties if n.name.startswith(elem)]
                names = sorted(names, key=lambda n: int(n.split('_')[-1]))
                if len(names) == 0:
                    continue
                
                
                data = np.zeros((self.num_points, len(names)))
                for i, name in enumerate(names):
                    data[:,i] = plydata.elements[0][name]
                setattr(self, '_'+elem, torch.from_numpy(data).float().to(self.device))

        print(f"Loaded {self.num_points} points from {ply_path}")
    
    def get_attribute(self, attribute):
        return getattr(self, '_'+attribute)