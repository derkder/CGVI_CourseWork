import torch
torch.manual_seed(42)
from PIL import Image

class CTDataset(torch.utils.data.Dataset):
    '''
    This utility corresponds to the Dataset class construction for the custom example of camera trap data.
    The class is responsible for how images and the corresponding data are loaded in batches during training or evaluation.
    '''   
    def __init__(self,annotation_dict,transform):
        
        ### TODO: Given annotation dictionary for a given split type (train,validation or test) and the selected set of transformations, 
        ### we initialize the properties of the class (img_paths, targets, species). Hint: loop through dictionary rows.
        self.img_paths = []
        self.targets = []
        self.species = []
        
        self.transform = transform
        ## TODO: Fill properly the values in the three attribute lists (img_paths, targets, species) defined below
        
        for item in annotation_dict:
            self.img_paths.append(item[0])
            self.targets.append(item[1])
            self.species.append(item[2])
        
        
        
                
    def __len__(self):
        return len(self.img_paths)
     
    def get_image(self,img_path):  

        with open(img_path, 'rb') as f:
            try:
                img = Image.open(f)
                return img.convert('RGB')
            except:
                print("Image from {} cannot be read. It will be skipped".format(img_path))
                return None
            
    def __getitem__(self, idx):        
        ### TODO: Given an index we want to return the corresponding dictionary (op) with 'img_path','target','species' as its keys, along with 
        ### 'img' that corresponds to the transformed image. This routine is typically called at a batch level during train/validation.
        img_path = self.img_paths[idx]
        img = self.get_image(img_path)
        
        op = {}
        
        ### TODO: Comment out the following placeholder commands that load values in the dictionary to be returned 
        ### (the existing lines return the same image always!) and replace them with the corresponding values to be loaded at every batch.
        op['target'] = self.targets[idx]
        op['img_path'] = img_path
        op['species'] = self.species[idx]
        op['img'] =  self.transform(img)
        return op