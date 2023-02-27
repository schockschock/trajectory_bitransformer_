import pandas as pd
import torch
import torchvision
from PIL import Image
import os

# Other variables
T_obs = 8
T_pred = 12
T_total = T_obs + T_pred  # 8+12=20
image_size = 256
in_size = 2
# "dataset_T_length_"+str(T_total)+"delta_coordinates"
table = "dataset_T_length_20delta_coordinates"


class TrajectoryPredictionDataset(torch.utils.data.Dataset):
    # Enc.cinématique reçoit la trajectoire observée de humain cible (input) de la forme T=(u1,u2-u1,u3-u2,..) qui consiste en les coordonnées de la position de départ et en les déplacements relatifs de l'humain entre les images consécutives.
    # Ce format a été choisi car il permet au modèle de mieux capturer les similarités entre des trajectoires presque identiques qui peuvent avoir des points de départ différents.
    def __init__(self, ROOT_DIR, cnx):

        self.pos_df = pd.read_sql_query("SELECT * FROM "+str(table), cnx)
        self.root_dir = ROOT_DIR+'/visual_data'
        self.transform = torchvision.transforms.Compose([torchvision.transforms.Resize((image_size, image_size)),
                                                         torchvision.transforms.ToTensor(),
                                                         torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))])
        self.visual_data = []
        # read sorted frames
        for img in sorted(os.listdir(self.root_dir)):
            self.visual_data.append(self.transform(
                Image.open(os.path.join(self.root_dir)+"/"+img)))
        self.visual_data = torch.stack(self.visual_data)

    def __len__(self):
        return self.pos_df.data_id.max()  # data_id maximum dans dataset

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        #print("idx :", idx)

        # table dont data_id=idx
        extracted_df = self.pos_df[self.pos_df["data_id"] == idx]

        tensor = torch.tensor(extracted_df[['pos_x_delta', 'pos_y_delta']].values).reshape(
            -1, T_total, in_size)  # juste pos_x_delta et pos_y_delta de extracted_df (tensor)
        # obs de 8 et pred de 12 à partir de tensor construit
        obs, pred = torch.split(tensor, [T_obs, T_pred], dim=1)

        # extracted_df dont data_id=idx, on prend minimum frame_num et aprés on divise par 10, cela represente start_frame
        start_frames = (extracted_df.groupby(
            'data_id').frame_num.min().values/10).astype('int')
        extracted_frames = []
        for i in start_frames:
            extracted_frames.append(self.visual_data[i:i+T_obs])
        # stack concatenates a sequence of tensors along a new dimension.
        frames = torch.stack(extracted_frames)
        start_frames = torch.tensor(start_frames)  # tensor([start_frames])
        return obs, pred, frames, start_frames
