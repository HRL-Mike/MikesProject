
import os
import glob

from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from transformers import ViTFeatureExtractor, AutoFeatureExtractor


class EndoVis18VQAGPTClassification(Dataset):  # use this one
    '''
    	seq: train_seq  = [2, 3, 4, 6, 7, 9, 10, 11, 12, 14, 15]
    	     val_seq    = [1, 5, 16]
    	folder_head     = 'dataset/EndoVis-18-VQA/seq_'
    	folder_tail     = '/vqa/Classification/*.txt'
    '''
    def __init__(self, seq, folder_head, folder_tail, model_ver = None, transform=None):

        if model_ver == "efvlegpt2ViT":  # 这里的self.image_processor要改
            self.transform = None
            self.image_processor = ViTFeatureExtractor.from_pretrained('google/vit-base-patch16-224-in21k')
        elif model_ver == "efvlegpt2Swin":
            self.transform = None
            self.image_processor = AutoFeatureExtractor.from_pretrained("microsoft/swin-tiny-patch4-window7-224")
        elif transform:
            self.transform = transform
        else:
            self.transform = transforms.Compose([
                transforms.Resize((300, 256)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

        # files, question and answers
        filenames = []
        for curr_seq in seq:
            filenames = filenames + glob.glob(folder_head + str(curr_seq) + folder_tail)
            # print('filenames: ', filenames)
            # /home/mikehe/projects/SurgicalGPT/dataset/EndoVis-18-VQA/seq_2/vqa/Classification/frame000_QA.txt
        self.vqas = []
        for file in filenames:
            file_data = open(file, "r")
            lines = [line.strip("\n") for line in file_data if line != "\n"]
            file_data.close()
            for line in lines:
                self.vqas.append([file, line])
        # print('vqas: ', self.vqas)
        # ['/home/mikehe/projects/SurgicalGPT/dataset/EndoVis-18-VQA/seq_2/vqa/Classification/frame000_QA.txt',
        # 'What organ is being operated?|kidney']
        print('Total files: %d | Total question: %.d' % (len(filenames), len(self.vqas)))
        # self.vqas每一行是文本路径和QA

        # Labels
        self.labels = ['kidney', 'Idle', 'Grasping', 'Retraction', 'Tissue_Manipulation',
                       'Tool_Manipulation', 'Cutting', 'Cauterization', 'Suction',
                       'Looping', 'Suturing', 'Clipping', 'Staple', 'Ultrasound_Sensing',
                       'left-top', 'right-top', 'left-bottom', 'right-bottom']

    def __len__(self):
        return len(self.vqas)

    def __getitem__(self, idx):
        # dataset/EndoVis-18-VQA/seq_
        # /home/mikehe/projects/SurgicalGPT/dataset/EndoVis-18-VQA/seq_2
        loc = self.vqas[idx][0].split('/')
        # print('loc: ', loc)

        # img
        img_loc = os.path.join('/', loc[1], loc[2], loc[3], loc[4], loc[5], loc[6], loc[7], 'left_fr', loc[-1].split('_')[0] + '.png')
        # print(img_loc)
        if self.transform:
            img = Image.open(img_loc)
            img = self.transform(img)
        else:
            img = self.image_processor(Image.open(img_loc), return_tensors="pt")

        # question and answer
        question = self.vqas[idx][1].split('|')[0]
        label = self.labels.index(str(self.vqas[idx][1].split('|')[1]))

        return os.path.join(loc[0], loc[1], loc[2], 'left_frames', loc[-1].split('_')[0] + '.png'), img, question, label
    # 这么处理数据，是为了符合模型的输入

