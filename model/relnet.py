import torch
from torch import nn
from torch.nn.init import kaiming_uniform_, normal_
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.init as init
from torch.autograd import Variable
                                                                                
import collections                                                              
import functools                                                                
import torchvision.models as models                                                   
import torch.nn.functional as F                                                 
from transformers import AutoModel                                              
from transformers import AutoTokenizer                                          
import os                                                                       
import numpy as np                                                              
import pandas as pd                                                             
import torch                                                                    
from torch.utils.data import Dataset, DataLoader                                
from torchvision.transforms import transforms                                   
from PIL import Image                                                           
from transformers import ViltProcessor, ViltForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split                            
import transformers 

class RelationNetworks(nn.Module):
    def __init__(
        self,
        n_vocab=64,
        conv_hidden=24,
        embed_hidden=32,
        lstm_hidden=128,
        mlp_hidden=256,
        classes=3129,
    ):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
            nn.Conv2d(conv_hidden, conv_hidden, [3, 3], 2, 1, bias=False),
            nn.BatchNorm2d(conv_hidden),
            nn.ReLU(),
        )

        self.embed = nn.Embedding(n_vocab, 64)
        self.lstm = nn.LSTM(64, lstm_hidden, batch_first=True)

        self.n_concat = conv_hidden * 2 + lstm_hidden + 2 * 2

        self.g = nn.Sequential(
            nn.Linear(self.n_concat, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
        )

        self.f = nn.Sequential(
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(mlp_hidden, classes),
        )

        self.conv_hidden = conv_hidden
        self.lstm_hidden = lstm_hidden
        self.mlp_hidden = mlp_hidden

        coords = torch.linspace(-4, 4, 14)
        x = coords.unsqueeze(0).repeat(14, 1)
        y = coords.unsqueeze(1).repeat(1, 14)
        coords = torch.stack([x, y]).unsqueeze(0)
        self.register_buffer('coords', coords)

    def forward(self, image, question, question_len):
        conv = self.conv(image)
        batch_size, n_channel, conv_h, conv_w = conv.size()
        n_pair = conv_h * conv_w
        question = torch.clamp(question, max=63)
        embed = self.embed(question)
        embed_pack = nn.utils.rnn.pack_padded_sequence(
            embed, question_len, batch_first=True
        )
        _, (h, c) = self.lstm(embed_pack)
        h_tile = h.permute(1, 0, 2).expand(
            batch_size, n_pair * n_pair, self.lstm_hidden
        )

        conv = torch.cat([conv, self.coords.expand(batch_size, 2, conv_h, conv_w)], 1)
        n_channel += 2
        conv_tr = conv.view(batch_size, n_channel, -1).permute(0, 2, 1)
        conv1 = conv_tr.unsqueeze(1).expand(batch_size, n_pair, n_pair, n_channel)
        conv2 = conv_tr.unsqueeze(2).expand(batch_size, n_pair, n_pair, n_channel)
        conv1 = conv1.contiguous().view(-1, n_pair * n_pair, n_channel)
        conv2 = conv2.contiguous().view(-1, n_pair * n_pair, n_channel)

        concat_vec = torch.cat([conv1, conv2, h_tile], 2).view(-1, self.n_concat)
        g = self.g(concat_vec)
        g = g.view(-1, n_pair * n_pair, self.mlp_hidden).sum(1).squeeze()

        f = self.f(g)

        return f

class VQA(Dataset):
    def __init__(self, image_dirs, root_dir, key, csv_file, transform=None):
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.image_dirs = image_dirs
        self.transform = transform
        self.tokenizer = transformers.AutoTokenizer.from_pretrained('bert-base-uncased')
        self.model1=ViltForQuestionAnswering.from_pretrained("dandelin/vilt-b32-finetuned-vqa")
        # Build a dictionary that maps image IDs to rows in the CSV file
        self.id_to_rows = {}
        for _, row in self.df.iterrows():
            image_id = int(row['image_id'])
            if image_id not in self.id_to_rows:
                self.id_to_rows[image_id] = []
            self.id_to_rows[image_id].append(row)
        print("here")
        #model.eval()

        # Build a list of (image_path, question, label) tuples
        self.samples = []
        for image_dir in self.image_dirs:
            image_dir = os.path.join(self.root_dir, image_dir)
            for filename in os.listdir(image_dir):
                if filename.endswith('.jpg'):
                    image_id = int(filename.split('_')[-1].split('.')[0])
                    try:
                        row = self.id_to_rows[image_id]
                        for i in row:
                            if(key=="paraphrase"):
                               paraphrased_questions = i["paraphrase"][2:-2]
                            else:
                               paraphrased_questions = i["question"]
                            answer = i['answer']
                            self.samples.append((os.path.join(image_dir, filename), paraphrased_questions, answer))
                    except:
                           pass

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Load the image and apply the transform
        image_path, question, answer = self.samples[idx]
        with Image.open(image_path) as image:
            if self.transform is not None:
                image = self.transform(image)

        # Convert the answer to a tensor
        tensor = torch.zeros(3129)
        try:
            label = self.model1.config.label2id[answer]
        except KeyError:
            label=545
        tensor[label] = 1
        encoded = self.tokenizer(
            question,  
            padding="max_length", 
            max_length=64, 
            truncation=True,  
            return_tensors='pt'
        )
        #self.model.eval()
        return image, encoded['input_ids'].squeeze(), tensor

# Define the transform to apply to the images
transform = transforms.Compose([                                                
    transforms.Resize((224, 224)),                                              
    transforms.ToTensor(),                                                      
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),                     
])


def collate_fn(batch):
    images, lengths, answers, questions = [], [], [], []
    for image, question, tensor in batch:
        #print(image,question,tensor)
        images.append(image)
        lengths.append(len(question))
        answers.append(tensor)
        #print(type(question))
        questions.append(question)
    #print(lengths)
    images = torch.stack(images, dim=0)
    answers = torch.stack(answers, dim=0)
    lengths= torch.tensor(lengths)
    padded_questions = torch.nn.utils.rnn.pad_sequence(questions, batch_first=True)
    #print(type(padded_questions))
    return images,padded_questions,answers,lengths


dataset = VQA(image_dirs=['val2014'],root_dir='/home/ubuntu/suraj/',key="paraphrase",csv_file='merged_result1.csv', transform=transform)

train_size = 350000                                                             
val_size = len(dataset) - train_size                                            
print(val_size)                                                                 
                     
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
print(len(train_dataset))                                                       
                                                                                
# Create dataloaders for the training and validation sets                       
train_dataloader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
val_dataloader = DataLoader(val_dataset, batch_size=8, shuffle=False, collate_fn=collate_fn)

#pprint(iter(train_dataloader))                                                         
                                                                                
dataset1 = VQA(image_dirs=['val2014'],root_dir='/home/ubuntu/suraj/', key="question", csv_file='merged_result1.csv', transform=transform)
print(len(dataset1)) 
dataset1 = DataLoader(dataset1, batch_size=16, shuffle=True, collate_fn=collate_fn)    
num_epochs = 25                                                                
                                                                                
# Initialize the best validation accuracy to 0 and the corresponding model state dictionary to None                                                   
relnet=RelationNetworks()
print(relnet)
loss_fn = torch.nn.CrossEntropyLoss()
import torch.optim as optim
optimizer = optim.Adam(relnet.parameters(), lr=0.001)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
best_val_acc = 0                                                                
best_model_state_dict = None

for i in range(num_epochs):                                                             
    relnet.train()                                                               
    for image,question,answer,question_length in train_dataloader:                                                                    
            output = relnet(image,question,question_length).to(device)
            print(output.shape)
            answer=answer.to(device)
            print(output.shape,answer.shape)
            optimizer.zero_grad()
            loss = loss_fn(output, answer)
            loss.backward()
    relnet.eval()
    correct=0
    total=0
    for image,question,answer,question_length in val_dataloader:                                                                    
            output = relnet(image,question,question_length).to(device)
            correct += (output.detach().cpu().argmax(1) == answer.detach().cpu().argmax(1)).sum().item()
            total+=output.shape[0]
    acc=(correct/total)*100        
    print(f"Epoch {i} : validation accuracy:{acc}")
    if acc > best_val_acc:                                                  
        best_val_acc = acc                                                  
        best_model_state_dict = mac.state_dict()                              
        torch.save(best_model_state_dict, 'best_model2_relnet.pt')

correct=0
total=0
for image,question,answer,question_length in dataset1:                                                                    
            output = relnet(image,question,question_length).to(device)
            correct += (output.detach().cpu().argmax(1) == answer.detach().cpu().argmax(1)).sum().item()
            total+=output.shape[0]
acc=(correct/total)*100        
print(f"Test accuracy : {acc}")  
