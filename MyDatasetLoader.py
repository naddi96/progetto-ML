from torch_geometric.data import Data
import numpy as np
import torch
from torch_geometric.data import InMemoryDataset
import random


import torch
import random
import torch
from torch_geometric.data import Data


from torch_geometric.data import DataLoader,InMemoryDataset
import numpy as np

class MyDatasetLoader(InMemoryDataset):
    
    
    
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        #self.num_node_features=1433
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['cora.cites', 'cora.content']

    @property
    def processed_file_names(self):
        return ['cites.pt',"content.pt"]

    def download(self):
        pass

    def process(self): 
        feat_data, labels, edge_index, word_count, labels_count = self.load_cora_raw()
        edge_index= torch.tensor(edge_index)
        labels=  torch.tensor(labels, dtype=torch.long)
        num_nodes=2708
        feat_data=torch.tensor(feat_data,dtype=torch.float32)
        data = Data(x=feat_data,
                    edge_index=edge_index,
                    y=labels,
                    word_count=word_count,
                    labels_count=labels_count)
        data.train_mask= torch.zeros([2708], dtype=torch.bool) 
        data.train_mask.fill_(True)
        data.test_mask= torch.zeros([2708], dtype=torch.bool) 
        data.test_mask.fill_(False)
        data.val_mask= torch.zeros([2708], dtype=torch.bool) 
        data.val_mask.fill_(False)

        sampled_test = random.sample(range(num_nodes), int(num_nodes*0.40))
        data.train_mask[sampled_test]=False
        sampled_val=[]
        data.x[sampled_test]= torch.zeros([1433], dtype=torch.float32)
        for x in range(0,int(num_nodes*0.1)):
            num=sampled_test.pop()
            sampled_val.append(num)
        
        data.val_mask[sampled_val]=True
        data.test_mask[sampled_test]=True
        

        data = data if self.pre_transform is None else self.pre_transform(data)
        torch.save(self.collate([data]), self.processed_paths[0])

    @staticmethod
    def load_cora_raw():
        word_count={}
        for x in range(0,1433):
            word_count[x]=0
        labels_count={}
        numero_nodi=2708
        numero_feature=1433
        feat_data = np.zeros((numero_nodi, numero_feature))
        labels = np.empty((numero_nodi), dtype=np.int32)
        node_map = {}
        label_map = {}
        with open("./dataset-raw/cora.content") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
            # print(info[1:-1])
                data =[]
                for x in info[1:-1]:
                    data.append(float(x))
                node_map[info[0]]=i
                feat_data[i,:] = data
                
                if not info[-1] in label_map:
                    label_map[info[-1]] = len(label_map)
                    labels_count[info[-1]]=1    

                labels_count[info[-1]]=labels_count[info[-1]]+1
                labels[i] = label_map[info[-1]]

                

                for i,val in enumerate(data):
                    word_count[i]=word_count[i]+val
        
        from collections import defaultdict
        adj_lists = defaultdict(set)
        edge_list1=[]
        edge_list2=[]
        with open("./dataset-raw/cora.cites") as fp:
            for i,line in enumerate(fp):
                info = line.strip().split()
                paper1 = node_map[info[0]]
                paper2 = node_map[info[1]]

                edge_list1.append(paper1)
                edge_list1.append(paper2)
                edge_list2.append(paper2)
                edge_list2.append(paper1)


        return feat_data, labels , [edge_list1,edge_list2],word_count,labels_count
