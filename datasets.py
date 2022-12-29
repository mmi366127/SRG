from torch.utils.data import Dataset
import numpy as np
import torch
import os

class myDataset(Dataset):
    def __init__(self, numClasses: int, numInstances: int, numFeatures: int, fileName: str):
        self.datasetPath = 'dataset'
        self.numClasses = numClasses
        self.numInstance = numInstances
        self.numFeatures = numFeatures
        self.fileName = fileName
        self.label = torch.zeros(numInstances)
        self.data = torch.zeros((numInstances, numFeatures))
        self.readDataset()

    def readDataset(self):
        filename = os.path.join('./', self.datasetPath, self.fileName)
        print(f'read dataset: {filename}')
        with open(filename, 'r') as f:
            for (idx, line) in enumerate(f.readlines()):
                line = line.split()
                self.label[idx] = int(line[0])
                for i in range(1, len(line)):
                    temp = line[i].split(':')
                    self.data[idx, int(temp[0]) - 1] = float(temp[1]) 

    def __getitem__(self, idx):
        return self.data[idx], self.label[idx]

    def __len__(self):
        return self.numInstance

class Mushrooms(myDataset):
    def __init__(self):
        """ 
        mushrooms
        Source: UCI / mushrooms
        Preprocessing: Each nominal attribute is expanded into several binary attributes. The original attribute #12 has missing values and is not used.
        # of classes: 2
        # of data: 8124
        # of features: 112
        Files: mushrooms.txt
        """
        super(Mushrooms, self).__init__(2, 8124, 112, "mushrooms.txt")
        # set label from [1, 2] to [0, 1]
        self.label = (self.label - 1.0)
        self.label = self.label.unsqueeze(1)

class Phishing(myDataset):
    def __init__(self):
        """
        phishing
        Source: UCI / Phishing Websites
        Preprocessing: All features are categorical. We use binary encoding to generate feature vectors. Each feature vector is normalized to maintain unit-length. [YJ16a]
        # of classes: 2
        # of data: 11,055
        # of features: 68
        Files: phishing.txt
        """
        super(Phishing, self).__init__(2, 11055, 68, "phishing.txt")
        self.label = self.label.unsqueeze(1)


class W8A(myDataset):
    def __init__(self):
        """
        w8a
        Source: [JP98a]
        # of classes: 2
        # of data: 49,749 / 14,951 (testing)
        # of features: 300 / 300 (testing)
        Files: w8a.txt
        """
        super(W8A, self).__init__(2, 49749, 300, "w8a.txt")
        # set label from [-1, +1] to [0, 1]
        self.label = (self.label + 1.0) / 2.0
        self.label = self.label.unsqueeze(1)

class IJCNN1(myDataset):
    def __init__(self):
        """
        Source: [DP01a]
        Preprocessing: We use winner's transformation [Chang01d]
        # of classes: 2
        # of data: 49,990 / 91,701 (testing)
        # of features: 22
        Files: ijcnn1.txt
        """
        super(IJCNN1, self).__init__(2, 49990, 22, "ijcnn1.txt")
        # set label from [-1, +1] to [0, 1]
        self.label = (self.label + 1.0) / 2.0
        self.label = self.label.unsqueeze(1)
