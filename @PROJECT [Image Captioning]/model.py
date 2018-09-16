import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features) # feature size (batch size, embedded size)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        #super(DecoderRNN, self).__init__()
        super().__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.embed = nn.Embedding(vocab_size, embed_size)
    
    def forward(self, features, captions):
        # caption size (batch size, caption length) -> (batch size, caption length, embedded size)
        embeddings = self.embed(captions)
        
        # Change the feature size (batch size, embedded size) -> (batch size, 1, embedded size)
        # Add with the caption input (batch size, 1 + caption length, embedded size) 
        embeddings = torch.cat((features.unsqueeze(1), embeddings), 1)
        
        # output is short term memory, hidden is (short term memory, long term memory)
        # embedding[:,:-1,:] at this code, put :-1 because to take off <end>
        # output size should be (batch size, caption length, vocab size)
        output, hidden = self.lstm(embeddings[:,:-1,:])
        output = self.linear(output)
        
        return output


    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        # inputs size (batch size = 1, 1, embedded size)
        sampled_ids = []
        for i in range(max_len):
            # output size (1, 1, hidden size). states (hidden, cell) will be the same size (1, 1, hidden size)
            output, states = self.lstm(inputs, states)
            # output size (1, 1, vocab size)
            output = self.linear(output)
            output = output.squeeze()
            predict = torch.argmax(output, dim=0)
            sampled_ids.append(predict.item())         
            if predict.item() == 1:
                break
            # inputs size (1, embedded size)
            inputs = self.embed(predict.unsqueeze(0))
            # inputs size (1, 1, embedded size)
            inputs = inputs.unsqueeze(1)
         
        return sampled_ids
            
                
        
        
        
        
        
        
        
        
        
        
        
        
        
        