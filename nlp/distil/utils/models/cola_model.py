from transformers import BertModel, AdamW, BertConfig
import torch
import torch.nn as nn
class BERT_COLA(nn.Module):
    def __init__(self, name="bert-base-uncased", num_labels = 2):
        super(BERT_COLA, self).__init__()
        self.encoder =  BertModel.from_pretrained('bert-base-uncased')
        self.embDim = 768
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, x,last=False, freeze=False):
        input, attention_mask = x;# print(input.shape, attention_mask.shape)
        if freeze:
            with torch.no_grad():
                features = (self.encoder(input_ids = input, attention_mask=attention_mask)x[0])[:, 0, :]
        else:
            features = (self.encoder(input_ids = input, attention_mask=attention_mask)[0])[:, 0, :]

        logits = self.classifier(features)
        if last:
            return logits, features
        else:   
            return logits

    def get_embedding_dim(self):
        return self.embDim

    def get_grads(self) -> torch.Tensor:
        """
        Returns all the gradients concatenated in a single tensor.
        :return: gradients tensor (??)
        """
        grads = []
        for pp in list(self.parameters()):
            if pp.requires_grad: # only using the parameter that require the gradient
                grads.append(pp.grad.view(-1))
        return torch.cat(grads)




