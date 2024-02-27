import torch
import pandas as pd
import argparse
from transformers import BertTokenizer
from torch.utils.data import TensorDataset, random_split
class COLA_TensorDataset(torch.utils.data.Dataset):

    def __init__(self, input_sentences, attention_masks, labels):
        self.input_sentences = input_sentences
        self.attention_masks = attention_masks
        self.labels = labels

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (doc, target) where target is index of the target class.
        """
        return (self.input_sentences[index], self.attention_masks[index]), self.labels[index]

    def __len__(self):
        return len(self.input_sentences)

def create_cola_dataset(datasets_dir, num_train=7000, transform = None, target_transform = None, train = True):
    df = pd.read_csv(datasets_dir, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    sentences = df.sentence.values;labels = df.label.values # df = pd.read_csv(datasets_dir, delimiter='\t', header=None, names=['sentence_source', 'label', 'label_notes', 'sentence'])
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    max_len = 0

    # For every sentence...
    for sent in sentences:

        # Tokenize the text and add `[CLS]` and `[SEP]` tokens.
        input_ids = tokenizer.encode(sent, add_special_tokens=True)

        # Update the maximum sentence length.
        max_len = max(max_len, len(input_ids))


    # Tokenize all of the sentences and map the tokens to thier word IDs.
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in sentences:
        # `encode_plus` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        #   (5) Pad or truncate the sentence to `max_length`
        #   (6) Create attention masks for [PAD] tokens.
        encoded_dict = tokenizer.encode_plus(
                            sent,                      # Sentence to encode.
                            add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                            max_length = 100,           # Pad & truncate all sentences.
                            pad_to_max_length = True,
                            return_attention_mask = True,   # Construct attn. masks.
                            return_tensors = 'pt',     # Return pytorch tensors.
                       )
        
        # Add the encoded sentence to the list.    
        input_ids.append(encoded_dict['input_ids'])
        
        # And its attention mask (simply differentiates padding from non-padding).
        attention_masks.append(encoded_dict['attention_mask'])

    # Convert the lists into tensors.
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels = torch.tensor(labels)
    dataset = COLA_TensorDataset(input_ids, attention_masks, labels)
    train_size = 7000
    val_size = len(dataset) - train_size

    # Divide the dataset by randomly selecting samples.
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    

    return train_dataset, val_dataset



