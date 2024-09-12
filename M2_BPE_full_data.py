import sys
# Correct the path to the cloned 'mamba' repository on your cluster
# sys.path.append('/hhwang/kagaku/mamba')

import setuptools

print(setuptools.__version__)

# Data Loading and Preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

# Training
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import time
from transformers import AutoTokenizer

from tqdm import tqdm
from mamba.mamba_ssm.modules.mamba2 import Mamba2
from mamba.mamba_ssm.models.config_mamba import MambaConfig
from mamba.mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
import pandas as pd


import os
import json
import logging
import struct

### Data Loading and Preprocessing ###
class SMILESDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length=512):
        super(SMILESDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.smiles = []

        # Load the data file and process each line
        with open(file_path, 'r') as file:
            for line in file:
                line = line.strip()
                if line:  # Ensure the line is not empty
                    self.smiles.append(line)

    def __len__(self):
        return len(self.smiles)

    def __getitem__(self, idx):
        smiles_string = self.smiles[idx]
        # Encode the SMILES string and truncate
        tokenized = self.tokenizer.encode(smiles_string, add_special_tokens=True, max_length=self.max_length, truncation=True)
        # Truncate the sequence to max_length
        tokenized = tokenized[:self.max_length] #check if this is correct
        tensor = torch.tensor(tokenized, dtype=torch.long)
        return tensor


def collate_batch(batch):
    batch_padded = pad_sequence(batch, batch_first=True, padding_value=0)
    #inputs = batch_padded[:, :-1].unsqueeze(-1)  # Adding dummy dimension for compatibility
    inputs = batch_padded[:, :-1]
    targets = batch_padded[:, 1:]  # targets do not need the feature dimension

    # If using dummy features, adjust the size correctly
    #input_features = inputs.expand(-1, -1, 512)  # Expand to the correct feature size
    inputs = inputs.long()

    return inputs, targets

### Evaluation Function ###
def evaluate(model, data_loader, criterion, device):
    model.eval()
    eval_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs = inputs.to(device)
            targets = targets.to(device).view(-1).long()

            # Forward pass
            outputs = model(inputs)

            # Assuming outputs is an instance of CausalLMOutput, access logits
            logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]

            # Now you can use view safely
            logits = logits.view(-1, logits.shape[-1])
            targets = targets.view(-1)

            loss = criterion(logits, targets)
            eval_loss += loss.item()
            num_batches += 1

    avg_loss = eval_loss / num_batches
    perplexity = torch.exp(torch.tensor(avg_loss))
    return avg_loss, perplexity

### Training Function ###
def train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0.0
        num_batches = 0

        with tqdm(total=len(train_loader), desc=f"Epoch {epoch + 1}/{num_epochs}", unit="batch") as pbar:
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(device).long()
                targets = targets.to(device).view(-1).long()

                optimizer.zero_grad()

                # Calling model and accessing logits
                outputs = model(inputs.long())
                logits = outputs.logits  # Access the logits from the CausalLMOutput named tuple or data class

                # Ensure outputs are reshaped correctly for the loss calculation
                logits = logits.view(-1, logits.size(-1))
                targets = targets.view(-1)

                loss = criterion(logits, targets)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1

                pbar.update(1)  # Update progress bar

        avg_train_loss = epoch_loss / num_batches
        train_perplexity = torch.exp(torch.tensor(avg_train_loss))

        # Evaluate on validation set
        avg_val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

        print(f"Epoch {epoch + 1}/{num_epochs}")
        print(f"Train Loss: {avg_train_loss:.4f}, Train Perplexity: {train_perplexity:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Perplexity: {val_perplexity:.4f}")
        

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


config = MambaConfig(
    d_model=1024,
    n_layer=12,
    d_intermediate=4096, #has to be d_model*4
    vocab_size=7924,
    ssm_cfg={'layer': 'Mamba2'},
    attn_layer_idx=[],
    attn_cfg={},
    rms_norm=True,
    residual_in_fp32=True,
    fused_add_norm=True
)

model = MambaLMHeadModel(config)
model = model.to(device)


optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = CrossEntropyLoss()

data_file = "./hhwang/shiawaseda/Datasets/train.txt"
test_data_file = "./hhwang/shiawaseda/Datasets/test.txt"
tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

total_dataset = SMILESDataset(data_file, tokenizer, max_length=512)
test_dataset = SMILESDataset(test_data_file, tokenizer, max_length=512)
train_size = int(0.8 * len(total_dataset))
val_size = len(total_dataset) - train_size
train_dataset, val_dataset = random_split(total_dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, collate_fn=collate_batch)


num_epochs = 2
train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)


# Testing using the evaluate function
test_loss, test_perplexity = evaluate(model, test_loader, criterion, device)
print(f'Test Loss: {test_loss:.4f}, Test Perplexity: {test_perplexity:.4f}')

def save_pretrained(model, save_directory):
    # Ensure save_directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Save the model's state_dict
    model_path = os.path.join(save_directory, 'pytorch_model.bin')
    torch.save(model.state_dict(), model_path)

    # Save the configuration of the model
    config_path = os.path.join(save_directory, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(model.config.__dict__, f, indent=4)


def dtype_to_abbreviation(dtype):
    # This function maps PyTorch dtypes to their abbreviations
    dtype_str = str(dtype)
    return {
        'torch.float32': 'F32',
        'torch.float64': 'F64',
        'torch.float16': 'F16',
        'torch.int32': 'I32',
        'torch.int64': 'I64',
        'torch.int16': 'I16',
        'torch.int8': 'I8',
        'torch.uint8': 'U8'
    }.get(dtype_str, dtype_str)  # Fallback to the full string if no abbreviation


def save_model_as_safetensors(model, save_directory, filename='model.safetensors'):
    model.to('cpu')  # Move the model to CPU to handle any potential device-specific tensors
    state_dict = model.state_dict()
    metadata = {'__metadata__': {'format': 'pt'}}
    tensor_data = bytearray()

    current_offset = 0
    for name, tensor in state_dict.items():
        tensor_bytes = tensor.numpy().tobytes()
        dtype_abbreviation = dtype_to_abbreviation(tensor.dtype)  # Convert dtype to abbreviation
        metadata[name] = {
            'dtype': dtype_abbreviation,  # Use abbreviated dtype
            'shape': list(tensor.shape),
            'data_offsets': [current_offset, current_offset + len(tensor_bytes)]
        }
        current_offset += len(tensor_bytes)
        tensor_data.extend(tensor_bytes)

    metadata_json = json.dumps(metadata)
    metadata_bytes = metadata_json.encode('utf-8')
    metadata_length = len(metadata_bytes)

    # Open the file in binary mode and write
    with open(os.path.join(save_directory, filename), 'wb') as f:
        f.write(struct.pack('<Q', metadata_length))  # Write the length of metadata as an unsigned long long
        f.write(metadata_bytes)  # Write the metadata
        f.write(tensor_data)  # Write the actual tensor data

    return os.path.join(save_directory, filename)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define the save directory
save_directory = './model_saves'

try:
    # Ensure the directory exists
    os.makedirs(save_directory, exist_ok=True)

    # Save using the original save_pretrained method
    save_pretrained(model, save_directory)
    logging.info(f"Standard model files saved in {save_directory}.")

    # Save using custom safetensors method
    safetensors_path = save_model_as_safetensors(model, save_directory)
    logging.info(f"Model saved successfully in custom safetensors format at {safetensors_path}.")
except Exception as e:
    logging.error(f"An error occurred while saving the model: {str(e)}")
    
    
print(model)