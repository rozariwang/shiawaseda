# Data Loading and Preprocessing
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split, Subset
from torch.nn.utils.rnn import pad_sequence

# Training
import torch.optim as optim
from torch.nn import CrossEntropyLoss
import time
from transformers import AutoTokenizer

# Mamba and Tokenizer 
from tqdm import tqdm
from mamba_ssm.modules.mamba2 import Mamba2
from mamba_ssm.models.config_mamba import MambaConfig
from mamba_ssm.models.mixer_seq_simple import MambaLMHeadModel
from transformers import AutoTokenizer
import pandas as pd
import random


# Data Loading and Preprocessing
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


# Evaluation Function 
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

# Training Function 
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

train_data_file = "./hhwang/shiawaseda/Datasets/train.txt"
val_data_file = "./hhwang/shiawaseda/Datasets/val.txt"

tokenizer = AutoTokenizer.from_pretrained("seyonec/PubChem10M_SMILES_BPE_450k")

# Load the datasets
total_train_dataset = SMILESDataset(train_data_file, tokenizer, max_length=512)
total_val_dataset = SMILESDataset(val_data_file, tokenizer, max_length=512)

print(f"Total training samples: {len(total_train_dataset)}")
print(f"Total validation samples: {len(total_val_dataset)}")

# Select 5% of the training and validation data randomly
train_subset_indices = torch.randperm(len(total_train_dataset))[:int(0.05 * len(total_train_dataset))]
print(f"Train subset size: {len(train_subset_indices)}")
train_subset_dataset = Subset(total_train_dataset, train_subset_indices)
val_subset_indices = torch.randperm(len(total_val_dataset))[:int(0.05 * len(total_val_dataset))]
val_subset_dataset = Subset(total_val_dataset, val_subset_indices)

print(f"Length of train_subset_dataset: {len(train_subset_dataset)}")
print(f"Length of val_subset_dataset: {len(val_subset_dataset)}")


# Create DataLoader 
train_loader = DataLoader(train_subset_dataset, batch_size=32, shuffle=True, collate_fn=collate_batch)
val_loader = DataLoader(val_subset_dataset, batch_size=32, shuffle=False, collate_fn=collate_batch)

# Define the range for each parameter
d_model_options = [256, 512, 1024]
n_layer_options = [4, 8, 12]
lr_options = [0.01, 0.001, 0.005, 0.0001, 0.0005, 0.00005]
num_epochs = 5
num_search_iters = 27  # Define how many iterations of random search you want

results = []

# Set to store tested hyperparameter combinations
tested_combinations = set()

# Random search
for _ in range(num_search_iters):
    # Generate a unique combination of hyperparameters
    while True:
        d_model = random.choice(d_model_options)
        n_layer = random.choice(n_layer_options)
        lr = random.choice(lr_options)
        combination = (d_model, n_layer, lr)

        # Check if this combination has already been tested
        if combination not in tested_combinations:
            tested_combinations.add(combination)
            break

    config = MambaConfig(
        d_model=d_model,
        n_layer=n_layer,
        d_intermediate=d_model * 4,
        vocab_size=7924,  # change according to tokenizer
        ssm_cfg={'layer': 'Mamba2'},
        attn_layer_idx=[],
        attn_cfg={},
        rms_norm=True,
        residual_in_fp32=True,
        fused_add_norm=True
    )
    model = MambaLMHeadModel(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    # Use the train function
    train(model, train_loader, val_loader, optimizer, criterion, device, num_epochs)

    # Evaluate the model after training
    avg_val_loss, val_perplexity = evaluate(model, val_loader, criterion, device)

    # Store results
    results.append({
        'd_model': d_model,
        'n_layer': n_layer,
        'lr': lr,
        'epochs': num_epochs,
        'val_loss': avg_val_loss,
        'val_perplexity': val_perplexity.item()
    })

# Save results to DataFrame and then to CSV file
results_df = pd.DataFrame(results)
results_df.to_csv('./random_search_results.csv', index=False)
print("Random search complete. Results saved to 'random_search_results.csv'.")

# Output the DataFrame to review here as well
print(results_df)