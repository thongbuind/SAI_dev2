from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import json
import gc
import sys
from utils import log_progress, load_data
from model import TransformerModel

current_file = Path(__file__).resolve()
project_root = current_file.parent.parent
sys.path.append(str(project_root))
config_dir = project_root / "config"
data_dir = project_root / "data"
model_dir = project_root / "model"
src_dir = project_root / "src"
config_file = config_dir / "config.json"
model_dir.mkdir(parents=True, exist_ok=True)
data_processed_dir = project_root / "data" / "processed"
pretrain_tokenized_file = data_processed_dir / "pretrain_data_shorted_ids.npz"
continued_pretrain_tokenized_file = data_processed_dir / "continued_pretrain_ids.npz"

class Dataset(torch.utils.data.Dataset):
    def __init__(self, X, Y, lengths):
        self.X = X
        self.Y = Y
        self.lengths = lengths

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx], self.lengths[idx]

def split_train_val_test(X, Y, lengths, train_ratio, val_ratio, seed=54):
    total_sample = len(X)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(total_sample)

    train_end = int(total_sample * train_ratio)
    val_end = int(total_sample * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, Y_train, lengths_train = X[train_idx], Y[train_idx], lengths[train_idx]
    X_val, Y_val, lengths_val = X[val_idx], Y[val_idx], lengths[val_idx]
    X_test, Y_test, lengths_test = X[test_idx], Y[test_idx], lengths[test_idx]
    
    return (X_train, Y_train, lengths_train, 
            X_val, Y_val, lengths_val, 
            X_test, Y_test, lengths_test)

def create_dataset(X, Y, lengths, batch_size, shuffle):

    log_progress(f"Đang tạo dataset từ {len(X)} samples...")
    X_tensor = [torch.tensor(x, dtype=torch.long) for x in X]
    Y_tensor = [torch.tensor(y, dtype=torch.long) for y in Y]
    lengths_tensor = torch.tensor(lengths, dtype=torch.long)

    if shuffle:
        total_sample = len(X)
        indices = torch.randperm(total_sample)
        X_tensor = [X_tensor[i] for i in indices]
        Y_tensor = [Y_tensor[i] for i in indices]
        lengths_tensor = lengths_tensor[indices]

    log_progress("Đã convert sang tensors, đang tạo dataset...")

    dataset = Dataset(X_tensor, Y_tensor, lengths_tensor)

    def collate_fn(batch):
        X_batch = [item[0] for item in batch]
        Y_batch = [item[1] for item in batch]
        lengths_batch = [item[2] for item in batch]
        
        max_len = max(len(x) for x in X_batch)
        bucket_len = ((max_len + 19) // 20) * 20
        
        X_padded = torch.stack([torch.nn.functional.pad(x, (0, bucket_len - len(x))) for x in X_batch])
        Y_padded = torch.stack([torch.nn.functional.pad(y, (0, bucket_len - len(y))) for y in Y_batch])
        lengths_tensor = torch.stack(lengths_batch)
        
        return X_padded, Y_padded, lengths_tensor
    
    dataloader = torch.utils.data.DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=collate_fn
    )

    return dataloader

def pretrain(model, optimizer, scheduler, device, pretrain_tokenized_file, num_epochs, model_folder, train_ratio, val_ratio, batch_size):
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                            BẮT ĐẦU LOAD PRETRAIN DATA                              ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    X, Y, lengths = load_data("pretrain", pretrain_tokenized_file)
    X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    train_ds = create_dataset(X_train, Y_train, lengths_train, batch_size, shuffle=True)    
    val_ds = create_dataset(X_val, Y_val, lengths_val, batch_size, shuffle=False)    
    test_ds = create_dataset(X_test, Y_test, lengths_test, batch_size, shuffle=False)

    del X, Y, lengths
    del X_train, Y_train, lengths_train
    del X_val, Y_val, lengths_val  
    del X_test, Y_test, lengths_test
    gc.collect()

    criterion = nn.CrossEntropyLoss(ignore_index=0)
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch, lengths_batch in train_ds:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch, lengths_batch in val_ds:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        
        log_progress(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_folder / "s_a_i.pt")
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               ĐÁNH GIÁ TRÊN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch, lengths_batch in test_ds:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            test_loss += loss.item()
    
    test_loss /= len(test_ds)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

def continued_pretrain(model, optimizer, scheduler, device, continued_pretrain_tokenized_file, pretrain_tokenized_file, num_epochs, model_folder, train_ratio, val_ratio, batch_size):
    print("╔════════════════════════════════════════════════════════════════════════════════════╗")
    print("║                       BẮT ĐẦU LOAD CONTINUED PRETRAIN DATA                         ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    X, Y, lengths = load_data("continued_pretrain", continued_pretrain_tokenized_file, pretrain_tokenized_file)
    X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)
    log_progress(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    train_ds = create_dataset(X_train, Y_train, lengths_train, batch_size, shuffle=True)    
    val_ds = create_dataset(X_val, Y_val, lengths_val, batch_size, shuffle=False)    
    test_ds = create_dataset(X_test, Y_test, lengths_test, batch_size, shuffle=False)

    del X, Y, lengths
    del X_train, Y_train, lengths_train
    del X_val, Y_val, lengths_val  
    del X_test, Y_test, lengths_test
    gc.collect()

    criterion = nn.CrossEntropyLoss()
    best_val_loss = float('inf')

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        for X_batch, Y_batch, lengths_batch in train_ds:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
        
        train_loss /= len(train_ds)
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X_batch, Y_batch, lengths_batch in val_ds:
                X_batch = X_batch.to(device)
                Y_batch = Y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
                val_loss += loss.item()
        
        val_loss /= len(val_ds)
        scheduler.step(val_loss)
        
        log_progress(f"Epoch {epoch+1}/{num_epochs} Train Loss: {train_loss:.4f} Val Loss: {val_loss:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_folder / "s_a_i.pt")
            print(f"Saved best model with val_loss: {val_loss:.4f}")

    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    print("║                               ĐÁNH GIÁ TRÊN TEST SET                               ║")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")
    
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for X_batch, Y_batch, lengths_batch in test_ds:
            X_batch = X_batch.to(device)
            Y_batch = Y_batch.to(device)
            
            outputs = model(X_batch)
            loss = criterion(outputs.view(-1, outputs.size(-1)), Y_batch.view(-1))
            test_loss += loss.item()
    
    test_loss /= len(test_ds)
    log_progress(f"Test Loss: {test_loss:.4f}")
    print("╠════════════════════════════════════════════════════════════════════════════════════╣")

    return test_loss

### MAIN
with open(config_file, 'r') as f:
    config = json.load(f)
vocab_size = config['vocab_size']
max_seq_len = config['max_seq_len']
d_model = config['d_model']
num_heads = config['num_heads']
num_layers = config['num_layers']
ff_dim = config['ff_dim']
dropout = config['dropout']
epochs = config['epochs']
batch_size = config['batch_size']
train_ratio = config['train_ratio']
val_ratio = config['val_ratio']
learning_rate = config['learning_rate']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TransformerModel(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout).to(device)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6, verbose=True)

print("╠════════════════════════════════════════════════════════════════════════════════════╣")
print("║                                 BẮT ĐẦU TRAINING                                   ║")
print("╠════════════════════════════════════════════════════════════════════════════════════╣")

pretrain_test_loss = pretrain(
    model,
    optimizer,
    scheduler,
    device,
    pretrain_tokenized_file,
    num_epochs=epochs, 
    model_folder=model_dir,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    batch_size=batch_size
)

continued_pretrain_test_loss = continued_pretrain(
    model,
    optimizer,
    scheduler,
    device,
    continued_pretrain_tokenized_file, 
    pretrain_tokenized_file,
    num_epochs=epochs, 
    model_folder=model_dir,
    train_ratio=train_ratio,
    val_ratio=val_ratio,
    batch_size=batch_size
)

log_progress(f"Hoàn thành training!")
log_progress(f"Pretrain Test Loss: {pretrain_test_loss:.4f}")
log_progress(f"Continued Pretrain Test Loss: {continued_pretrain_test_loss:.4f}")
log_progress(f"Đã lưu model cuối cùng vào: {model_dir / 's_a_i.pt'}")
