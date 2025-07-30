import json
import numpy as np
import tensorflow as tf
from model import Model
import sys
from pathlib import Path
import psutil
import objgraph
from pympler import asizeof
import os
import gc

project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))
from data.processed.data_tokenized import X, Y, lengths

current_file = Path(__file__).resolve()

config_dir = current_file.parent.parent / "config" / "config.json"
with open(config_dir, 'r') as f:
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

def log_memory(epoch):
    print(f"\n=== Memory Usage at End of Epoch {epoch+1} ===")
    
    # System RAM
    mem = psutil.virtual_memory()
    process = psutil.Process(os.getpid())
    print(f"System RAM:")
    print(f"  Tổng RAM: {mem.total / (1024 ** 3):.2f} GB")
    print(f"  RAM đã sử dụng: {mem.used / (1024 ** 3):.2f} GB")
    print(f"  Phần trăm RAM đã sử dụng: {mem.percent}%")
    print(f"  RAM còn trống: {mem.available / (1024 ** 3):.2f} GB")
    print(f"  RAM của tiến trình Python: {process.memory_info().rss / (1024 ** 3):.2f} GB")

    # Kích thước các đối tượng lớn
    print(f"\nKích thước các đối tượng lớn trong RAM:")
    print(f"  X_train: {asizeof.asizeof(X_train) / (1024 ** 2):.2f} MB")
    print(f"  Y_train: {asizeof.asizeof(Y_train) / (1024 ** 2):.2f} MB")
    print(f"  lengths_train: {asizeof.asizeof(lengths_train) / (1024 ** 2):.2f} MB")
    print(f"  X_val: {asizeof.asizeof(X_val) / (1024 ** 2):.2f} MB")
    print(f"  Y_val: {asizeof.asizeof(Y_val) / (1024 ** 2):.2f} MB")
    print(f"  lengths_val: {asizeof.asizeof(lengths_val) / (1024 ** 2):.2f} MB")
    print(f"  model: {asizeof.asizeof(model) / (1024 ** 2):.2f} MB")

    # GPU RAM (VRAM)
    try:
        gpu_mem = tf.config.experimental.get_memory_info('GPU:0')
        print(f"\nGPU VRAM:")
        print(f"  VRAM hiện tại: {gpu_mem['current'] / (1024 ** 3):.2f} GB")
        print(f"  VRAM đỉnh: {gpu_mem['peak'] / (1024 ** 3):.2f} GB")
    except RuntimeError:
        print(f"\nGPU VRAM: Không tìm thấy GPU hoặc không hỗ trợ kiểm tra bộ nhớ.")

    # Thu thập rác để giảm bộ nhớ
    gc.collect()

def split_train_val_test(X, Y, lengths, train_ratio, val_ratio):
    """Chia dữ liệu thành train/validation/test set"""
    total_samples = len(X)
    indices = np.random.permutation(total_samples)
    
    train_size = int(total_samples * train_ratio)
    val_size = int(total_samples * val_ratio)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    X_train = [X[i] for i in train_indices]
    Y_train = [Y[i] for i in train_indices]
    lengths_train = [lengths[i] for i in train_indices]
    
    X_val = [X[i] for i in val_indices]
    Y_val = [Y[i] for i in val_indices]
    lengths_val = [lengths[i] for i in val_indices]
    
    X_test = [X[i] for i in test_indices]
    Y_test = [Y[i] for i in test_indices]
    lengths_test = [lengths[i] for i in test_indices]
    
    return X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test

def create_tf_dataset(X, Y, lengths, batch_size, shuffle=True, prefetch_size=tf.data.AUTOTUNE):
    """
    Tạo tf.data.Dataset với dynamic padding và optimizations
    """
    def generator():
        indices = list(range(len(X)))
        if shuffle:
            np.random.shuffle(indices)
        for i in indices:
            yield X[i], Y[i], lengths[i]
    
    # Tạo dataset từ generator
    dataset = tf.data.Dataset.from_generator(
        generator,
        output_signature=(
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(None,), dtype=tf.int32),
            tf.TensorSpec(shape=(), dtype=tf.int32)
        )
    )
    
    # Group sequences by similar lengths để tối ưu padding
    def length_bucket_key(x, y, length):
        # Chia sequences thành các bucket theo độ dài
        bucket_width = 50
        return length // bucket_width
    
    def reduce_func(key, windowed_data):
        return windowed_data.batch(batch_size)
    
    # Bucket by sequence length và batch
    dataset = dataset.group_by_window(
        key_func=length_bucket_key,
        reduce_func=reduce_func,
        window_size=batch_size
    )
    
    # Padding function cho batch
    def pad_batch(batch_x, batch_y, batch_lengths):
        # Pad sequences trong batch đến max length của batch đó
        padded_x = tf.keras.preprocessing.sequence.pad_sequences(
            batch_x.numpy(), padding='post', dtype='int32'
        )
        padded_y = tf.keras.preprocessing.sequence.pad_sequences(
            batch_y.numpy(), padding='post', dtype='int32'
        )
        return tf.constant(padded_x), tf.constant(padded_y)
    
    # Apply padding với py_function
    dataset = dataset.map(
        lambda x, y, lengths: tf.py_function(
            pad_batch, [x, y, lengths], [tf.int32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    
    # Optimizations
    if prefetch_size:
        dataset = dataset.prefetch(prefetch_size)
    
    return dataset

class CustomLRScheduler(tf.keras.callbacks.Callback):
    """
    Custom Learning Rate Scheduler như callback
    """
    def __init__(self, patience=3, min_lr=0.00001, warmup_epochs=5, max_lr=0.01, T_max=10):
        super().__init__()
        self.patience = patience
        self.min_lr = min_lr
        self.warmup_epochs = warmup_epochs
        self.max_lr = max_lr
        self.T_max = T_max
        self.best_val_loss = float('inf')
        self.wait = 0
        self.current_lr = min_lr
    
    def warmup_lr(self, epoch):
        return self.min_lr + (self.max_lr - self.min_lr) * epoch / self.warmup_epochs
    
    def cosine_annealing(self, epoch):
        return self.min_lr + 0.5 * (self.max_lr - self.min_lr) * (1 + np.cos(np.pi * (epoch - self.warmup_epochs) / self.T_max))
    
    def on_epoch_end(self, epoch, logs=None):
        val_loss = logs.get('val_loss', float('inf'))
        
        if epoch < self.warmup_epochs:
            self.current_lr = self.warmup_lr(epoch)
        else:
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
            
            self.current_lr = self.cosine_annealing(epoch)
            if self.wait >= self.patience:
                self.wait = 0
        
        self.model.optimizer.learning_rate.assign(self.current_lr)
        print(f"║ Epoch {epoch + 1}: Learning rate = {self.current_lr:.6f}, Val Loss = {val_loss:.4f} ║")

# Split data
X_train, Y_train, lengths_train, X_val, Y_val, lengths_val, X_test, Y_test, lengths_test = split_train_val_test(X, Y, lengths, train_ratio, val_ratio)

# Create model
model = Model(vocab_size, d_model, num_heads, num_layers, ff_dim, max_seq_len, dropout)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
model.compile(
    loss="sparse_categorical_crossentropy", 
    optimizer=optimizer,
    metrics=['accuracy']
)

print("Tạo training dataset...")
train_dataset = create_simple_tf_dataset(X_train, Y_train, batch_size, max_seq_len, shuffle=True)

print("Tạo validation dataset...")
val_dataset = create_simple_tf_dataset(X_val, Y_val, batch_size, max_seq_len, shuffle=False)

print("Tạo test dataset...")
test_dataset = create_simple_tf_dataset(X_test, Y_test, batch_size, max_seq_len, shuffle=False)

callbacks = [
    CustomLRScheduler(),
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),
    tf.keras.callbacks.ModelCheckpoint(
        filepath=str(project_root / "model" / "best_model.keras"),
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    tf.keras.callbacks.LambdaCallback(
        on_epoch_end=lambda epoch, logs: log_memory(epoch)
    )
]

print("╔═════════════════════════════════════════╗")
print("║            BẮT ĐẦU PRE-TRAIN            ║")
print("╠═════════════════════════════════════════╣")

model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=epochs,
    callbacks=callbacks,
    verbose=1
)

print("╠═════════════════════════════════════════╣")
print("║          ĐÁNH GIÁ TRÊN TEST SET         ║")
print("╠═════════════════════════════════════════╣")

# Evaluate model
test_results = model.evaluate(test_dataset, verbose=1)
test_loss = test_results[0]
test_accuracy = test_results[1] if len(test_results) > 1 else None

print(f"║ Test Loss: {test_loss:.4f}")
if test_accuracy:
    print(f"║ Test Accuracy: {test_accuracy:.4f}")
print("╚═════════════════════════════════════════╝")

# Lưu model cuối cùng
model_folder = project_root / "model"
model_folder.mkdir(parents=True, exist_ok=True)
model.save(model_folder / "s_a_i.keras")
print(f"Đã lưu model cuối cùng vào: {model_folder / 's_a_i.keras'}")
print(f"Test Loss cuối cùng: {test_loss:.4f}")
