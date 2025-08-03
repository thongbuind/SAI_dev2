from tokenizers import Tokenizer, trainers, models, pre_tokenizers
from tokenizers.normalizers import NFD, Lowercase, StripAccents, Sequence
from tokenizers.pre_tokenizers import Whitespace
import json
import numpy as np
from pathlib import Path

current_file = Path(__file__).resolve()
data_dir = current_file.parent
config_file = data_dir.parent / "config" / "config.json"
raw_dir = data_dir / "raw"
processed_dir = data_dir / "processed"
processed_dir.mkdir(parents=True, exist_ok=True)

# Đọc config để lấy max_seq_len
with open(config_file, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']
vocab_size = config['vocab_size']

# Bước 1: Tải dữ liệu
dataset = []
with open(raw_dir / "pre_train.json", "r", encoding="utf-8") as f:
    json_data = json.load(f)
    dataset = [item.strip() for item in json_data if isinstance(item, str) and item.strip()]

# Bước 2: Tạo tokenizer BPE
tokenizer = Tokenizer(models.BPE())
tokenizer.normalizer = Sequence([NFD(), Lowercase(), StripAccents()])
tokenizer.pre_tokenizer = Whitespace()
trainer = trainers.BpeTrainer(
    vocab_size=vocab_size, min_frequency=2,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"]
)
tokenizer.train_from_iterator(dataset, trainer=trainer)

# Bước 3: Lưu tokenizer và vocab
tokenizer.save(str(processed_dir / "bpe_tokenizer.json"))

vocab = tokenizer.get_vocab()
sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
with open(data_dir / "new_vocab.txt", 'w', encoding='utf-8') as f:
    for token, idx in sorted_vocab:
        f.write(f"{token}\t{idx}\n")

# Bước 4: Tokenize và tạo X, Y, lengths (cùng format với VnCoreNLP)
X, Y, lengths = [], [], []

for line in dataset:
    encoded = tokenizer.encode(line.lower())
    tokens = encoded.ids
    
    # Bỏ qua câu quá ngắn hoặc quá dài (giống logic VnCoreNLP)
    if len(tokens) < 2 or len(tokens) > max_seq_len - 2:  # -2 để dành chỗ cho BOS/EOS nếu cần
        continue
    
    # KHÔNG padding ở đây - để train.py xử lý dynamic padding
    X.append(tokens)
    Y.append(tokens)  # Y cũng là tokens (tương tự như VnCoreNLP)
    lengths.append(len(tokens))

# Bước 5: Lưu cùng format với VnCoreNLP
np.savez_compressed(
    processed_dir / "new_data_tokenized.npz",
    X=np.array(X, dtype=object),
    Y=np.array(Y, dtype=object),
    lengths=np.array(lengths)
)

print(f"✅ Đã lưu dữ liệu vào: {processed_dir}/new_data_tokenized.npz")
print(f"📊 Tổng số mẫu: {len(X)}")
print(f"📈 Độ dài sequence trung bình: {np.mean(lengths):.2f}")
print(f"📉 Độ dài sequence min/max: {min(lengths)}/{max(lengths)}")