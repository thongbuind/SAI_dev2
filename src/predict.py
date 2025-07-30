import tensorflow as tf
import numpy as np
from keras import models
import json
import sys
from pathlib import Path
from vncorenlp import VnCoreNLP
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

current_file = Path(__file__).resolve()

model_path = project_root / "model" / "s_a_i.keras"
model = models.load_model(model_path)

config_dir = current_file.parent.parent / "config" / "config.json"
with open(config_dir, 'r') as f:
    config = json.load(f)
max_seq_len = config['max_seq_len']

# Đọc vocab
vocab = {}
vocab_path = current_file.parent.parent/ "data" / "vocab.txt"
with open(vocab_path, "r", encoding="utf-8") as f:
    for line in f:
        word, idx = line.strip().split('\t')
        vocab[word] = int(idx)

idx2word = {i: w for w, i in vocab.items()}

VNCORENLP_PATH = "/Users/thongbui.nd/vncorenlp/VnCoreNLP/VnCoreNLP-1.1.1.jar"
annotator = VnCoreNLP(VNCORENLP_PATH, annotators="wseg", max_heap_size='-Xmx2g')

def tokenize(sentence):
    """Chuyển đổi câu thành token số, sử dụng VnCoreNLP để tách từ tiếng Việt"""
    word_segments = annotator.tokenize(sentence.lower())
    words = [word for segment in word_segments for word in segment]
    tokens = [vocab.get(w, vocab["[UNK]"]) for w in words]
    return tokens

def detokenize(tokens, infor=None):
    """Chuyển token số về câu văn bản, thay thế token đặc biệt"""
    special_tokens = {0, 1, 2, 3, 4, 5, 6}  # PAD, UNK, BOS, EOS, SEP
    words = []
    for t in tokens:
        if t in special_tokens or t not in idx2word:
            continue
        word = idx2word[t]
        if infor and word in infor:
            words.append(infor[word])
        else:
            words.append(word)
    return " ".join(words)

def new_generate_response(sentence, max_new_tokens=max_seq_len, top_k=3, temperature=1.0):
    """
    Tạo phản hồi từ câu đầu vào:
        current_sequence = [BOS] + req
        sequence = loop(predict(current_sequence))
    Nâng cấp:
        - Tối ưu hoá Padding
        - Đa dạng cơ chế lấy mẫu (sampling), sử dụng Top-k
        - Thay vì chỉ sử dụng token cuối thì sử dụng cả đoạn từ đầu để dự đoán
    """
    req_tokens = tokenize(sentence)
    current_sequence = [vocab["[BOS]"]] + req_tokens

    padded_input = tf.keras.preprocessing.sequence.pad_sequences(
        [current_sequence], maxlen=max_seq_len, padding='post', dtype='int32'
    )

    for step in range(max_new_tokens):
        preds = model(padded_input, training=False)
        next_token_probs = preds[0, len(current_sequence) - 1, :].numpy()

        # Áp dụng temperature
        next_token_probs = np.exp(np.log(next_token_probs + 1e-10) / temperature)
        next_token_probs /= np.sum(next_token_probs)

        # Top-k sampling
        top_k_indices = np.argsort(next_token_probs)[-top_k:]
        top_k_probs = next_token_probs[top_k_indices] / np.sum(next_token_probs[top_k_indices])
        next_token = np.random.choice(top_k_indices, p=top_k_probs)

        if next_token in [vocab["[EOS]"], vocab["[PAD]"]]:
            break

        current_sequence.append(int(next_token))
        padded_input[0, len(current_sequence) - 1] = next_token

        if len(current_sequence) >= max_seq_len:
            break
    
    return detokenize(current_sequence[1:])

# ================
# Kiểm Tra Mô Hình
# ================

prompts = [
    "bánh mì",
    "bánh mì có nguồn gốc từ",
    "việt nam",
    "việt nam sở hữu",
    "phở",
    "buổi sáng người việt nam thường ăn",
    "đám mây",
    "Đinh Tiên Hoàng lên ngôi",
    "lê thái tổ có miếu hiệu",
    "công thức 1",
    "sáng hôm ấy",
    "sau khi ăn xong, chúng tôi đi",
    "mặc dù",
    "bởi vì trời mưa,"
]

print("\n=== Test pre-train ===")
for req in prompts:
    print(f"Req: {req} \nRes: {new_generate_response(req)}")


