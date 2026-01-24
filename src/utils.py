import numpy as np
import gc
# import torch
# import psutil
# import os
# from pympler import asizeof

def load_data(data_type, path_1, path_2=None):
    if data_type == "pretrain":
        data = np.load(path_1, allow_pickle=True)
        X, Y, lengths = data["X"], data["Y"], data["lengths"]
        data.close()
        return X, Y, lengths

    elif data_type == "continued_pretrain":
        cont_data = np.load(path_1, allow_pickle=True)
        X_c, Y_c, L_c = cont_data["X"], cont_data["Y"], cont_data["lengths"]
        cont_data.close()

        if path_2 is not None:
            pre_data = np.load(path_2, allow_pickle=True)
            X_p, Y_p, L_p = pre_data["X"], pre_data["Y"], pre_data["lengths"]
            pre_data.close()

            n_continued = len(X_c)
            n_pretrain_needed = 5 * n_continued
            total_pretrain = len(X_p)
            
            rng = np.random.default_rng(54)
            shuffled_indices = rng.permutation(total_pretrain)
            
            n_samples = min(n_pretrain_needed, total_pretrain)
            selected_indices = shuffled_indices[:n_samples]
            
            X_p_sampled = X_p[selected_indices]
            Y_p_sampled = Y_p[selected_indices]
            L_p_sampled = L_p[selected_indices]
            
            del X_p, Y_p, L_p, shuffled_indices, selected_indices
            gc.collect()
            
            X_combined = np.concatenate([X_c, X_p_sampled])
            Y_combined = np.concatenate([Y_c, Y_p_sampled])
            L_combined = np.concatenate([L_c, L_p_sampled])
            
            combined_indices = rng.permutation(len(X_combined))
            X = X_combined[combined_indices]
            Y = Y_combined[combined_indices]
            lengths = L_combined[combined_indices]
            
        else:
            X, Y, lengths = X_c, Y_c, L_c

        return X, Y, lengths

    elif data_type == "finetune":
        data = np.load(path_1, allow_pickle=True)
        input = data["input"]
        response = data["response"]
        input_lengths = data["input_lengths"]
        response_lengths = data["response_lengths"]
        data.close()
        return input, response, input_lengths, response_lengths

    else:
        raise ValueError(f"Unknown data_type: {data_type}")

def log_progress(text):
    fixed_width = 82
    formatted_text = f"║ {text:<{fixed_width}} ║"
    print(formatted_text)

# def log_memory_usage(note="", top_k=20):
#     process = psutil.Process(os.getpid())
#     mem_info = process.memory_info()
#     rss_in_mb = mem_info.rss / (1024 ** 2)
#     log_progress(f"[MEMORY] {note} | RSS RAM used: {rss_in_mb:.2f} MB")

#     all_objects = gc.get_objects()
#     var_sizes = []

#     for obj in all_objects:
#         try:
#             size = asizeof.asizeof(obj)
#             var_sizes.append((type(obj).__name__, size, repr(obj)[:80]))
#         except Exception:
#             continue

#     var_sizes.sort(key=lambda x: x[1], reverse=True)

#     log_progress(f"Top {top_k} Python objects by size:")
#     for typename, size, preview in var_sizes[:top_k]:
#         log_progress(f"  {typename:<25} {size/1024/1024:.2f} MB | {preview}")

#     try:
#         cpu_tensors = [obj for obj in gc.get_objects() if torch.is_tensor(obj) and obj.device.type == 'cpu']
#         cpu_mem_mb = sum(obj.element_size() * obj.nelement() for obj in cpu_tensors) / (1024 ** 2)
#         log_progress(f"[PyTorch][CPU] Tensors memory: {cpu_mem_mb:.2f} MB | Count: {len(cpu_tensors)}")
#     except Exception:
#         pass

#     if torch.cuda.is_available():
#         try:
#             for i in range(torch.cuda.device_count()):
#                 allocated = torch.cuda.memory_allocated(i) / (1024 ** 2)
#                 reserved = torch.cuda.memory_reserved(i) / (1024 ** 2)
#                 max_allocated = torch.cuda.max_memory_allocated(i) / (1024 ** 2)
#                 log_progress(f"[PyTorch][GPU:{i}] Allocated: {allocated:.2f} MB | Reserved: {reserved:.2f} MB | Peak: {max_allocated:.2f} MB")
#         except Exception:
#             pass
