import struct
import os
from safetensors.torch import load_file


class Config:
    vocab_size = 24576
    block_size = 768
    num_hidden_layers = 12
    n_head = 12
    n_kv_head = 4
    n_embd = 768
    use_rope = True
    use_rmsnorm = True
    use_swiglu = True


config = Config()


def export_safetensors_to_bin(safetensors_path, output_bin_path):
    print(f"Loading Safetensors from: {safetensors_path}")
    state_dict = load_file(safetensors_path)

    dim = config.n_embd
    hidden_dim = int(2 * (4 * config.n_embd) / 3)
    n_layers = config.num_hidden_layers
    n_heads = config.n_head
    n_kv_heads = config.n_kv_head
    vocab_size = config.vocab_size
    max_seq_len = config.block_size

    header = struct.pack(
        "iiiiiii",
        dim,
        hidden_dim,
        n_layers,
        n_heads,
        n_kv_heads,
        -vocab_size,
        max_seq_len,
    )

    print("\n--- Model Architecture Target Layout ---")
    print(f"Embedding Dim (dim):       {dim}")
    print(f"SwiGLU Inner Dim (hidden): {hidden_dim}")
    print(f"Layers (n_layers):         {n_layers}")
    print(f"Query Heads (n_heads):     {n_heads}")
    print(f"KV Heads (n_kv_heads):     {n_kv_heads}")
    print(f"Vocabulary Size:           {vocab_size} (Tied Embeddings)")
    print(f"Max Sequence Length:       {max_seq_len}\n")

    total_bytes_written = 0

    with open(output_bin_path, "wb") as f:
        f.write(header)
        total_bytes_written += len(header)

        def write_tensor(tensor_name):
            nonlocal total_bytes_written
            if tensor_name not in state_dict:
                raise KeyError(
                    f"Expected tensor '{tensor_name}' was missing from Safetensors."
                )

            t = state_dict[tensor_name].detach().cpu().float()

            tensor_bytes = t.numpy().tobytes()
            f.write(tensor_bytes)
            total_bytes_written += len(tensor_bytes)

        print("Writing global input embedding table...")
        write_tensor("wte.weight")

        print("Writing sequential model layer blocks...")
        for i in range(n_layers):
            write_tensor(f"blocks.{i}.ln_1.norm.weight")
            write_tensor(f"blocks.{i}.ln_2.norm.weight")
            write_tensor(f"blocks.{i}.attn.q_proj.weight")
            write_tensor(f"blocks.{i}.attn.k_proj.weight")
            write_tensor(f"blocks.{i}.attn.v_proj.weight")
            write_tensor(f"blocks.{i}.attn.c_proj.weight")
            write_tensor(f"blocks.{i}.mlp.mlp.w1.weight")
            write_tensor(f"blocks.{i}.mlp.mlp.w2.weight")
            write_tensor(f"blocks.{i}.mlp.mlp.w3.weight")

        print("Writing final normalization layers...")
        write_tensor("ln_f.norm.weight")

    print(f"\nSuccess! Total File Size: {total_bytes_written / (1024 * 1024):.2f} MB")


if __name__ == "__main__":
    export_safetensors_to_bin("../SLMa-94M/model.safetensors", "model.bin")
