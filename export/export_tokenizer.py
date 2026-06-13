import struct
from transformers import AutoTokenizer


def export_tokenizer(repo_id, output_bin):
    print(f"Loading tokenizer from Hugging Face: {repo_id}")
    tokenizer = AutoTokenizer.from_pretrained(repo_id)
    vocab_size = len(tokenizer)
    max_token_length = 128

    bos_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else 0

    with open(output_bin, "wb") as f:

        f.write(
            struct.pack("iiiii", max_token_length, vocab_size, bos_id, eos_id, unk_id)
        )

        for i in range(vocab_size):
            piece = tokenizer.convert_ids_to_tokens(i)

            if not (
                piece.startswith("<0x") and piece.endswith(">") and len(piece) == 6
            ):
                piece = piece.replace("Ġ", " ").replace("Ċ", "\n")

            piece_bytes = piece.encode("utf-8")
            length = len(piece_bytes)

            try:
                score = tokenizer._tokenizer.get_score(i)
            except:
                score = -100.0
            f.write(struct.pack("f", score))
            f.write(struct.pack("i", length))
            f.write(piece_bytes)

    print(f"Successfully generated clean {output_bin} with {vocab_size} tokens.")


if __name__ == "__main__":
    export_tokenizer("amartya-pandey/slm-tokenizer-24k", "tokenizer.bin")
