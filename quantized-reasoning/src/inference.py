from mlx_lm import load
import mlx.core as mx

MODEL_NAME = "mlx-community/Qwen3-4B-Instruct-2507-bf16"


def run_inference(model, tokenizer, prompt: str, max_new_tokens: int = 20) -> str:

    messages = [
        {"role": "user", "content": prompt}
    ]

    formatted_prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    # 1. ENCODE
    token_ids = tokenizer.encode(formatted_prompt)


    # Convert token IDs into an MLX array the model can process
    input_ids = mx.array([token_ids])
    generated_tokens = []
    eos_token_id = tokenizer.eos_token_id
    for _ in range(max_new_tokens):
        logits = model(input_ids)
        next_token_logits = logits[:, -1, :]
        next_token_id = int(mx.argmax(next_token_logits, axis=-1).item())
        if next_token_id == eos_token_id:
            break
        generated_tokens.append(next_token_id)
        next_token = mx.array([[next_token_id]])
        input_ids = mx.concatenate([input_ids, next_token], axis=1)

    decoded_resp = tokenizer.decode(generated_tokens)
    return decoded_resp


if __name__ == "__main__":
    model, tokenizer = load(MODEL_NAME)
    prompt = (
        "Alice is taller than Bob. "
        "Bob is taller than Charlie. "
        "Who is taller, Alice or Charlie? "
        "Answer with only the name."
    )

    print("\nDecoded next token:")
    print(run_inference(model, tokenizer, prompt))
