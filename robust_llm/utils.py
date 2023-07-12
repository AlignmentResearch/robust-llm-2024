def check_input_length(input_text, tokenizer):
    max_length = tokenizer.max_length

    inputs = tokenizer.encode(input_text, truncation=True, padding=False)
    input_length = len(inputs)

    if input_length > max_length:
        print(
            f"Warning: Input length ({input_length}) exceeds the maximum model length ({max_length})."
        )
        raise ValueError
