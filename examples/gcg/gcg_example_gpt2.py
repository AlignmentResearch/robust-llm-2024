import torch
from transformers import AutoTokenizer, GPT2LMHeadModel, PreTrainedModel

from robust_llm.attacks.search_based.runners import GCGRunner
from robust_llm.attacks.search_based.utils import PromptTemplate, get_wrapped_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    assert isinstance(model, PreTrainedModel)
    model.to(device=device)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    user_prompt = "Hello there."
    prompt_template = PromptTemplate(before_attack=user_prompt)
    # specify parameters for the attack
    wrapped_model = get_wrapped_model(model, tokenizer)
    runner = GCGRunner(
        wrapped_model=wrapped_model,
        top_k=8,
        n_candidates_per_it=32,
        n_its=50,
        target="TARGET",
        n_attack_tokens=10,
        prompt_template=prompt_template,
    )

    # run the attack with the parameters specified above
    attack_text, _ = runner.run()
    print(f"{attack_text=}")

    full_prompt = prompt_template.build_prompt(attack_text=attack_text, target="")
    print(f"{full_prompt=}")
    # confirm that the suffix works by using it to generate a continuation
    tokens = tokenizer(full_prompt, return_tensors="pt").input_ids.to(device=device)
    all_tokens = model.generate(tokens, max_new_tokens=5)
    print(f"{all_tokens=}")
    print(f"{tokenizer.decode(all_tokens[0])=} ")


if __name__ == "__main__":
    main()
