import torch
from transformers import AutoTokenizer, BertForSequenceClassification, PreTrainedModel

from robust_llm.attacks.search_based.runners import GCGRunner
from robust_llm.attacks.search_based.utils import PromptTemplate, get_wrapped_model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertForSequenceClassification.from_pretrained("bert-base-uncased")
    assert isinstance(model, PreTrainedModel)
    model = model.to(device=device)  # type: ignore
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    user_prompt = "Hello there."
    prompt_template = PromptTemplate(before_attack=user_prompt)
    # specify parameters for the attack
    wrapped_model = get_wrapped_model(model, tokenizer)

    def run_with_target(clf_target: int):
        runner = GCGRunner(
            wrapped_model=wrapped_model,
            top_k=256,
            n_candidates_per_it=512,
            n_its=40,
            seq_clf=True,
            clf_target=clf_target,
            n_attack_tokens=10,
            prompt_template=prompt_template,
        )

        print(f"For {clf_target=}:")
        # run the attack with the parameters specified above
        attack_text, _ = runner.run()
        print(f"{attack_text=}")

        # confirm that the suffix works by using it to generate a continuation
        prompt = runner.prompt_template.build_prompt(attack_text=attack_text)
        tokens = tokenizer(prompt, return_tensors="pt").input_ids.to(device=device)
        print(tokens)
        logits = model(tokens).logits
        print(f"{logits=}")
        print(f"{torch.softmax(logits, dim=1)=}")

    run_with_target(0)
    run_with_target(1)


if __name__ == "__main__":
    main()
