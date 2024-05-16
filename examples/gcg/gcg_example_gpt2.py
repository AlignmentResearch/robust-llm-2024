from accelerate import Accelerator
from transformers import AutoTokenizer, GPT2LMHeadModel

from robust_llm.attacks.search_based.runners import GCGRunner
from robust_llm.attacks.search_based.utils import PreppedExample, PromptTemplate
from robust_llm.models import GPT2Model
from robust_llm.models.model_utils import InferenceType


def main():
    accelerator = Accelerator()
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    user_prompt = "Hello there."
    prompt_template = PromptTemplate(before_attack=user_prompt)
    # specify parameters for the attack
    assert isinstance(model, GPT2LMHeadModel)
    wrapped_model = GPT2Model(
        model, tokenizer, accelerator, inference_type=InferenceType("generation")
    )

    prepped_examples = [
        PreppedExample(
            prompt_template=prompt_template,
            clf_target=-1,  # causal models don't have a clf_target (old)
        )
    ]

    runner = GCGRunner(
        wrapped_model=wrapped_model,
        top_k=8,
        n_candidates_per_it=32,
        n_its=50,
        target="TARGET",
        n_attack_tokens=10,
        prepped_examples=prepped_examples,
    )

    # run the attack with the parameters specified above
    attack_text, _ = runner.run()
    print(f"{attack_text=}")

    full_prompt = runner.example.prompt_template.build_prompt(
        attack_text=attack_text, target=""
    )
    print(f"{full_prompt=}")
    # confirm that the suffix works by using it to generate a continuation
    tokens = tokenizer(full_prompt, return_tensors="pt").input_ids.to(
        device=accelerator.device
    )
    all_tokens = model.generate(tokens, max_new_tokens=5)
    print(f"{all_tokens=}")
    print(f"{tokenizer.decode(all_tokens[0])=} ")


if __name__ == "__main__":
    main()
