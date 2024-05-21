from accelerate import Accelerator

from robust_llm.attacks.search_based.runners import GCGRunner
from robust_llm.attacks.search_based.utils import PreppedExample, PromptTemplate
from robust_llm.config.model_configs import ModelConfig
from robust_llm.models.wrapped_model import WrappedModel


def main():
    accelerator = Accelerator()
    model_config = ModelConfig(
        name_or_path="gpt2",
        family="gpt2",
        inference_type="generation",
        train_minibatch_size=2,
        eval_minibatch_size=2,
    )
    victim = WrappedModel.from_config(model_config, accelerator=accelerator)

    user_prompt = "Hello there."
    prompt_template = PromptTemplate(before_attack=user_prompt)

    prepped_examples = [
        PreppedExample(
            prompt_template=prompt_template,
            clf_target=-1,  # causal models don't have a clf_target (old)
        )
    ]

    runner = GCGRunner(
        wrapped_model=victim,
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
    tokens = victim.tokenizer(full_prompt, return_tensors="pt").input_ids.to(
        device=victim.device
    )
    all_tokens = victim.model.generate(tokens, max_new_tokens=5)
    print(f"{all_tokens=}")
    print(f"{victim.tokenizer.decode(all_tokens[0])=} ")


if __name__ == "__main__":
    main()
