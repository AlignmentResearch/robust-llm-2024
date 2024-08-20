## Describe your changes

## Issue ticket number and link

## Describe tests performed


## Checklist before requesting a review
- [ ] I have performed a self-review of my code
- [ ] If it is a new feature, I have added thorough tests.
- [ ] I have considered how it would affect the multi-GPU case and tested on a multi-GPU node if appropriate.
- [ ] If it is a high-risk change, I have run the following experiments to verify that the results are as expected:
    - [ ] [experiment=ian/027_eval_refactored_random_token_imdb model.name_or_path=AlignmentResearch/robust_llm_pythia-imdb-1.4b-mz-ada-v3](https://wandb.ai/farai/robust-llm/runs/zlynwimq)
    - [ ] [experiment=ian/028_eval_refactored_gcg_imdb model.name_or_path=AlignmentResearch/robust_llm_pythia-imdb-2.8b-mz-ada-v3](https://wandb.ai/farai/robust-llm/runs/pi6eq0wp)
    - [ ] [experiment=niki/1_DEBUG_niki_128](https://wandb.ai/farai/robust-llm/runs/76h9beb3)
