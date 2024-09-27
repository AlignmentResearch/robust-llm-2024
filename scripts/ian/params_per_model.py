"""For pretrain compute plots, we need to know how many parameters were trained"""

from transformers import GPTNeoXForCausalLM

model_sizes = [
    "14m",
    "31m",
    "70m",
    "160m",
    "410m",
    "1b",
    "1.4b",
    "2.8b",
    "6.9b",
    "12b",
]
no_embed_size_to_params = {}
with_embed_size_to_params = {}

for model_size in model_sizes:
    model = GPTNeoXForCausalLM.from_pretrained(f"EleutherAI/pythia-{model_size}")
    assert isinstance(model, GPTNeoXForCausalLM)
    n_params_with_embeds = model.num_parameters()
    n_params_no_embeds = model.num_parameters(exclude_embeddings=True)
    del model
    with_embed_size_to_params[model_size] = n_params_with_embeds
    no_embed_size_to_params[model_size] = n_params_no_embeds
    print(
        f"{model_size}:"
        f" {n_params_with_embeds} (with embeds), {n_params_no_embeds} (without embeds)"
    )

print(f"{with_embed_size_to_params = }")
# With GPTNeoXModel, with_embed_size_to_params = {'14m': 7628800, '31m': 17616896, '70m': 44670976, '160m': 123689472, '410m': 353822720, '1b': 908759040, '1.4b': 1311625216, '2.8b': 2646430720, '6.9b': 6650732544, '12b': 11586549760}  # noqa: E501
# With GPTNeoXForCausalLM, with_embed_size_to_params = {'14m': 14067712, '31m': 30494720, '70m': 70426624, '160m': 162322944, '410m': 405334016, '1b': 1011781632, '1.4b': 1414647808, '2.8b': 2775208960, '6.9b': 6857302016, '12b': 11846072320}  # noqa: E501
print(f"{no_embed_size_to_params = }")
# With GPTNeoXModel, no_embed_size_to_params = {'14m': 1189888, '31m': 4739072, '70m': 18915328, '160m': 85056000, '410m': 302311424, '1b': 805736448, '1.4b': 1208602624, '2.8b': 2517652480, '6.9b': 6444163072, '12b': 11327027200}  # noqa: E501
# With GPTNeoXForCausalLM, no_embed_size_to_params = {'14m': 7628800, '31m': 17616896, '70m': 44670976, '160m': 123689472, '410m': 353822720, '1b': 908759040, '1.4b': 1311625216, '2.8b': 2646430720, '6.9b': 6650732544, '12b': 11586549760}  # noqa: E501
