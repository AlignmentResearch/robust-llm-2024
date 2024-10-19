from robust_llm.plotting_utils.generate_csvs.adv_training import (
    main as adv_training_main,
)
from robust_llm.plotting_utils.generate_csvs.adv_training_transfer import (
    main as adv_training_transfer_main,
)
from robust_llm.plotting_utils.generate_csvs.asr_adv_training import (
    main as asr_adv_training_main,
)
from robust_llm.plotting_utils.generate_csvs.asr_finetuned import (
    main as asr_finetuned_main,
)
from robust_llm.plotting_utils.generate_csvs.finetuned import main as finetuned_main
from robust_llm.plotting_utils.generate_csvs.offense_defense import (
    main as offense_defense_main,
)
from robust_llm.plotting_utils.generate_csvs.post_adv_training import (
    main as post_adv_training_main,
)

if __name__ == "__main__":
    print("Generating CSVs...")
    print("Transfer experiments...")
    adv_training_transfer_main()
    print("Regular adversarial training experiments...")
    adv_training_main()
    print("Attack scaling for adversarially trained models...")
    asr_adv_training_main()
    print("Attack scaling for finetuned models...")
    asr_finetuned_main()
    print("Finetuning experiments...")
    finetuned_main()
    print("Offense-defense experiments...")
    offense_defense_main()
    print("Finetuning-like experiments for adversarially trained models...")
    post_adv_training_main()
