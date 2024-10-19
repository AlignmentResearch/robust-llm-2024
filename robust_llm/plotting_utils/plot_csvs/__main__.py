from robust_llm.plotting_utils.plot_csvs.adv_training import main as adv_training_main
from robust_llm.plotting_utils.plot_csvs.adv_training_transfer import (
    main as adv_training_transfer_main,
)
from robust_llm.plotting_utils.plot_csvs.asr_adv_training import (
    main as asr_adv_training_main,
)
from robust_llm.plotting_utils.plot_csvs.asr_finetuned import main as asr_finetuned_main
from robust_llm.plotting_utils.plot_csvs.asr_slopes import main as asr_slopes_main
from robust_llm.plotting_utils.plot_csvs.finetuned import main as finetuned_main
from robust_llm.plotting_utils.plot_csvs.offense_defense import (
    main as offense_defense_main,
)
from robust_llm.plotting_utils.plot_csvs.post_adv_training import (
    main as post_adv_training_main,
)

if __name__ == "__main__":
    adv_training_transfer_main()
    adv_training_main()
    asr_adv_training_main()
    asr_finetuned_main()
    asr_slopes_main()
    finetuned_main()
    offense_defense_main()
    post_adv_training_main()
