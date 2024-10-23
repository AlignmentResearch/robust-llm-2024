from dataclasses import dataclass
from pathlib import Path

from datasets import Dataset

from robust_llm.config.configs import AttackScheduleConfig
from robust_llm.logging_utils import log


@dataclass
class AttackSchedule:
    config: AttackScheduleConfig
    num_rounds: int

    def __post_init__(self):
        """Process the attack schedule configuration.

        If num_round is 1 or below, we set the rate to 0 and ensure that
        start == end since we will run the attack at most once.
        """
        # We subtract one since we want to include the start and end values.
        num_rounds_minus_one = self.num_rounds - 1
        if self.config.end is not None and self.config.rate is not None:
            if self.num_rounds <= 1:
                if self.config.rate != 0:
                    raise ValueError("If num_rounds<=1, rate must be 0.")
                self.start = self.config.end
                self.end = self.config.end
                self.rate = 0.0
            else:
                self.start = self.config.end - int(
                    self.config.rate * num_rounds_minus_one
                )
                self.end = self.config.end
                self.rate = self.config.rate

        elif self.config.start is not None and self.config.rate is not None:
            if self.num_rounds <= 1:
                if self.config.rate != 0:
                    raise ValueError("If num_rounds<=1, rate must be 0.")
                self.start = self.config.start
                self.end = self.config.start
                self.rate = 0
            else:
                self.end = self.config.start + int(
                    self.config.rate * num_rounds_minus_one
                )
                self.start = self.config.start
                self.rate = self.config.rate

        elif self.config.start is not None and self.config.end is not None:
            if self.num_rounds <= 1:
                if self.config.start != self.config.end:
                    raise ValueError("If num_rounds<=1, start must equal end.")
                self.start = self.config.end
                self.end = self.config.end
                self.rate = 0
            else:
                # We subtract one because we want to include the start and end values.
                self.rate = (self.config.end - self.config.start) / (
                    num_rounds_minus_one
                )
                self.start = self.config.start
                self.end = self.config.end

        elif self.config.start is not None:
            self.start = self.config.start
            self.end = self.config.start
            self.rate = 0

        elif self.config.end is not None:
            self.start = self.config.end
            self.end = self.config.end
            self.rate = 0

        else:
            raise ValueError(f"Bad attack schedule config: {self.config}")

    def __getitem__(self, i: int) -> int:
        if i < 0 or i >= self.num_rounds:
            raise IndexError(f"Index {i} out of bounds for {self.num_rounds} rounds")
        return self.start + int(i * self.rate)


def construct_combined_dataset(
    ds1: Dataset, ds2: Dataset, index_map1: dict, index_map2: dict
):
    # Determine the size of the new dataset
    max_idx = max(
        max(index_map1.keys(), default=-1), max(index_map2.keys(), default=-1)
    )

    # Function to get a record from ds1 or ds2
    def get_record(idx):
        if idx in index_map1:
            return ds1[index_map1[idx]]
        elif idx in index_map2:
            return ds2[index_map2[idx]]
        else:
            raise ValueError(f"Index {idx} not found in either dataset")

    # Create a list of records
    records = [get_record(i) for i in range(max_idx + 1)]

    # Create a new dataset from the records
    return Dataset.from_list(records)


def get_sorted_checkpoints(path: Path) -> list[Path]:
    return sorted(path.iterdir(), reverse=True)


def find_most_recent_checkpoint(path: Path) -> Path:
    # Find the most recent epoch that is safely saved and load that
    for subdir in get_sorted_checkpoints(path):
        if (subdir / "save_complete").exists():
            log(f"Loading state from {subdir}")
            return subdir

    raise FileNotFoundError(f"No saved state found for {path}.")
