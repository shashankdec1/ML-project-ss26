import json
import sys
from pathlib import Path

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import DATA_PATH
from src.data import FEATURES


if __name__ == "__main__":
    sample = pd.read_csv(DATA_PATH).dropna(subset=FEATURES).iloc[0][FEATURES].to_dict()
    print(json.dumps(sample, indent=2))
