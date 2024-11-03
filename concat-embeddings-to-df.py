# ######################################################################
# CSCI 444 Machine Learning
# Advanced Project Phase 2
# 
# John Kendall
# Gabriel Gaspar
# Zephyr Zamora
# 
# Script concatentaing the image embeddings to the DataFrame conatining
# the leaf IDs.
# 
# ######################################################################

import sys
import pandas as pd
from pathlib import Path


def main() -> int:
    METADATA_PATH = Path(sys.argv[1])
    EMBEDDED_PATH = Path(sys.argv[2])

    out_name = f"{EMBEDDED_PATH.stem}-complete.csv"
    meta_dir = METADATA_PATH.parent.as_posix()
    embedded_dir = EMBEDDED_PATH.parent.as_posix()
    out_dir = Path.cwd()
    if meta_dir == embedded_dir:
        out_dir = out_dir / Path(meta_dir)
    
    metadata_df = pd.read_parquet(METADATA_PATH)
    embedded_df = pd.read_parquet(EMBEDDED_PATH)
    df = metadata_df.join(embedded_df)

    csv_file = f"{out_name}.csv"
    df.to_csv(out_dir / out_name, index=False)
    print(f"In {out_dir}")
    print(f"Created {out_name} from {METADATA_PATH.name} and {EMBEDDED_PATH.name}")
    return 0


if __name__ == "__main__":
    main()