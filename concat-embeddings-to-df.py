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

# usage: python concat-embeddings-to-df.py <metadata.csv> <embedded.parquet>

# NOTE
# So far, this has not been run, because image count was reduced in embedding 
# (~25%), causing a loss in the ability to concat them again


import sys
import pandas as pd
from pathlib import Path


def main() -> int:
    METADATA_PATH = Path(sys.argv[1])
    EMBEDDED_PATH = Path(sys.argv[2])

    out_name = f"{EMBEDDED_PATH.stem}-complete.parquet"
    meta_dir = METADATA_PATH.parent.as_posix()
    embedded_dir = EMBEDDED_PATH.parent.as_posix()
    out_dir = Path.cwd()
    if meta_dir == embedded_dir:
        out_dir = out_dir / Path(meta_dir)
    
    metadata_df = pd.read_csv(METADATA_PATH)
    embedded_df = pd.read_parquet(EMBEDDED_PATH)
    # Join the metadata and embedded images. (Embedding should have mainting ordering+count of records.)
    df = metadata_df.join(embedded_df)

    df.to_parquet(out_dir / out_name, index=False)
    print(f"In {out_dir}")
    print(f"Created {out_name} from {METADATA_PATH.name} and {EMBEDDED_PATH.name}")
    return 0


if __name__ == "__main__":
    main()