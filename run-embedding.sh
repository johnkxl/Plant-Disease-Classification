#!/bin/bash
# ######################################################################
# CSCI 444 Machine Learning
# Advanced Project Phase 2
#
# John Kendall
# Gabriel Gaspar
# Zephyr Zamora
#
# Script for embedding an image data subset using df-embed
#
# ######################################################################

# recommended: if using Mac, use caffeinate -i before the below command
# 
# usage: bash run-embedding.sh <df-analyze-dirpath> <data-dir> <data-file> 
# 
# `df-analyze-dirpath`  The directory containing df-analyze.py and df-embed.py
# 
# `data-dir`            The directory containing `datafile`
# 
# `data-file`           The file name of the parquet file containing images to 
#                       be embedded. NOTE: this is not the relative path; just 
#                       the file name.
# 
# The parquet file produced will be saved in the same directory as `data-file`



# ---------------------------------------------------------------------------
#                           FUNCTION DECLARATIONS
# ---------------------------------------------------------------------------

# run_df_analyze <df> <target> <outdir>
run_df_analyze () {
    python $DF_ANALYZE \
        --df "$1" \
        --target "$2" \
        --mode classify \
        --classifiers knn lgbm rf lr sgd dummy gandalf \
        --regressors knn lgbm rf elastic sgd dummy gandalf \
        --feat-select filter embed wrap \
        --redundant-wrapper-selection \
        --embed-select lgbm linear \
        --wrapper-select step-up \
        --wrapper-model linear \
        --norm robust \
        --nan median \
        --filter-method assoc pred \
        --filter-assoc-cont-classify mut_info \
        --filter-assoc-cat-classify mut_info \
        --filter-assoc-cont-regress mut_info \
        --filter-assoc-cat-regress mut_info \
        --filter-pred-regress mae \
        --filter-pred-classify acc \
        --htune-trials 50 \
        --htune-cls-metric acc \
        --htune-reg-metric mae \
        --test-val-size 0.4 \
        --outdir "$3"
}


# ---------------------------------------------------------------------------
#                                   MAIN
# ---------------------------------------------------------------------------

# ~/dev/machine-learning/df-analyze
DF_ANALYZE_PATH=$1
DF_EMBED= "$DF_ANALYZE_PATH/df-embed.py"
DF_ANALYZE= "$DF_ANALYZE_PATH/df-analyze.py"
ACTIVATE= "$DF_ANALYZE_PATH/.venv/bin/activate"

DATA_DIR=$2
DATA=$3

# dataset/color/<subset>
subset_name=$(echo ${DATA_DIR} | cut -d '/' -f 3)
outname= "$subset_name-results.parquet"

if [ ! -e "$DATA" ]
then
    echo "file $DATA does not exist" \n terminating execution...
    exit 1
fi

source "$ACTIVATE"

# Save the embedded images parquet file in the same directory as `data-file`
python "$DF_EMBED" \
    --modality vision \
    --data "$DATA_DIR/$DATA" \
    --out "$DATA_DIR/$outname"

# Save the image embeddings concatented with leaf IDs into a csv file
python concat-embeddings-to-df.py "$DATA_DIR/$DATA_DIR.csv" "$DATA_DIR/$outname"
concat_df_stem=$(echo ${outname} | cut -d '.' -f 1)
concat_df_file="$concat_df_stem-complete.csv" # name of resulting file

ALL_RESULTS_DIR="all-results/"
if [ ! -d "$ALL_RESULTS_DIR" ]
then 
    mkdir "$ALL_RESULTS_DIR"
fi

subset_results_dir="$ALL_RESULTS_DIR/$subset_name"

run_df_analyze "$DATA_DIR/$concat_df_file" "target" "$subset_results_dir"

# END