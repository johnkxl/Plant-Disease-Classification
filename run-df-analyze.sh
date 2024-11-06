#!/bin/bash
# ######################################################################
# CSCI 444 Machine Learning
# Advanced Project Phase 2
#
# John Kendall
# Gabriel Gaspar
# Zephyr Zamora
#
# Script for running embedded images files in df-analyze
#
# ######################################################################

# usage: bash run-df-analyze.sh <df-analyze-dirpath> <datafile> <outdir-name>
# 
# `dir-analyze-dirpath` Th directory containing df-analyze.py and df-embed.oy
# 
# `datafile`    The name of the parquet file 
# 
# 

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

# run_df_analyze <df> <target> <outdir>
run_df_analyze_cluster () {
    ./run_python_with_home.sh python "$DF_ANALYZE" \
        --df "$(realpath $1)" \
        --target "$(realpath $2)" \
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
        --outdir "$(realpath $3)"
}

# ---------------------------------------------------------------------------
#                                   MAIN
# ---------------------------------------------------------------------------

# Command Line Arguments
DF_ANALYZE_PATH=$1 # ~/dev/machine-learning/df-analyze
DF=$2
OUT_DIR=$3
LOCAL=$4

# DF_EMBED="$DF_ANALYZE_PATH/df-embed.py"
DF_ANALYZE="$DF_ANALYZE_PATH/df-analyze.py"

if [ "$LOCAL" == "cluster" ]; then
    cd $DF_ANALYZE_PATH/containers
    ./build_container_cc.sh
    cd ../..
    ./run_python_with_home.sh "$DF" "target" "$OUT_DIR"
else
    ACTIVATE="$DF_ANALYZE_PATH/.venv/bin/activate"
    source "$ACTIVATE"
    run_df_analyze "$DF" "target" "$OUT_DIR"
fi

# END