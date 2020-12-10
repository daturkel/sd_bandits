Follow this link for guide on how to use benchmarking https://github.com/st-tech/zr-obp/blob/master/benchmark/ope/README.md

Make sure the full open bandit dataset is set in the `../obd_full/` directory

Right now, the settings in [ope_train.sh](ope_train.sh) and [ope_benchmark.sh](ope_benchmark.sh) are to exactly replicate the ZOZOTown paper, with 30 bootstrapped samples with 1,000,000 simulations each to compute the action distribution for the 'ground truth' estimate, on `random_forest` and `lightgbm` methods.

However, we can lower either of these to get faster, less robust results.