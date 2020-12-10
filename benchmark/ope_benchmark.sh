# first run benchmarks for lightgbm
python benchmark_off_policy_estimators.py\
    --n_runs 1\
    --base_model lightgbm\
    --behavior_policy random\
    --campaign all\
    --n_sim_to_compute_action_dist 1000000\
    --is_timeseries_split False\
    --n_jobs 10\
    --random_state 66

# then for random forest
python benchmark_off_policy_estimators.py\
    --n_runs 1\
    --base_model random_forest\
    --behavior_policy random\
    --campaign all\
    --n_sim_to_compute_action_dist 1000000\
    --is_timeseries_split False\
    --n_jobs 10\
    --random_state 66