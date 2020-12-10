# Train regression model on lightgm with 30 runs

python train_regression_model.py\
    --n_runs 30\
    --base_model lightgbm\
    --behavior_policy random\
    --campaign all\
    --n_sim_to_compute_action_dist 1000000\
    --is_timeseries_split False\
    --is_mrdr False\
    --n_jobs 10\
    --random_state 66

# Then do mrdr training
python train_regression_model.py\
    --n_runs 30\
    --base_model lightgbm\
    --behavior_policy random\
    --campaign all\
    --n_sim_to_compute_action_dist 1000000\
    --is_timeseries_split False\
    --is_mrdr True\
    --n_jobs 10\
    --random_state 66

# Then train random forest

python train_regression_model.py\
    --n_runs 30\
    --base_model random_forest\
    --behavior_policy random\
    --campaign all\
    --n_sim_to_compute_action_dist 1000000\
    --is_timeseries_split False\
    --is_mrdr False\
    --n_jobs 10\
    --random_state 66

python train_regression_model.py\
    --n_runs 30\
    --base_model random_forest\
    --behavior_policy random\
    --campaign all\
    --n_sim_to_compute_action_dist 1000000\
    --is_timeseries_split False\
    --is_mrdr True\
    --n_jobs 10\
    --random_state 66