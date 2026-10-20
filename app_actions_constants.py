RESET_STATE = "reset_state"

# Data Cleanup actions
PARSE_JSON = "parse_json"
CLEANUP_DATA = "clean_data"

# Data Preprocessing actions
CALCULATE_ORIGINAL_TIME_DELTAS = "calculate_original_time_deltas"
CREATE_MERGED_DATASET = "create_merged_dataset"
CALCULATE_MERGED_TIME_DELTAS = "calculate_merged_time_deltas"
CALCULATE_MERGED_EVENTS_PER_MINUTE = "calculate_merged_events_per_minute"

# Pipeline Explorer actions
CALCULATE_ALL_PIPELINE_METADATA = "calculate_all_pipeline_metadata"
VALIDATE_PIPELINE = "validate_pipeline"
ANALYSIS_PER_RECORD = "analysis_per_record"

# Model Training actions
TRAIN_MODELS = "train_models"

# Model Serving actions
GET_PREDICTION = "get_prediction"
REFRESH_PREDICTIONS = "refresh_predictions"

# Model Lifecycle actions
LIST_CANDIDATE_RUNS = "list_candidate_runs"
REGISTER_RUN_AS_MODEL = "register_run_as_model"
SET_MODEL_STAGE = "set_model_stage"
ROLLBACK_MODEL = "rollback_model"
GET_CURRENT_MODELS = "get_current_models"

# RL Training actions
TRAIN_RL_AGENT = "train_rl_agent"
EVALUATE_RL_POLICY = "evaluate_rl_policy"

# Sync Policy Evaluation actions
EVALUATE_SYNC_POLICY = "evaluate_sync_policy"
COMPARE_POLICIES = "compare_policies"

# Dataset Comparison actions
COMPUTE_FINGERPRINTS = "compute_fingerprints"
COMPUTE_PAIRWISE_COMPARISON = "compute_pairwise_comparison"
CLUSTER_DATASETS = "cluster_datasets"
CLUSTER_RECORDS = "cluster_records"
