#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="lp_teacher_all_envs_final"
ADD_GOAL_REWARD="--ADD_GOAL_REWARD"
CONDITION_ON_GOAL="--CONDITION_ON_GOAL"
USE_LEARNING_PROGRESS_REWARD="--USE_LEARNING_PROGRESS_REWARD"
TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTSS=(0)
ENV_NAMES=("humanoid_u_maze_single_goal")
USE_MAX_IN_LP_REWARD="--USE_MAX_IN_LP_REWARD"
TEACHER_NORMALIZE_ADVANTAGES="--TEACHER_NORMALIZE_ADVANTAGES"
TEACHER_CONDITION_ONLY_ON_COMPETENCE="--no-TEACHER_CONDITION_ONLY_ON_COMPETENCE"
SEPARATE_Z_GOAL_PENALTY="--no-SEPARATE_Z_GOAL_PENALTY"
TOTAL_TIMESTEPS_=(500000000)
LRS=(0.0003)
TEACHER_LRS=(0.0003)
SEEDS=(30 75937 123 1 842434353)
COMMENT="granular_sweep_over_teacher_eps_both_ent_coeff_teacher_lr_lp_ema_alpha"
SAVE_MODEL="--SAVE_MODEL"

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
NUM_ENVSS=(512 1024)
NUM_STEPS_=(64)
STUDENT_ENTROPY_COFFS=(0.001 0.0) 
GAE_LAMBDA=(0.8)
CLIP_EPSS=(0.2) 
TEACHER_CLIP_EPSS=(0.3)
MAX_GRAD_NORM=(1.0)
UPDATE_EPOCHSS=(4)
NUM_MINIBATCHES=(8)
NORMALIZE_ENVS=(--NORMALIZE_ENV)
HIDDEN_DIMS=(256)
GOAL_REWARD_COEF=(1)
### Teacher hyperparameters
TEACHER_ROLLOUT_BUFFER_SIZES=(1)
ABSOLUTE_LEARNING_PROGRESSS=(--no-ABSOLUTE_LEARNING_PROGRESS)
NUM_EVAL_ENVSS=(8)
TEACHER_ENTROPY_COEFSS=(0.01 0.005 0.001)
TEACHER_NUM_MINIBATCHESS=(8)
TEACHER_UPDATE_EPOCHSS=(8)
TASK_REWARD_COEFSS=(1 2 5 10)
LP_EMA_ALPHAS=(0.1)
OBS_NORM_WARMUP_STEPSS=(5000)
TEACHER_NUM_GOAL_POINTS=(30)
GOAL_REACH_EPSILONSS=(1.0)
TEACHER_GOAL_X_MINS=(2.0)
TEACHER_GOAL_X_MAXS=(14.0)
TEACHER_GOAL_Y_MINS=(2.0)
TEACHER_GOAL_Y_MAXS=(14.0)

run_count=0

for ENV_NAME in "${ENV_NAMES[@]}"; do
  for TOTAL_TIMESTEPS in "${TOTAL_TIMESTEPS_[@]}"; do
    for LR in "${LRS[@]}"; do
                        for num_steps in "${NUM_STEPS_[@]}"; do
                        for student_entropy_coef in "${STUDENT_ENTROPY_COFFS[@]}"; do
                        for num_envs in "${NUM_ENVSS[@]}"; do
                        for gae_lambda in "${GAE_LAMBDA[@]}"; do
                        for clip_eps in "${CLIP_EPSS[@]}"; do
                        for max_grad_norm in "${MAX_GRAD_NORM[@]}"; do
                        for update_epochs in "${UPDATE_EPOCHSS[@]}"; do
                        for normalize_env in "${NORMALIZE_ENVS[@]}"; do
                        for hidden_dim in "${HIDDEN_DIMS[@]}"; do
                        for num_minibatches in "${NUM_MINIBATCHES[@]}"; do
                        for goal_reward_coef in "${GOAL_REWARD_COEF[@]}"; do
                        for teacher_rollout_buffer_size in "${TEACHER_ROLLOUT_BUFFER_SIZES[@]}"; do
                        for absolute_learning_progress in "${ABSOLUTE_LEARNING_PROGRESSS[@]}"; do
                        for num_eval_envs in "${NUM_EVAL_ENVSS[@]}"; do
                        for teacher_softmax_viz_num_snapshots in "${TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTSS[@]}"; do
                        for teacher_num_minibatches in "${TEACHER_NUM_MINIBATCHESS[@]}"; do
                        for teacher_update_epochs in "${TEACHER_UPDATE_EPOCHSS[@]}"; do
                        for task_reward_coef in "${TASK_REWARD_COEFSS[@]}"; do
                        for teacher_entropy_coef in "${TEACHER_ENTROPY_COEFSS[@]}"; do
                        for teacher_clip_eps in "${TEACHER_CLIP_EPSS[@]}"; do
                        for teacher_lr in "${TEACHER_LRS[@]}"; do
                        for lp_ema_alpha in "${LP_EMA_ALPHAS[@]}"; do
                        for SEED in "${SEEDS[@]}"; do
                        for obs_norm_warmup_steps in "${OBS_NORM_WARMUP_STEPSS[@]}"; do
                        for goal_reach_epsilon in "${GOAL_REACH_EPSILONSS[@]}"; do
                        for teacher_goal_x_min in "${TEACHER_GOAL_X_MINS[@]}"; do
                        for teacher_goal_x_max in "${TEACHER_GOAL_X_MAXS[@]}"; do
                        for teacher_goal_y_min in "${TEACHER_GOAL_Y_MINS[@]}"; do
                        for teacher_goal_y_max in "${TEACHER_GOAL_Y_MAXS[@]}"; do
                        for teacher_num_goal_points in "${TEACHER_NUM_GOAL_POINTS[@]}"; do
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_entropy${student_entropy_coef}_num_envs${num_envs}_num_steps${num_steps}_gae_lambda${gae_lambda}_clip_eps${clip_eps}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_continuous_action_custom_brax_with_teacher_simple_reward.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --HIDDEN_DIM=${hidden_dim} \
                        --NUM_MINIBATCHES=${num_minibatches} \
                        ${USE_LEARNING_PROGRESS_REWARD} \
                        ${USE_MAX_IN_LP_REWARD} \
                        --SEED=${SEED} \
                        ${TEACHER_CONDITION_ONLY_ON_COMPETENCE} \
                        ${ADD_GOAL_REWARD} \
                        ${CONDITION_ON_GOAL} \
                        --NUM_STEPS=${num_steps} \
                        --GOAL_REACH_EPSILON=${goal_reach_epsilon} \
                        ${TEACHER_NORMALIZE_ADVANTAGES} \
                        --TEACHER_NUM_GOAL_POINTS=${teacher_num_goal_points} \
                        --LP_EMA_ALPHA=${lp_ema_alpha} \
                        ${SAVE_MODEL} \
                        ${SEPARATE_Z_GOAL_PENALTY} \
                        --GAE_LAMBDA=${gae_lambda} \
                        --TEACHER_ROLLOUT_BUFFER_SIZE=${teacher_rollout_buffer_size} \
                        --TEACHER_CLIP_EPS=${teacher_clip_eps} \
                        --TEACHER_NUM_MINIBATCHES=${teacher_num_minibatches} \
                        --TEACHER_LR=${teacher_lr} \
                        --OBS_NORM_WARMUP_STEPS=${obs_norm_warmup_steps} \
                        --TEACHER_UPDATE_EPOCHS=${teacher_update_epochs} \
                        --TASK_REWARD_COEF=${task_reward_coef} \
                        --TEACHER_ENT_COEF=${teacher_entropy_coef} \
                        --MAX_GRAD_NORM=${max_grad_norm} \
                        --UPDATE_EPOCHS=${update_epochs} \
                        --TEACHER_GOAL_X_MIN=${teacher_goal_x_min} \
                        --TEACHER_GOAL_X_MAX=${teacher_goal_x_max} \
                        --TEACHER_GOAL_Y_MIN=${teacher_goal_y_min} \
                        --TEACHER_GOAL_Y_MAX=${teacher_goal_y_max} \
                        --TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTS=${teacher_softmax_viz_num_snapshots} \
                        --GOAL_REWARD_COEF=${goal_reward_coef} \
                        --NUM_EVAL_ENVS=${num_eval_envs} \
                        ${absolute_learning_progress} \
                        --COMMENT=${COMMENT} \
                        --NUM_ENVS=${num_envs} \
                        ${normalize_env} \
                        --CLIP_EPS=${clip_eps} \
                        --PROJECT=\"${WANDB_PROJECT_NAME}\" \
                        --ENT_COEF=${student_entropy_coef}"
                      eval ${CMD}
                      run_count=$((run_count + 1))
                    done
                  done
                done
              done
              done
            done
          done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
        done
echo "Total number of runs submitted: $run_count"
                        