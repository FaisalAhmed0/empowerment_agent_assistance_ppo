#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="purejaxrl_continuous_control_with_goals_from_mlp_teacher_lp_reward_use_max_in_lp_smaller_lrs"
ADD_GOAL_REWARD="--ADD_GOAL_REWARD"
CONDITION_ON_GOAL="--CONDITION_ON_GOAL"
USE_LEARNING_PROGRESS_REWARD="--USE_LEARNING_PROGRESS_REWARD"
TEACHER_SOFTMAX_VIZ_NUM_SNAPSHOTSS=(0)
ENV_NAMES=("ant_u_maze_single_goal")
USE_MAX_IN_LP_REWARD="--USE_MAX_IN_LP_REWARD"
TOTAL_TIMESTEPS_=(300000000)
LRS=(0.0003 0.00003)
TEACHER_LRS=(0.0003 0.00003)
SEEDS=(30 0 8943)
COMMENT="Use_max_in_lp_reward"

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
NUM_ENVSS=(256)
NUM_STEPS_=(64)
STUDENT_ENTROPY_COFFS=(0)
GAE_LAMBDA=(0.8)
CLIP_EPS=(0.2) 
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
TEACHER_ENTROPY_COEFSS=(0.0 0.001 0.01)
TEACHER_NUM_MINIBATCHESS=(8 16)
TEACHER_UPDATE_EPOCHSS=(2 4 8)
TASK_REWARD_COEFSS=(1 2 5)

run_count=0

for ENV_NAME in "${ENV_NAMES[@]}"; do
  for TOTAL_TIMESTEPS in "${TOTAL_TIMESTEPS_[@]}"; do
    for LR in "${LRS[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                        for num_steps in "${NUM_STEPS_[@]}"; do
                        for student_entropy_coef in "${STUDENT_ENTROPY_COFFS[@]}"; do
                        for num_envs in "${NUM_ENVSS[@]}"; do
                        for gae_lambda in "${GAE_LAMBDA[@]}"; do
                        for clip_eps in "${CLIP_EPS[@]}"; do
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
                        ${ADD_GOAL_REWARD} \
                        ${CONDITION_ON_GOAL} \
                        --NUM_STEPS=${num_steps} \
                        --GAE_LAMBDA=${gae_lambda} \
                        --TEACHER_ROLLOUT_BUFFER_SIZE=${teacher_rollout_buffer_size} \
                        --TEACHER_NUM_MINIBATCHES=${teacher_num_minibatches} \
                        --TEACHER_UPDATE_EPOCHS=${teacher_update_epochs} \
                        --TASK_REWARD_COEF=${task_reward_coef} \
                        --TEACHER_ENT_COEF=${teacher_entropy_coef} \
                        --MAX_GRAD_NORM=${max_grad_norm} \
                        --UPDATE_EPOCHS=${update_epochs} \
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
echo "Total number of runs submitted: $run_count"
                        