#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="purejaxrl_ant_u_maze_single_goal"

ENV_NAMES=("ant_u_maze_single_goal")
SAVE_MODEL="--SAVE_MODEL"
TOTAL_TIMESTEPS_=(300000000)
LRS=(0.0003)
SEEDS=(30 0 8943)
COMMENT="no_teacher_baseline"

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
NUM_ENVSS=(256)
NUM_STEPS_=(64)
STUDENT_ENTROPY_COFFS=(0)
GAE_LAMBDA=(0.8)
CLIP_EPS=(0.2)
MAX_GRAD_NORM=(1.0)
UPDATE_EPOCHSS=(4)
HIDDEN_DIMS=(256)
NORMALIZE_ENVS=(--NORMALIZE_ENV)
NUM_MINIBATCHES=(8 16)


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
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_entropy${student_entropy_coef}_num_envs${num_envs}_num_steps${num_steps}_gae_lambda${gae_lambda}_clip_eps${clip_eps}_hidden_dim${hidden_dim}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_continuous_action_custom_brax.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --SEED=${SEED} \
                        --NUM_STEPS=${num_steps} \
                        --NUM_MINIBATCHES=${num_minibatches} \
                        --HIDDEN_DIM=${hidden_dim} \
                        --GAE_LAMBDA=${gae_lambda} \
                        --MAX_GRAD_NORM=${max_grad_norm} \
                        --UPDATE_EPOCHS=${update_epochs} \
                        ${SAVE_MODEL} \
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
echo "Total number of runs submitted: $run_count"
                        