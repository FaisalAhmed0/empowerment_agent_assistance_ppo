#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="purejaxrl_continuous_control_dense_reward"

ENV_NAMES=("ant_u_maze" "halfcheetah")
TOTAL_TIMESTEPS_=(100000000)
LRS=(0.0003)
SEEDS=(30)
COMMENT="just_want_to_make_sure_that_this_codebase_works_okay_on_the_ant-u-maze_environment_as_it_will_be_my_starting_for_further_modification_for_my_teacher-student_framework."

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
NUM_ENVSS=(2048)
NUM_STEPS_=(10)
STUDENT_ENTROPY_COFFS=(0 0.05 0.005)
GAE_LAMBDA=(0.8 0.9 0.95)
CLIP_EPS=(0.2 0.3)
MAX_GRAD_NORM=(0.5 1.0)
UPDATE_EPOCHSS=(4 10)


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
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_entropy${student_entropy_coef}_num_envs${num_envs}_num_steps${num_steps}_gae_lambda${gae_lambda}_clip_eps${clip_eps}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_continuous_action_custom_brax.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --SEED=${SEED} \
                        --NUM_STEPS=${num_steps} \
                        --GAE_LAMBDA=${gae_lambda} \
                        --MAX_GRAD_NORM=${max_grad_norm} \
                        --UPDATE_EPOCHS=${update_epochs} \
                        --COMMENT=${COMMENT} \
                        --NUM_ENVS=${num_envs} \
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
echo "Total number of runs submitted: $run_count"
                        