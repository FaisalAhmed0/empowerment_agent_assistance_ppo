#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="minigrid_with_teacher"

ENV_NAMES=("Navix-DoorKey-16x16-v0")
TOTAL_TIMESTEPS_=(50000000)
LRS=(0.00025 0.0025)
SEEDS=(30)

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
NUM_ENVSS=(128 256)
NUM_STEPS_=(128 64)
STUDENT_ENTROPY_COFFS=(0 0.001 0.01)


run_count=0

for ENV_NAME in "${ENV_NAMES[@]}"; do
  for TOTAL_TIMESTEPS in "${TOTAL_TIMESTEPS_[@]}"; do
    for LR in "${LRS[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                        for num_steps in "${NUM_STEPS_[@]}"; do
                        for student_entropy_coef in "${STUDENT_ENTROPY_COFFS[@]}"; do
                        for num_envs in "${NUM_ENVSS[@]}"; do
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_teacherlr${teacher_lr}_probeagg${teacher_probe_agg}_treward${teacher_reward_type}_sgoal${student_goal_reward_type}_teplen${teacher_episode_length}_sampleevery${teacher_sample_every_n_episodes}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_minigrid_no_teacher.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --SEED=${SEED} \
                        --NUM_STEPS=${num_steps} \
                        --NUM_ENVS=${num_envs} \
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
echo "Total number of runs submitted: $run_count"
                        