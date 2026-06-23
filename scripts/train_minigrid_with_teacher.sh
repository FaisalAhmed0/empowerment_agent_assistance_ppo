#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="minigrid_with_teacher"

ENV_NAMES=("Navix-DoorKey-16x16-v0")
TOTAL_TIMESTEPS_=(50000000)
LRS=(0.00025 0.0025)
SEEDS=(30)

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
TEACHER_REWARD_TYPE=(competence_lp)
ABSOLUTE_LP=(--TEACHER_LP_ABSOLUTE --no-TEACHER_LP_ABSOLUTE)
NUM_ENVSS=(128 256)
TEACHER_LR=(0.00025 0.0025)
TEACHER_EPISODE_LENGTHS=(2 5 10)
NUM_STEPS_=(128 64)
TEACHER_ENTROPY_COFFS=(0 0.01 0.001)
STUDENT_ENTROPY_COFFS=(0 0.001 0.01)


run_count=0

for ENV_NAME in "${ENV_NAMES[@]}"; do
  for TOTAL_TIMESTEPS in "${TOTAL_TIMESTEPS_[@]}"; do
    for LR in "${LRS[@]}"; do
          for teacher_reward_type in "${TEACHER_REWARD_TYPE[@]}"; do
              for teacher_lr in "${TEACHER_LR[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                        for num_steps in "${NUM_STEPS_[@]}"; do
                        for teacher_entropy_coef in "${TEACHER_ENTROPY_COFFS[@]}"; do
                        for student_entropy_coef in "${STUDENT_ENTROPY_COFFS[@]}"; do
                        for num_envs in "${NUM_ENVSS[@]}"; do
                        for teacher_episode_length in "${TEACHER_EPISODE_LENGTHS[@]}"; do
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_teacherlr${teacher_lr}_probeagg${teacher_probe_agg}_treward${teacher_reward_type}_sgoal${student_goal_reward_type}_teplen${teacher_episode_length}_sampleevery${teacher_sample_every_n_episodes}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_minigrid.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --SEED=${SEED} \
                        ${ABSOLUTE_LP} \
                        --NUM_STEPS=${num_steps} \
                        --NUM_ENVS=${num_envs} \
                        --PROJECT=\"${WANDB_PROJECT_NAME}\" \
                        --TEACHER_REWARD_TYPE=${teacher_reward_type} \
                        --TEACHER_LR=${teacher_lr} \
                        --TEACHER_ENT_COEF=${teacher_entropy_coef} \
                        --ENT_COEF=${student_entropy_coef} \
                        --TEACHER_EPISODE_LENGTH=${teacher_episode_length}"
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
                        