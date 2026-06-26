#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="minigrid_with_teacher_empowerment"

ENV_NAMES=("Navix-DoorKey-16x16-v0")
TOTAL_TIMESTEPS_=(500000000)
LRS=(0.00025)
SEEDS=(30)

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
TEACHER_REWARD_TYPE=(empowerment)
NUM_ENVSS=(32 128)
TEACHER_LR=(0.00025 0.0025)
TEACHER_BATCH_SIZES=(32 256 512 1280)
NUM_STEPS_=(128)
TEACHER_ENTROPY_COFFS=(0.001 0.1 0.01)
STUDENT_ENTROPY_COFFS=(0.001 0.1 0.01)
EMPOWERMENT_ENERGY_FNS=(norm l2 cosine dot)
GAMMA_CLS=(0.7 0.99)


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
                        for teacher_batch_size in "${TEACHER_BATCH_SIZES[@]}"; do
                        for empowerment_energy_fn in "${EMPOWERMENT_ENERGY_FNS[@]}"; do
                        for gamma_cl in "${GAMMA_CLS[@]}"; do
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_teacherlr${teacher_lr}_probeagg${teacher_probe_agg}_treward${teacher_reward_type}_sgoal${student_goal_reward_type}_tbatch${teacher_batch_size}_sampleevery${teacher_sample_every_n_episodes}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_minigrid_empowerment_batch_v.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --SEED=${SEED} \
                        --NUM_STEPS=${num_steps} \
                        --NUM_ENVS=${num_envs} \
                        --PROJECT=\"${WANDB_PROJECT_NAME}\" \
                        --TEACHER_REWARD_TYPE=${teacher_reward_type} \
                        --TEACHER_LR=${teacher_lr} \
                        --TEACHER_ENT_COEF=${teacher_entropy_coef} \
                        --GAMMA_CL=${gamma_cl} \
                        --ENT_COEF=${student_entropy_coef} \
                        --EMPOWERMENT_ENERGY_FN=${empowerment_energy_fn} \
                        --TEACHER_BATCH_SIZE=${teacher_batch_size}"
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
echo "Total number of runs submitted: $run_count"
                        