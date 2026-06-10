#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="teacher_single_student_bounded_variance_in_policy"
ANNEAL_GOAL_REWARD_WEIGHT="--no-ANNEAL_GOAL_REWARD_WEIGHT"

ENV_NAMES=("ant_u_maze_single_goal")
TOTAL_TIMESTEPS_=(500000000)
LRS=(0.0003)
SEEDS=(30)

BOUND_STUDENT_VARIANCE="--no-BOUND_STUDENT_VARIANCE"
BOUND_TEACHER_VARIANCE="--no-BOUND_TEACHER_VARIANCE"
# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
TEACHER_PROBE_AGG=(concat)
TEACHER_REWARD_TYPE=(competence_lp)
STUDENT_GOAL_REWARD_TYPE=(sparse dense)
NUM_ENVSS=(256 1024)
TEACHER_LR=(0.001)
TEACHER_EPISODE_LENGTH=(4 8)
TEACHER_SAMPLE_EVERY_N_EPISODES=(1)
TEACHER_NUM_MINIBATCHES_=(1 2)
TEACHER_UPDATE_EPOCHS_=(1 2)
NUM_STEPS_=(10 64)
TEACHER_ENTROPY_COFFS=(0 0.05 0.005)
STUDENT_ENTROPY_COFFS=(0 0.005)
RWD_NORM_TYPE=(running_std batch_minmax)
GOAL_REWARD_WEIGHTS=(1 0.1 0.01 0.001)


run_count=0

for ENV_NAME in "${ENV_NAMES[@]}"; do
  for TOTAL_TIMESTEPS in "${TOTAL_TIMESTEPS_[@]}"; do
    for LR in "${LRS[@]}"; do
        for teacher_probe_agg in "${TEACHER_PROBE_AGG[@]}"; do
          for teacher_reward_type in "${TEACHER_REWARD_TYPE[@]}"; do
            for student_goal_reward_type in "${STUDENT_GOAL_REWARD_TYPE[@]}"; do
              for teacher_lr in "${TEACHER_LR[@]}"; do
                for teacher_episode_length in "${TEACHER_EPISODE_LENGTH[@]}"; do
                  for teacher_sample_every_n_episodes in "${TEACHER_SAMPLE_EVERY_N_EPISODES[@]}"; do
                    for SEED in "${SEEDS[@]}"; do
                      for teacher_num_minibatches in "${TEACHER_NUM_MINIBATCHES_[@]}"; do
                        for teacher_update_epochs in "${TEACHER_UPDATE_EPOCHS_[@]}"; do
                        for num_steps in "${NUM_STEPS_[@]}"; do
                        for teacher_entropy_coef in "${TEACHER_ENTROPY_COFFS[@]}"; do
                        for student_entropy_coef in "${STUDENT_ENTROPY_COFFS[@]}"; do
                        for num_envs in "${NUM_ENVSS[@]}"; do
                        for rwd_norm_type in "${RWD_NORM_TYPE[@]}"; do
                        for goal_reward_weight in "${GOAL_REWARD_WEIGHTS[@]}"; do
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_teacherlr${teacher_lr}_probeagg${teacher_probe_agg}_treward${teacher_reward_type}_sgoal${student_goal_reward_type}_teplen${teacher_episode_length}_sampleevery${teacher_sample_every_n_episodes}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --SEED=${SEED} \
                        ${ANNEAL_GOAL_REWARD_WEIGHT} \
                        ${BOUND_STUDENT_VARIANCE} \
                        ${BOUND_TEACHER_VARIANCE} \
                        --TEACHER_NUM_MINIBATCHES=${teacher_num_minibatches} \
                        --TEACHER_UPDATE_EPOCHS=${teacher_update_epochs} \
                        --GOAL_REWARD_WEIGHT=${goal_reward_weight} \
                        --NUM_STEPS=${num_steps} \
                        --GOAL_PENALTY_NORM_TYPE=${rwd_norm_type} \
                        --NUM_ENVS=${num_envs} \
                        --PROJECT=\"${WANDB_PROJECT_NAME}\" \
                        --TEACHER_PROBE_AGG=${teacher_probe_agg} \
                        --TEACHER_REWARD_TYPE=${teacher_reward_type} \
                        --STUDENT_GOAL_REWARD_TYPE=${student_goal_reward_type} \
                        --TEACHER_LR=${teacher_lr} \
                        --TEACHER_ENT_COEF=${teacher_entropy_coef} \
                        --ENT_COEF=${student_entropy_coef} \
                        --TEACHER_EPISODE_LENGTH=${teacher_episode_length} \
                        --TEACHER_SAMPLE_EVERY_N_EPISODES=${teacher_sample_every_n_episodes}"
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

echo "Total number of runs submitted: $run_count"
                        