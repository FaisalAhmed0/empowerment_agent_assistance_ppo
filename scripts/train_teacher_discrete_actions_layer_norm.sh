#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="discrete_actions_teacher_single_student_layer_norm"
ANNEAL_GOAL_REWARD_WEIGHT="--no-ANNEAL_GOAL_REWARD_WEIGHT"

ENV_NAMES=("ant_u_maze_single_goal")
TOTAL_TIMESTEPS_=(500000000)
LRS=(0.001)
SEEDS=(1)
NUM_EPOCHS_=(8)
CLIP_EPS_=(0.2)

BOUND_STUDENT_VARIANCE="--BOUND_STUDENT_VARIANCE"
BOUND_TEACHER_VARIANCE="--no-BOUND_TEACHER_VARIANCE"
# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
TEACHER_PROBE_AGG=(concat)
TEACHER_REWARD_TYPE=(competence_lp)
STUDENT_GOAL_REWARD_TYPE=(sparse)
NUM_ENVSS=(1024)
TEACHER_LR=(0.001)
HIDDEN_DIM=(256)
TEACHER_HIDDEN_DIM=(256)
TEACHER_EPISODE_LENGTH=(4)
TEACHER_SAMPLE_EVERY_N_EPISODES=(1)
TEACHER_NUM_MINIBATCHES_=(2)
TEACHER_UPDATE_EPOCHS_=(1)
NUM_STEPS_=(10)
TEACHER_ENTROPY_COFFS=(0 0.05)
STUDENT_ENTROPY_COFFS=(0)
GOAL_REWARD_WEIGHTS=(1)
USE_SEPARATE_VALUE_FUNCTIONS=("--USE_SEPARATE_STUDENT_VALUE_FUNCTIONS")
USE_ACTOR_PROBING_STATES=("--USE_ACTOR_PROBING_STATES")
USE_CRITIC_PROBING_STATES=("--USE_CRITIC_PROBING_STATES")
LAYER_NORM=("--LAYER_NORM" "--no-LAYER_NORM")


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
                        for hidden_dim in "${HIDDEN_DIM[@]}"; do
                        for teacher_hidden_dim in "${TEACHER_HIDDEN_DIM[@]}"; do
                        for num_epochs in "${NUM_EPOCHS_[@]}"; do
                        for goal_reward_weight in "${GOAL_REWARD_WEIGHTS[@]}"; do
                        for clip_eps in "${CLIP_EPS_[@]}"; do
                        for use_separate_value_functions in "${USE_SEPARATE_VALUE_FUNCTIONS[@]}"; do
                        for use_actor_probing_states in "${USE_ACTOR_PROBING_STATES[@]}"; do
                        for use_critic_probing_states in "${USE_CRITIC_PROBING_STATES[@]}"; do
                        for layer_norm in "${LAYER_NORM[@]}"; do
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_teacherlr${teacher_lr}_probeagg${teacher_probe_agg}_treward${teacher_reward_type}_sgoal${student_goal_reward_type}_teplen${teacher_episode_length}_sampleevery${teacher_sample_every_n_episodes}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_continuous_action_custom_brax_with_teacher_discrete.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --SEED=${SEED} \
                        ${ANNEAL_GOAL_REWARD_WEIGHT} \
                        ${BOUND_STUDENT_VARIANCE} \
                        ${BOUND_TEACHER_VARIANCE} \
                        ${use_separate_value_functions} \
                        ${use_actor_probing_states} \
                        ${use_critic_probing_states} \
                        --TEACHER_NUM_MINIBATCHES=${teacher_num_minibatches} \
                        --TEACHER_UPDATE_EPOCHS=${teacher_update_epochs} \
                        ${layer_norm} \
                        --NUM_STEPS=${num_steps} \
                        --CLIP_EPS=${clip_eps} \
                        --TEACHER_CLIP_EPS=${clip_eps} \
                        --GOAL_REWARD_WEIGHT=${goal_reward_weight} \
                        --NUM_ENVS=${num_envs} \
                        --PROJECT=\"${WANDB_PROJECT_NAME}\" \
                        --UPDATE_EPOCHS=${num_epochs} \
                        --HIDDEN_DIM=${hidden_dim} \
                        --TEACHER_HIDDEN_DIM=${teacher_hidden_dim} \
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
  done 
  done
  done
  done
  done
  done
  done

echo "Total number of runs submitted: $run_count"
                        