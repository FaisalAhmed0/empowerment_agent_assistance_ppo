#!/bin/bash

# Define common parameters (fixed values)
WANDB_PROJECT_NAME="purejaxrl_continuous_control_with_goals_from_mlp_teacher_emp_reward_annleaded_emp_lr"
ADD_GOAL_REWARD="--ADD_GOAL_REWARD"
CONDITION_ON_GOAL="--CONDITION_ON_GOAL"
USE_LEARNING_PROGRESS_REWARD="--no-USE_LEARNING_PROGRESS_REWARD"
USE_TEACHER_EMPOWERMENT_REWARD="--USE_TEACHER_EMPOWERMENT_REWARD"
ENV_NAMES=("ant_u_maze_single_goal")
TOTAL_TIMESTEPS_=(300000000)
LRS=(0.0003)
SEEDS=(30)
COMMENT="here_we_are_using_the_mlp_teacher_to_generate_the_goals_the_teacher_is_trained_with_different_goal_reward_coefficients_to_how_much_sensitivity_to_the_goal_reward_we_want_to_give_and_we_are_using_a_teacher_empowerment_reward"

# PPO teacher-specific sweep args from
# purejaxrl/ppo_continuous_action_custom_brax_with_teacher.py
NUM_ENVSS=(256)
NUM_STEPS_=(64)
STUDENT_ENTROPY_COFFS=(0)
GAE_LAMBDA=(0.8)
CLIP_EPS=(0.2)
MAX_GRAD_NORM=(1.0)
UPDATE_EPOCHSS=(4)
NUM_MINIBATCHES=(16)
NORMALIZE_ENVS=(--NORMALIZE_ENV)
HIDDEN_DIMS=(256)
GOAL_REWARD_COEF=(1)
### Teacher hyperparameters
TEACHER_ROLLOUT_BUFFER_SIZES=(1 10)
ABSOLUTE_LEARNING_PROGRESSS=(--no-ABSOLUTE_LEARNING_PROGRESS)
USE_DISTANCE_IN_COMPETENCES=(--no-USE_DISTANCE_IN_COMPETENCE --USE_DISTANCE_IN_COMPETENCE)
ENERGY_FUNCTIONS=(l2)
CONTRASTIVE_LOSS_TYPES=(fwd_infonce bwd_infonce sym_infonce)
run_count=0
EMPOWERMENT_NUM_MINIBATCHES=(128 256)
GAMMA_CLS=(0.99 0.9)
EMPOWERMENT_UPDATE_EPOCHSS=(1 2 4)

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
                        for energy_function in "${ENERGY_FUNCTIONS[@]}"; do
                        for contrastive_loss_type in "${CONTRASTIVE_LOSS_TYPES[@]}"; do
                        for empowerment_num_minibatches in "${EMPOWERMENT_NUM_MINIBATCHES[@]}"; do
                        for gamma_cls in "${GAMMA_CLS[@]}"; do
                        for empowerment_update_epochs in "${EMPOWERMENT_UPDATE_EPOCHSS[@]}"; do
                        for USE_DISTANCE_IN_COMPETENCE in "${USE_DISTANCE_IN_COMPETENCES[@]}"; do
                      RUN_NAME="${ENV_NAME}_steps${TOTAL_TIMESTEPS}_lr${LR}_entropy${student_entropy_coef}_num_envs${num_envs}_num_steps${num_steps}_gae_lambda${gae_lambda}_clip_eps${clip_eps}"
                      CMD="sbatch scripts/submit_job purejaxrl/ppo_continuous_action_custom_brax_with_teacher_emp_reward.py \
                        --ENV_NAME=${ENV_NAME} \
                        --TOTAL_TIMESTEPS=${TOTAL_TIMESTEPS} \
                        --LR=${LR} \
                        --HIDDEN_DIM=${hidden_dim} \
                        --NUM_MINIBATCHES=${num_minibatches} \
                        --GAMMA_CL=${gamma_cls} \
                        --EMPOWERMENT_UPDATE_EPOCHS=${empowerment_update_epochs} \
                        ${USE_LEARNING_PROGRESS_REWARD} \
                        ${USE_TEACHER_EMPOWERMENT_REWARD} \
                        --EMPOWERMENT_ENERGY_FN=${energy_function} \
                        --EMPOWERMENT_CONTRASTIVE_LOSS=${contrastive_loss_type} \
                        --EMPOWERMENT_NUM_MINIBATCHES=${empowerment_num_minibatches} \
                        --SEED=${SEED} \
                        ${ADD_GOAL_REWARD} \
                        ${USE_DISTANCE_IN_COMPETENCE} \
                        ${CONDITION_ON_GOAL} \
                        --NUM_STEPS=${num_steps} \
                        --GAE_LAMBDA=${gae_lambda} \
                        --TEACHER_ROLLOUT_BUFFER_SIZE=${teacher_rollout_buffer_size} \
                        --MAX_GRAD_NORM=${max_grad_norm} \
                        --UPDATE_EPOCHS=${update_epochs} \
                        --GOAL_REWARD_COEF=${goal_reward_coef} \
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
                        