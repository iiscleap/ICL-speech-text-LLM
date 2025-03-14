#!/bin/bash

# # Training Job ##


# model_type="salmonn"
# # # model_type="qwen2"

# dataset_type="voxceleb" 
# # #hvb,voxceleb,voxceleb_swap, hvb_greek,hvb_swap

# # # input_mode="speech_and_text"
# # input_mode="speech_only"
# input_mode="text_only"

# fewshot_mode="text"
# # # fewshot_mode="speech"

# # resume_from_checkpoint="./results/trained_models/INTERSPEECH/ft_15e8b_salmonn_speech_text_hvb_swap_/final_model.pt"
# resume_from_checkpoint=""

# if [ "$model_type" == "salmonn" ]; then
#     export CONDA_ENV="salmon"
# elif [ "$model_type" == "qwen2" ]; then
#     export CONDA_ENV="qwen2"
# else
#     echo "Invalid model type. Please specify 'salmonn' or 'qwen2'"
#     exit 1
# fi

# echo "Set conda environment to: $CONDA_ENV"

# source /home/share/anaconda3/etc/profile.d/conda.sh  
# conda deactivate
# conda activate $CONDA_ENV


# # Create more descriptive RUN_NAME
# input_mode_short="${input_mode/_only/}"  # Convert speech_only -> speech
# RUN_NAME="ft_20e8b_${model_type}_${input_mode_short}_${fewshot_mode}_${dataset_type}"


# SCRIPT_PATH="./scripts_new/finetune_llamma2_Salmon_final.py"


# rm -f "./results/logs/train/log_${RUN_NAME}.log" 

# qsub -q med.q -V -cwd \
#     -l hostname=compute-0-9 \
#     -o "./results/logs/train/log_${RUN_NAME}.log" \
#     -j y \
#     -m ae \
#     -M neeraj13788@gmail.com \
#     -v "CUDA_VISIBLE_DEVICES=1,RUN_NAME=${RUN_NAME},SCRIPT_PATH=${SCRIPT_PATH},input_mode=${input_mode},dataset_type=${dataset_type},fewshot_mode=${fewshot_mode},model_type=${model_type},resume_from_checkpoint=${resume_from_checkpoint}" \
#     -S /bin/bash ./scripts_new/train_sentiment.sh 



# # # # # ####
# # # # ##### Inference ######


# MODEL_PATH="./results/trained_models/ft_20e8b_salmonn_speech_text_hvb_greek/final_model.pt"
# MODEL_PATH="./results/trained_models/ft_15e8b_salmonn_speech_text_hvb_swap/final_model.pt"
# MODEL_PATH="./results/trained_models/ft_20e8b_salmonn_speech_text_hvb_swap_/checkpoints/epoch_10_loss_0.0230/model.pt"
# MODEL_PATH="/data2/neeraja/neeraja/code/SALMONN/results/trained_models/ft_20e8b_salmonn_speech_text_hvb_swap/final_model.pt"

# MODEL_PATH="./results/trained_models/ft_15e8b_salmonn_speech_text_hvb/final_model_.pt"


# MODEL_PATH="./results/trained_models/finetune_llama2_salmon_speech_15e8b_Q_voxceleb_swap/final_model.pt"
# MODEL_PATH="./results/trained_models/INTERSPEECH/ft_15e8b_salmonn_speech_text_voxceleb_swap_symbol/final_model.pt"
# MODEL_PATH="./results/trained_models/ft_15e8b_salmonn_speech_text_voxceleb_swap_new/final_model.pt"
# MODEL_PATH="./results/trained_models/finetune_llama2_salmon_speech_text_15e1b_Q_voxceleb/final_model.pt"

# MODEL_PATH="./results/trained_models/ft_15e8b_qwen2_speech_text_voxceleb_swap/final_model.pt"
# MODEL_PATH="./results/trained_models/ft_15e8b_qwen2_speech_text_voxceleb_swap_new/final_model.pt"
# MODEL_PATH="./results/trained_models/INTERSPEECH/ft_15e8b_qwen2_speech_text_voxceleb/final_model.pt"
# MODEL_PATH="./results/trained_models/ft_20e8b_qwen2_speech_text_voxceleb/checkpoints/epoch_10_loss_0.0000/model.pt"

# MODEL_PATH=""

MODEL_PATH="/data2/neeraja/neeraja/code/SALMONN/results/trained_models/INTERSPEECH/finetune_llama2_salmon_speech_15e8b_Q_voxceleb_swap/final_model.pt"
######################################################

TAG=${TAG:-"old"} 

# Set default RUN_NAME if MODEL_PATH is empty
if [ -z "$MODEL_PATH" ]; then
    RUN_NAME="default"
else
    # First try to extract name after INTERSPEECH/, then fallback to trained_models/
    RUN_NAME=$(echo "$MODEL_PATH" | sed -n 's/.*INTERSPEECH\/\([^/]*\).*/\1/p')
    if [ -z "$RUN_NAME" ]; then
        RUN_NAME=$(echo "$MODEL_PATH" | sed -n 's/.*trained_models\/\([^/]*\).*/\1/p')
    fi
    RUN_NAME=$(echo "$RUN_NAME" | sed 's/speech/sp/g; s/text/txt/g; s/salmonn/sal/g; s/qwen2/qw/g; s/voxceleb/vox/g')
fi

if [ ! -z "$TAG" ]; then
    RUN_NAME="${RUN_NAME}_${TAG}"
fi

# Set default values if MODEL_PATH is empty
if [ -z "$MODEL_PATH" ]; then
    model_type="salmonn"
    conda_env="salmon"
# If MODEL_PATH exists, determine model type from path
elif [[ $MODEL_PATH == *"qwen2"* ]]; then
    model_type="qwen2"
    conda_env="qwen2"
elif [[ $MODEL_PATH == *"salmonn"* ]] || [[ $MODEL_PATH == *"salmon"* ]]; then
    model_type="salmonn"
    conda_env="salmon"
else
    echo "Error: Cannot determine model type from model path: $MODEL_PATH"
    exit 1
fi
#model_type="qwen2"
#conda_env="qwen2"
echo "Detected model type: $model_type"
echo "Using conda environment: $conda_env"
echo "RUN_NAME: $RUN_NAME"

source /home/share/anaconda3/etc/profile.d/conda.sh  
conda deactivate
conda activate $conda_env


SCRIPT_PATH="/data2/neeraja/neeraja/code/ICL/ZscriptsINTERSPEECH/inference_llama2_salmon_final.py"
TODAY=$(date +"%Y-%m-%d")

mkdir -p "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}"
mkdir -p "/data2/neeraja/neeraja/results/model_ICL/metrics/${TODAY}"

########################################################

num_examples=1

# dataset_type="hvb"
# dataset_type="voxceleb"
dataset_type="voxpopuli"

input_mode="speech_only"
# input_mode="text_only"

fewshot_mode="text"
# fewshot_mode="speech"

rm -f "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}_${dataset_type}_${input_mode}_${fewshot_mode}_${num_examples}shots.log"  

qsub -q long.q -V -cwd \
    -l hostname=compute-0-9 \
    -l h_rt=72:00:00 \
    -o "/data2/neeraja/neeraja/results/model_ICL/logs/test/${TODAY}/${RUN_NAME}_${dataset_type}_${input_mode}_${fewshot_mode}_${num_examples}shots.log" \
    -j y \
    -v CUDA_VISIBLE_DEVICES=1,\
TODAY=${TODAY},\
PYTHONUNBUFFERED=1,\
RUN_NAME=${RUN_NAME},\
num_examples=${num_examples},\
MODEL_PATH=${MODEL_PATH},\
SCRIPT_PATH=${SCRIPT_PATH},\
input_mode=${input_mode},\
dataset_type=${dataset_type},\
fewshot_mode=${fewshot_mode},\
model_type=${model_type} \
    -S /bin/bash /data2/neeraja/neeraja/code/ICL/ZscriptsINTERSPEECH/inference_sentiment.sh







# qsub -q med.q -V -cwd \
#     -l hostname=compute-0-5 \
#     -hold_jid 133884  \
#     -o "./results/logs/test/${TODAY}/${RUN_NAME}_${dataset_type}_${input_mode}_${fewshot_mode}_${num_examples}shots.log" \
#     -j y \
#     -v CUDA_VISIBLE_DEVICES=1,\
# TODAY=${TODAY},\
# PYTHONUNBUFFERED=1,\
# RUN_NAME=${RUN_NAME},\
# num_examples=${num_examples},\
# MODEL_PATH=${MODEL_PATH},\
# SCRIPT_PATH=${SCRIPT_PATH},\
# input_mode=${input_mode},\
# dataset_type=${dataset_type},\
# fewshot_mode=${fewshot_mode},\
# model_type=${model_type} \
#     -S /bin/bash ./scripts_new/inference_sentiment.sh










