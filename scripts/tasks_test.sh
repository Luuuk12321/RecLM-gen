#!/bin/bash

#MODEL_NAME="snap/Llama-2-7b-hf-chat/"
#MODEL_NAME="snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/"
#MODEL_NAME="snap/ICR_Steam_Title64_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch40/"
#MODEL_NAME='snap/ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_SubMovie_Title64T_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch37/Total_train_LM-True_VM-False_NR-8_SN-2_Q-False_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-False_VFC-0.1_KLT-0.05_LRP-2.0_GAS-4_LB-1_ND-False/RLHF_Step1900/'
#MODEL_NAME='snap/ICR_Steam_Title_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/RLHF_ICR_Steam_Title_0_Llama7bChat_LCT_E40_CCR2_SCG2-0.5_IDX/SFT_Epoch40/Total_train_LM-True_VM-False_NR-8_SN-2_Q-False_T5_FG-True_LR-5e-06_LDO-0.0_WD-0.0_KLC-0.3_EW-0.01_RS-False_RW-True_VFC-0.1_KLT-0.05_LRP-2.0_GAS-4_LB-1_ND-False/RLHF_Step3900/'
MODEL_NAME=$1
PORT=$2

CHECK=$(echo "$MODEL_NAME" | grep "Steam")
if [ "$CHECK" != "" ]; then
    ITEM_INDEX='title'
    DATASET='steam'
#    PORT=13580
else
    ITEM_INDEX='title64_t'
    DATASET='sub_movie'
#    PORT=13579
fi

python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestSeqRec --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestSeqRec --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 5 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestSeqRanking --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 5 --candidate_num 10 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestSeqRanking --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 3 --candidate_num 10 --teacher_port ${PORT}
python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFT+TestPersonalControlRec --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFT+TestPersonalControlRec --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 5 --teacher_port ${PORT}
python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFT-TestPersonalControlRec --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFT-TestPersonalControlRec --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 5 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateLP_30 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateLP_50 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateLP_70 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateLP1_20 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateEP_30 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
#python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateEP_50 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}

if [ "$CHECK" != "" ]; then
    python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateMP_30 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
#    python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateMP_40 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
else
#    python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateMP_20 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
    python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestPersonalCategoryRateMP_30 --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
fi
python task_test.py --data_path data/dataset/${DATASET}/ --item_index ${ITEM_INDEX} --SFT_test_task SFTTestItemCount --model_name ${MODEL_NAME} --llama2_chat_template --idx --topk 10 --teacher_port ${PORT}
