python3 run_quac_roberta.py \
  --bert_model roberta-base \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file QuAC_data/train_cs_his.json \
  --predict_file QuAC_data/dev_cs_his.json \
  --train_batch_size 1 \
  --learning_rate 3e-5 \
  --num_train_epochs 5 \
  --max_seq_length 512 \
  --max_query_length 64 \
  --doc_stride 128 \
  --output_dir output_quac/ \
  --log_freq 1000 \
# --no_flow \
# --no_cuda \
