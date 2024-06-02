## pokec
python main-batch.py --dataset pokec \
--hidden_channels 256 \
--local_epochs 2500 \
--batch_size 550000 \
--lr 0.0005 \
--runs 10 \
--local_layers 7 \
--global_layers 0 \
--in_drop 0.0 \
--dropout 0.2 \
--weight_decay 0.0 \
--post_bn \
--eval_step 9 \
--eval_epoch 2000 \
--device 0 \
--sample_rate 0.001 \
--test_sample_rate 0.001 \
--length 500 \
--walk_encoder_dropout 0.1 \
--seq_layer_type conv \
--d_conv 9 \
--save_model \
--save_result

## ogbn-products
python main-batch.py --dataset ogbn-products \
--local_attn \
--hidden_channels 32 \
--num_heads 8 \
--local_epochs 1500 \
--lr 0.0005 \
--batch_size 100000 \
--runs 10 \
--local_layers 10 \
--global_layers 0 \
--pre_ln \
--post_bn \
--in_drop 0.2 \
--weight_decay 0.0 \
--eval_step 9 \
--eval_epoch 1000 \
--sample_rate 0.001 \
--test_sample_rate 0.001 \
--length 500 \
--walk_encoder_dropout 0.2 \
--seq_layer_type conv \
--d_conv 9 \
--save_model \
--save_result