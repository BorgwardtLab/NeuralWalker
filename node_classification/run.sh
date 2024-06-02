## All the hyperparameters are adapted from Polynormer
## roman-empire
python main.py --dataset roman-empire \
--hidden_channels 64 \
--local_epochs 2500 \
--global_epochs 0 \
--lr 0.0005 \
--runs 10 \
--local_layers 10 \
--global_layers 0 \
--weight_decay 0.0 \
--dropout 0.3 \
--global_dropout 0.5 \
--in_dropout 0.15 \
--num_heads 8 \
--save_model \
--beta 0.5 \
--sample_rate 0.01 \
--length 1000 \
--test_sample_rate 0.1 \
--walk_encoder_dropout 0.3 \
--seq_layer_type mamba \
--d_conv 4


## amazon-ratings
python main.py --dataset amazon-ratings \
--hidden_channels 256 \
--local_epochs 2700 \
--global_epochs 0 \
--lr 0.0005 \
--runs 10 \
--local_layers 10 \
--global_layers 0 \
--weight_decay 0.0 \
--dropout 0.3 \
--in_dropout 0.2 \
--num_heads 2 \
--save_model \
--sample_rate 0.01 \
--length 1000 \
--test_sample_rate 0.1 \
--walk_encoder_dropout 0.2 \
--seq_layer_type mamba \
--d_conv 4


## minesweeper
python main.py --dataset minesweeper \
--hidden_channels 64 \
--local_epochs 2000 \
--global_epochs 0 \
--lr 0.0005 \
--runs 10 \
--local_layers 10 \
--global_layers 0 \
--weight_decay 0.0 \
--dropout 0.3 \
--in_dropout 0.2 \
--num_heads 8 \
--metric rocauc \
--save_model \
--sample_rate 0.01 \
--length 1000 \
--test_sample_rate 0.1 \
--walk_encoder_dropout 0.3 \
--seq_layer_type mamba \
--d_conv 4


## tolokers
python main.py --dataset tolokers \
--hidden_channels 60 \
--local_epochs 1000 \
--global_epochs 0 \
--lr 0.001 --runs 10 \
--local_layers 7 \
--global_layers 0 \
--weight_decay 0.0 \
--dropout 0.5 \
--in_dropout 0.2 \
--num_heads 16 \
--metric rocauc \
--save_model \
--beta 0.1 \
--sample_rate 0.01 \
--length 1000 \
--test_sample_rate 0.1 \
--walk_encoder_dropout 0.1 \
--seq_layer_type mamba \
--d_conv 4


## questions
python main.py --dataset questions \
--hidden_channels 64 \
--local_epochs 1700 \
--global_epochs 0 \
--lr 5e-5 \
--runs 10 \
--local_layers 5 \
--global_layers 0 \
--weight_decay 0.0 \
--dropout 0.2 \
--global_dropout 0.5 \
--num_heads 8 \
--metric rocauc \
--in_dropout 0.15 \
--save_model \
--beta 0.4 \
--pre_ln \
--sample_rate 0.01 \
--length 1000 \
--test_sample_rate 0.05 \
--walk_encoder_dropout 0.2 \
--seq_layer_type mamba \
--d_conv 4
