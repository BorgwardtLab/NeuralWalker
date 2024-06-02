from lg_model import Polynormer

def parse_method(args, n, c, d, device):
    model = Polynormer(d, args.hidden_channels, c, local_layers=args.local_layers,
            global_layers=args.global_layers, in_dropout=args.in_dropout, dropout=args.dropout,
            global_dropout=args.global_dropout, heads=args.num_heads, beta=args.beta, pre_ln=args.pre_ln,
            post_bn=args.post_bn, local_attn=args.local_attn,
            sequence_layer_type=args.seq_layer_type, window_size=args.window_size,
            d_conv=args.d_conv, walk_encoder_dropout=args.walk_encoder_dropout, use_edge_proj=args.use_edge_proj).to(device)
    return model


def parser_add_main_args(parser):
    # dataset and evaluation
    parser.add_argument('--dataset', type=str, default='ogbn-arxiv')
    parser.add_argument('--data_dir', type=str, default='./data/')
    parser.add_argument('--device', type=int, default=0,
                        help='which gpu to use if any (default: 0)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--local_epochs', type=int, default=1000,
                        help='warmup epochs for local attention')
    parser.add_argument('--global_epochs', type=int, default=0,
                        help='epochs for local-to-global attention')
    parser.add_argument('--batch_size', type=int, default=100000,
                        help='batch size for mini-batch training')
    parser.add_argument('--runs', type=int, default=1,
                        help='number of distinct runs')
    parser.add_argument('--metric', type=str, default='acc', choices=['acc', 'rocauc'],
                        help='evaluation metric')

    # model
    parser.add_argument('--method', type=str, default='poly')
    parser.add_argument('--hidden_channels', type=int, default=256)
    parser.add_argument('--local_layers', type=int, default=7,
                        help='number of layers for local attention')
    parser.add_argument('--global_layers', type=int, default=2,
                        help='number of layers for global attention')
    parser.add_argument('--num_heads', type=int, default=1,
                        help='number of heads for attention')
    parser.add_argument('--beta', type=float, default=0.9,
                        help='Polynormer beta initialization')
    parser.add_argument('--pre_ln', action='store_true')
    parser.add_argument('--post_bn', action='store_true')
    parser.add_argument('--local_attn', action='store_true')

    # NeuralWalker
    parser.add_argument('--length', type=int, default=500)
    parser.add_argument('--sample_rate', type=float, default=0.01)
    parser.add_argument('--test_sample_rate', type=float, default=0.1)
    parser.add_argument('--window_size', type=int, default=8)
    parser.add_argument('--seq_layer_type', type=str, default='mamba')
    parser.add_argument('--d_conv', type=int, default=4)
    parser.add_argument('--walk_encoder_dropout', type=float, default=0.1)
    parser.add_argument('--use_edge_proj', action='store_true')
    parser.add_argument('--test_runs', type=int, default=10)

    # training
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--in_dropout', type=float, default=0.15)
    parser.add_argument('--dropout', type=float, default=0.5)
    parser.add_argument('--global_dropout', type=float, default=None)

    # display and utility
    parser.add_argument('--display_step', type=int,
                        default=1, help='how often to print')
    parser.add_argument('--eval_step', type=int,
                        default=1, help='how often to evaluate')
    parser.add_argument('--eval_epoch', type=int,
                        default=-1, help='when to evaluate')
    parser.add_argument('--save_model', action='store_true', help='whether to save model')
    parser.add_argument('--model_dir', type=str, default='./model/', help='where to save model')
    parser.add_argument('--save_result', action='store_true', help='whether to save result')

