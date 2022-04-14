import argparse


def parse_opt():
    parser = argparse.ArgumentParser()
    # --------------------dynamic parameters setting-------------------- #
    parser.add_argument('--num_epochs', default=100, type=int,
                        help='Number of training epochs.')
    parser.add_argument('--lr_update', default=50, type=int,
                        help='Number of epochs to update the learning rate.')
    parser.add_argument('--batch_size', default=100, type=int,
                        help='Size of a training mini-batch.')
    parser.add_argument('--log_step', default=10, type=int,
                        help='Number of steps to print and record the log.')
    parser.add_argument('--logger_name', default='./runs/runX/log',
                        help='Path to save Tensorboard log.')
    parser.add_argument('--model_name', default='./runs/runX/checkpoint',
                        help='Path to save the model.')
    parser.add_argument('--gamma_1', default=1, type=float,
                        help='gamma_1 in msb loss.')
    parser.add_argument('--gamma_2', default=0.5, type=float,
                        help='gamma_2 in msb loss.')
    parser.add_argument('--alpha_1', default=0.4, type=float,
                        help='alpha_1 in global-level objective function.')
    parser.add_argument('--alpha_2', default=0.4, type=float,
                        help='alpha_2 in global-level objective function.')
    parser.add_argument('--beta_1', default=0.4, type=float,
                        help='beta_1 in local-level objective function.')
    parser.add_argument('--beta_2', default=0.4, type=float,
                        help='beta_2 in local-level objective function.')
    parser.add_argument('--data_path', default='./data/',
                        help='path to datasets.')
    parser.add_argument('--data_name', default='pascal_precomp',
                        help='{coco,f30k}_precomp')
    # --------------------fixed parameters setting-------------------- #
    parser.add_argument('--vocab_path', default='./vocab/',
                        help='Path to saved vocabulary json files.')
    parser.add_argument('--word_dim', default=300, type=int,
                        help='Dimensionality of the word embedding.')
    parser.add_argument('--embed_size', default=1024, type=int,
                        help='Dimensionality of the joint embedding.')
    parser.add_argument('--grad_clip', default=2., type=float,
                        help='Gradient clipping threshold.')
    parser.add_argument('--num_layers', default=1, type=int,
                        help='Number of GRU layers.')
    parser.add_argument('--cnn_dim', default=4096, type=int,
                        help='Dimensionality of the CNN embedding.')
    parser.add_argument('--bovw_dim', default=500, type=int,
                        help='Dimensionality of the boVW.')
    parser.add_argument('--bi_gru', default=True,
                        help='Use bidirectional GRU.')
    parser.add_argument('--raw_feature_norm', default="clipped_l2norm",
                        help='clipped_l2norm|l2norm|no_norm|softmax')
    parser.add_argument('--lambda_softmax', default=9., type=float,
                        help='Attention softmax temperature.')
    parser.add_argument('--Delta', default=0.1, type=float,
                        help='The constant in msb loss.')
    parser.add_argument('--gpuid', default=0, type=str,
                        help='gpuid')
    parser.add_argument('--learning_rate', default=.0002, type=float,
                        help='Initial learning rate.')
    args = parser.parse_args()
    return args
