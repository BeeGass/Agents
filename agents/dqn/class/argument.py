def add_arguments(parser):
    '''
    Add your arguments here if needed. The TA will run test.py to load
    your default arguments.

    For example:
        parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
        parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    '''
    parser.add_argument('--batch_size', type=int, default=32, help='batch size for training')
    parser.add_argument('--learning_rate', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--weight_decay', type=int, default=32, help='batch size for training')
    parser.add_argument('--replay_buffer_size', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--gamma', type=int, default=32, help='batch size for training')
    parser.add_argument('--delta', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--target_frequency', type=int, default=32, help='batch size for training')
    parser.add_argument('--p_end', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--epsilon_decay_rate', type=int, default=32, help='batch size for training')
    parser.add_argument('--max_episodes', type=float, default=0.01, help='learning rate for training')
    parser.add_argument('--max_steps', type=int, default=32, help='batch size for training')
    return parser
