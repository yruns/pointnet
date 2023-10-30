from utils.args import Args

class Args4Shapenet(Args):
    """
    超参数 for Shapenet
    """
    def __init__(self):
        super(Args4Shapenet, self).__init__()

        # Data settings
        self.data_dir = 'data/shapnetcorev0'
        self.checkpoint_path = 'checkpoints/'
        self.log_path = 'logs/'

        # Training settings
        self.seed = 2023
        self.batch_size = 64

        self.epochs = 300
        self.lr = 1e-3
        self.optimizer = "Adam"
        self.warmup_rate = 0.1
        self.weight_decay = 0.001
        self.adam_epsilon = 1e-8
        self.max_clip_norm = 1.0

        self.gradient_accumulation_steps = 1
        self.eval_interval = 2000
        self.log_interval = 50

        self.k = 50
        self.alpha = 0.3   # 正则化力度


        # Model settings
        self.points_num = 2500
        self.normal = True





if __name__ == '__main__':
    from loguru import logger

    args = Args4Shapenet()
    args.print_args(logger)


