from argparse import ArgumentParser
from configs.paths_config import model_paths


class TrainOptions:

    def __init__(self):
        self.parser = ArgumentParser()
        self.initialize()

    def initialize(self):
        self.parser.add_argument(
            '--exp_dir', type=str, help='Path to experiment output directory')
        self.parser.add_argument('--dataset_type', default='cub200_encode',
                                 type=str, help='Type of dataset/experiment to run')
        self.parser.add_argument(
            '--encoder_type', default='CLIPImageEmbedder', type=str, help='Which encoder to use')
        self.parser.add_argument(
            '--encoder_version', default='/data2/mhf/DXL/Lingxiao/Cache/huggingface/hub/models--openai--clip-vit-large-patch14', type=str, help='Which encoder version to use')
        self.parser.add_argument('--geometry', default=None,
                                 choices=['euclidean', 'hyperbolic', 'spherical'],
                                 help='Geometry used by the Stage II feature bottleneck')
        self.parser.add_argument('--hyperbolic', action='store_true',
                                 help='Deprecated: use --geometry hyperbolic')
        self.parser.add_argument('--input_nc', default=3, type=int,
                                 help='Number of input image channels to the psp encoder')
        self.parser.add_argument('--label_nc', default=0, type=int,
                                 help='Number of input label channels to the psp encoder')
        self.parser.add_argument('--feature_size', default=512, type=int,
                                 help='Dimension of latent code in hyperbolic space')
        self.parser.add_argument(
            '--output_size', default=1024, type=int, help='Output size of generator')
        self.parser.add_argument(
            '--transformer_layers', default=30, type=int, help='Number of transformer layers in the decoder')

        self.parser.add_argument(
            '--batch_size', default=4, type=int, help='Batch size for training')
        self.parser.add_argument('--test_batch_size', default=2,
                                 type=int, help='Batch size for testing and inference')
        self.parser.add_argument(
            '--workers', default=4, type=int, help='Number of train dataloader workers')
        self.parser.add_argument('--test_workers', default=2, type=int,
                                 help='Number of test/inference dataloader workers')

        self.parser.add_argument(
            '--train_all', action='store_true', help='Whether to train all model weights')
        self.parser.add_argument(
            '--learning_rate', default=0.0001, type=float, help='Optimizer learning rate')
        self.parser.add_argument(
            '--step_size', default=500, type=int, help='Step size for learning rate decay')
        self.parser.add_argument(
            '--gamma', default=0.5, type=float, help='Gamma for learning rate decay')
        self.parser.add_argument(
            '--optim_name', default='adamw', choices=['adamw', 'riemannian_adam'], type=str, help='Which optimizer to use')
        self.parser.add_argument('--train_decoder', default=False,
                                 type=bool, help='Whether to train the decoder model')
        self.parser.add_argument('--start_from_latent_avg', action='store_true',
                                 help='Whether to add average latent vector to generate codes from encoder.')
        self.parser.add_argument(
            '--learn_in_w', action='store_true', help='Whether to learn in w space instead of w+')

        self.parser.add_argument(
            "--c", default=1.0, type=float, help="Curvature of the Poincare ball (default: 0.1)",
        )
        self.parser.add_argument(
            '--hyperbolic_clip_radius_ratio', default=0.9, type=float,
            help='For hyperbolic Stage II, clip Poincare-ball feature and class-anchor Euclidean norms to this fraction of the ball radius. Set <= 0 to disable.')
        self.parser.add_argument(
            '--hyperbolic_tangent_radius_ratio', default=0.7, type=float,
            help='For hyperbolic Stage II, smoothly bound the Euclidean tangent feature before expmap0 so it maps to at most this Poincare radius fraction. Set <= 0 to disable.')
        self.parser.add_argument(
            '--disable_hyperbolic_input_norm', action='store_true',
            help='Disable the stateless LayerNorm applied before the hyperbolic expmap projection.')
        self.parser.add_argument(
            '--hyperbolic_anchor_init_std', default=1e-2, type=float,
            help='Tangent-space std used to initialize hyperbolic classifier anchor points before expmap0.')
        self.parser.add_argument(
            '--hyperbolic_normal_init_gain', default=5.0, type=float,
            help='Multiplier applied to hyperbolic classifier normal initialization.')
        self.parser.add_argument(
            '--riemannian_stabilize_steps', default=10, type=int,
            help='Projection/stabilization interval for Geoopt Riemannian optimizers. Set <= 0 to disable.')
        self.parser.add_argument(
            '--lpips_lambda', default=0.8, type=float, help='LPIPS loss multiplier factor')
        self.parser.add_argument(
            '--id_lambda', default=0, type=float, help='ID loss multiplier factor')
        self.parser.add_argument(
            '--l2_lambda', default=1.0, type=float, help='L2 loss multiplier factor')
        self.parser.add_argument(
            '--w_norm_lambda', default=0, type=float, help='W-norm loss multiplier factor')
        self.parser.add_argument('--lpips_lambda_crop', default=0, type=float,
                                 help='LPIPS loss multiplier factor for inner image region')
        self.parser.add_argument('--l2_lambda_crop', default=0, type=float,
                                 help='L2 loss multiplier factor for inner image region')
        self.parser.add_argument('--moco_lambda', default=0, type=float,
                                 help='Moco-based feature similarity loss multiplier factor')
        self.parser.add_argument('--contrastive_lambda', default=0.1, type=float,
                                 help='Un/Supervised Contrastive loss factor for hyperbolic feature learning')
        self.parser.add_argument('--geometry_lambda', default=None, type=float,
                                 help='Supervised classification loss factor for the selected Stage II geometry')
        self.parser.add_argument('--hyperbolic_lambda', default=0.1, type=float,
                                 help='Deprecated: use --geometry_lambda')
        self.parser.add_argument('--reverse_lambda', default=0.1, type=float,
                                 help='Project back to the original Euclidean space')
        self.parser.add_argument('--cosine_lambda', default=0.1, type=float,
                                 help='Cosine similarity loss for hyperbolic feature learning')
                                 

        self.parser.add_argument(
            '--stylegan_weights', default=None, type=str, help='Path to StyleGAN model weights')
        self.parser.add_argument('--stylegan_size', default=1024, type=int,
                                 help='size of pretrained StyleGAN Generator')
        self.parser.add_argument(
            '--checkpoint_path', default=None, type=str, help='Path to HAE model checkpoint')
        self.parser.add_argument(
            '--stage1_checkpoint_path', '--psp_checkpoint_path', default=None, type=str,
            dest='stage1_checkpoint_path',
            help='Path to the Stage 1 checkpoint. --psp_checkpoint_path is kept as a deprecated alias.')

        self.parser.add_argument(
            '--max_steps', default=5000, type=int, help='Maximum number of training steps')
        self.parser.add_argument('--image_interval', default=100, type=int,
                                 help='Interval for logging train images during training')
        self.parser.add_argument('--board_interval', default=50,
                                 type=int, help='Interval for logging metrics to tensorboard')
        self.parser.add_argument(
            '--val_interval', default=1000, type=int, help='Validation interval')
        self.parser.add_argument(
            '--save_interval', default=None, type=int, help='Model checkpoint interval')
        self.parser.add_argument('--early_stop_patience', default=20, type=int,
                                 help='Number of validation checks without improvement before stopping. Set <= 0 to disable.')
        self.parser.add_argument('--early_stop_min_delta', default=0.0001, type=float,
                                 help='Minimum validation loss improvement required to reset early stopping.')
        
        # Discriminator flags
        self.parser.add_argument(
            '--w_discriminator_lambda', default=0, type=float, help='Dw loss multiplier')
        self.parser.add_argument(
            '--w_discriminator_lr', default=2e-5, type=float, help='Dw learning rate')
        self.parser.add_argument(
            "--r1", type=float, default=10, help="weight of the r1 regularization")
        self.parser.add_argument("--d_reg_every", type=int, default=16,
                                 help="interval for applying r1 regularization")
        self.parser.add_argument('--use_w_pool', action='store_true',
                                 help='Whether to store a latnet codes pool for the discriminator\'s training')
        self.parser.add_argument("--w_pool_size", type=int, default=50,
                                 help="W\'s pool size, depends on --use_w_pool")

        # e4e specific
        self.parser.add_argument(
            '--delta_norm', type=int, default=2, help="norm type of the deltas")
        self.parser.add_argument(
            '--delta_norm_lambda', type=float, default=2e-4, help="lambda for delta norm loss")

        # Progressive training
        self.parser.add_argument('--progressive_steps', nargs='+', type=int, default=None,
                                 help="The training steps of training new deltas. steps[i] starts the delta_i training")
        self.parser.add_argument('--progressive_start', type=int, default=None,
                                 help="The training step to start training the deltas, overrides progressive_steps")
        self.parser.add_argument('--progressive_step_every', type=int, default=2_000,
                                 help="Amount of training steps for each progressive step")
        
        # Save additional training info to enable future training continuation from produced checkpoints
        self.parser.add_argument('--save_training_data', action='store_true',
                                 help='Save intermediate training data to resume training from the checkpoint')
        self.parser.add_argument(
            '--sub_exp_dir', default=None, type=str, help='Name of sub experiment directory')
        self.parser.add_argument('--keep_optimizer', action='store_true',
                                 help='Whether to continue from the checkpoint\'s optimizer')
        self.parser.add_argument('--resume_training_from_ckpt', default=None, type=str,
                                 help='Path to training checkpoint, works when --save_training_data was set to True')
        self.parser.add_argument('--update_param_list', nargs='+', type=str, default=None,
                                 help="Name of training parameters to update the loaded training checkpoint")

        # arguments for weights & biases support
        self.parser.add_argument('--use_wandb', action="store_true",
                                 help='Whether to use Weights & Biases to track experiment.')

        # arguments for CodeCarbon energy/emissions tracking
        self.parser.add_argument('--use_codecarbon', action='store_true',
                                 help='Whether to track energy/emissions with CodeCarbon on rank 0.')
        self.parser.add_argument('--codecarbon_project_name', default='hypdae_stage2', type=str,
                                 help='CodeCarbon project name.')
        self.parser.add_argument('--codecarbon_output_dir', default=None, type=str,
                                 help='Directory for CodeCarbon outputs. Defaults to exp_dir/logs/codecarbon.')
        self.parser.add_argument('--codecarbon_measure_power_secs', default=30, type=int,
                                 help='CodeCarbon power sampling interval in seconds.')
        self.parser.add_argument('--codecarbon_log_every_n_secs', default=300, type=int,
                                 help='How often to print CodeCarbon progress messages.')

        # arguments for super-resolution
        self.parser.add_argument('--resize_factors', type=str, default=None,
                                 help='For super-res, comma-separated resize factors to use for inference.')
        # multi-gpus
        self.parser.add_argument(
            "--world-size",
            default=-1,
            type=int,
            help="number of nodes for distributed training",
        )
        self.parser.add_argument(
            "--rank", default=-1, type=int, help="node rank for distributed training"
        )
        self.parser.add_argument(
            "--dist-url",
            default="tcp://224.66.41.62:23456",
            type=str,
            help="url used to set up distributed training",
        )
        self.parser.add_argument(
            "--dist-backend", default="nccl", type=str, help="distributed backend"
        )
        self.parser.add_argument(
            "--seed", default=None, type=int, help="seed for initializing training. "
        )
        self.parser.add_argument(
            "--gpu", default=None, type=int, help="GPU id to use.")
        self.parser.add_argument(
            "--ngpus-per-node", default=None, type=int, help="Number of GPUs to use.")
        self.parser.add_argument(
            "--multiprocessing-distributed",
            action="store_true",
            help="Use multi-processing distributed training to launch "
            "N processes per node, which has N GPUs. This is the "
            "fastest way to use PyTorch for either single node or "
            "multi node data parallel training",
        )
        

    def parse(self):
        opts = self.parser.parse_args()
        opts.psp_checkpoint_path = opts.stage1_checkpoint_path
        if opts.geometry is None:
            opts.geometry = 'hyperbolic' if opts.hyperbolic else 'euclidean'
        else:
            opts.hyperbolic = opts.geometry == 'hyperbolic'
        if opts.geometry_lambda is None:
            opts.geometry_lambda = opts.hyperbolic_lambda
        else:
            opts.hyperbolic_lambda = opts.geometry_lambda
        return opts
