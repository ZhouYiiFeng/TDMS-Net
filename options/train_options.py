from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        # model
        self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--nf', type=int, default=32, help='#Channels in conv layer')
        self.parser.add_argument('--norm', type=str, default='IN', choices=["BN", "IN", "none"], help='normalization layer')

        # lr
        self.parser.add_argument('--lr_init', type=float, default=1e-4, help='initial learning Rate')
        self.parser.add_argument('--lr_offset', type=int, default=20, help='epoch to start learning rate drop [-1 = no drop]')
        self.parser.add_argument('--lr_step', type=int, default=20, help='step size (epoch) to drop learning rate')
        self.parser.add_argument('--lr_drop', type=float, default=0.5, help='learning rate drop ratio')
        self.parser.add_argument('--lr_min_m', type=float, default=0.1, help='minimal learning Rate multiplier (lr >= lr_init * lr_min)')

        # results
        self.parser.add_argument('--display_freq', type=int, default=100, help='frequency of showing training results on screen')
        self.parser.add_argument('--print_freq', type=int, default=100,help='frequency of showing training results on console')
        self.parser.add_argument('--save_latest_freq', type=int, default=1000,help='frequency of saving the latest results')
        self.parser.add_argument('--save_epoch_freq', type=int, default=1, help='frequency of saving checkpoints at the end of epochs')
        self.parser.add_argument('--no_html', action='store_true', help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')

        # for temporal
        self.parser.add_argument('--w_ST', type=float, default=100, help='weight for short-term temporal loss')
        self.parser.add_argument('--w_VGG', type=float, default=10, help='weight for VGG perceptual loss')
        self.parser.add_argument('--w_l1p2Loss', type=float, default=1, help='weight for VGG perceptual loss')
        # self.parser.add_argument('--w_STCyc', type=float, default=10, help='weight for long-term temporal loss')
        # self.parser.add_argument('--w_L1', type=float, default=100, help='weight for short-term temporal loss')
        self.parser.add_argument('--w_w', type=float, default=1, help='weight for mix')
        self.parser.add_argument('--w_fl', type=float, default=0.5, help='weight for mix')
        self.parser.add_argument('--w_cfl', type=float, default=1, help='weight for mix')
        self.parser.add_argument('--VGGLayers', type=str, default="4", help="VGG layers for perceptual loss, combinations of 1, 2, 3, 4")

        # optimizer
        self.parser.add_argument('--solver', type=str, default="ADAM", choices=["SGD", "ADAIM"], help="optimizer")
        self.parser.add_argument('--momentum', type=float, default=0.9, help='momentum for SGD')
        self.parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for ADAM')
        self.parser.add_argument('--beta2', type=float, default=0.999, help='beta2 for ADAM')
        self.parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')
        self.parser.add_argument('--loss', type=str, default="L1", help="optimizer [Options: SGD, ADAM]")

        # others
        self.parser.add_argument('--train_epoch_size', type=int, default=1000, help='train epoch size')
        self.parser.add_argument('--valid_epoch_size', type=int, default=100, help='valid epoch size')
        self.parser.add_argument('--epoch_max', type=int, default=150, help='max #epochs')
        self.parser.add_argument('--sample_frames', type=int, default=2, help='#frames for training')

        self.isTrain = True