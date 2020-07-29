from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load? set to latest to use latest cached model')
        self.parser.add_argument('--save_dir', type=str, default='./results', help='where to save the output video')
        self.parser.add_argument('--testName', type=str, default='BiFVS', help='where to save the output video')
        self.parser.add_argument('--task', type=str, required=False, help='evaluated task')
        self.parser.add_argument('--redo', action="store_true", help='redo evaluation')
        self.parser.add_argument('--nf', type=int, default=32, help='#Channels in conv layer')
        self.parser.add_argument('--norm', type=str, default='IN', choices=["BN", "IN", "none"], help='normalization layer')
        self.parser.add_argument('--w_fp2', type=float, default=0.5, help='weight for mix')
        self.parser.add_argument('--w_fl', type=float, default=0.5, help='weight for mix')
        self.isTrain = False
