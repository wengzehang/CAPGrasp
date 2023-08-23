from .base_options import BaseOptions


class TestOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model'
        )
        self.is_train = False
        self.validate = False


class ValidationOptions(BaseOptions):
    def initialize(self):
        BaseOptions.initialize(self)
        self.parser.add_argument(
            '--which_epoch',
            type=str,
            default='latest',
            help='which epoch to load? set to latest to use latest cached model'
        )
        self.parser.add_argument(
            '--model_folder',
            type=str,
            required=True,
            help='Path to where the model is stored'
        )

        self.is_train = False
        self.validate = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt, _ = self.parser.parse_known_args()
        self.load_opt()
        self.opt.validate = self.validate
        return self.opt
