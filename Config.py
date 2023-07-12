import yaml
import os

class train:
    def __init__(self):
        self.train_dir: str
        self.valid_dir: str
        self.use_valid_data: bool
        self.output_dir: str 
        self.train_split: float
        self.feat_idx: list
        self.coord_idx: list
        self.add_range: bool
        self.label_idx: list
        self.normalize: bool
        self.binary: bool
        self.device: str
        self.batch_size: int
        self.epochs: int
        self.init_lr: float
        self.output_classes: int
        self.epoch_timeout: int
        self.threshold_method: str
        self.termination_criteria: str
        self.compute_weights: bool

class test:
    def __init__(self):
        self.test_dir: str
        self.device: str
        self.batch_size: int
        self.save_pred_clouds: bool
        



class Config():

    def __init__(self, _root_dir = ''):
        with open(_root_dir) as file:
            self.config = yaml.safe_load(file)

        self.config_path = _root_dir
        # ---------------------------------------------------------------------#
        # TRAIN CONFIGURATION
        self.train = train()
        self.train.train_dir = self.config["train"]["TRAIN_DIR"] #type: str
        self.train.valid_dir = self.config["train"]["VALID_DIR"] #type: str
        self.train.use_valid_data = self.config["train"]["USE_VALID_DATA"] #type: bool
        self.train.output_dir = self.config["train"]["OUTPUT_DIR"] #type: str
        self.train.train_split = self.config["train"]["TRAIN_SPLIT"] #type: float
        self.train.coord_idx = self.config["train"]["COORDS"] #type: list
        self.train.feat_idx = self.config["train"]["FEATURES"] #type: list
        self.train.label_idx = self.config["train"]["LABELS"] #type: list
        self.train.normalize = self.config["train"]["NORMALIZE"] #type: bool
        self.train.binary = self.config["train"]["BINARY"] #type: bool
        self.train.device = self.config["train"]["DEVICE"] #type: str
        self.train.batch_size = self.config["train"]["BATCH_SIZE"] #type: int
        self.train.epochs = self.config["train"]["EPOCHS"] #type: int
        self.train.init_lr = self.config["train"]["LR"] #type: float
        self.train.output_classes = self.config["train"]["OUTPUT_CLASSES"] #type: int
        self.train.threshold_method = self.config["train"]["THRESHOLD_METHOD"] #type: str
        self.train.termination_criteria = self.config["train"]["TERMINATION_CRITERIA"] #type: str
        self.train.epoch_timeout = self.config["train"]["EPOCH_TIMEOUT"] #type: int
        self.train.compute_weights = self.config["train"]["COMPUTE_WEIGHTS"] #type: bool
        
        # ---------------------------------------------------------------------#
        # TEST CONFIGURATION
        self.test = test()
        self.test.test_dir = self.config["test"]["TEST_DIR"] #type: str
        self.test.device = self.config["test"]["DEVICE"] #type: str
        self.test.batch_size = self.config["test"]["BATCH_SIZE"] #type: int
        self.test.save_pred_clouds = self.config["test"]["SAVE_PRED_CLOUDS"] #type: bool