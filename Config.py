import yaml
import os
import torch

class train:
    def __init__(self):
        
        # DATA PATS
        self.train_dir: str
        self.valid_dir: str

        self.use_valid_data: bool

        # HYPERPARAMETERS        
        self.epochs: int
        self.batch_size: int
        self.init_lr: float
        self.lr_decay: float
        self.train_split: float

        # MODEL
        self.coord_idx: list
        self.feat_idx: list
        self.label_idx: list
        self.output_classes: int
        self.activation_fn: torch.nn.Module
        self.voxel_size: float

        # DATA STUFF
        self.normalize: bool
        self.binary: bool
        self.add_range: bool
        self.compute_weights: bool

        # TRAINING
        self.device: torch.device
        self.optimizer: str
        self.loss_fn: torch.nn.Module
        self.lr_scheduler: torch.optim.lr_scheduler

        self.threshold_method: str
        self.termination_criteria: str
        self.epoch_timeout: int

        self.output_dir: str 

    def set_device(self, _device):
        if torch.cuda.is_available():
            self.device = torch.device(_device)
        else:
            self.device = torch.device("cpu")

class test:
    def __init__(self):
        # DATA PATS
        self.test_dir: str

        # HYPERPARAMETERS
        self.batch_size: int

        # TEST
        self.device: torch.device
        self.save_pred_clouds: bool
        
    def set_device(self, _device):
        if torch.cuda.is_available():
            self.device = torch.device(_device)
        else:
            self.device = torch.device("cpu")

class Config():

    def __init__(self, _root_dir = ''):
        with open(_root_dir) as file:
            self.config = yaml.safe_load(file)

        self.config_path = _root_dir
        # ---------------------------------------------------------------------#
        # TRAIN CONFIGURATION
        self.train = train()
        
        # DATA PATHS
        self.train.train_dir = self.config["train"]["TRAIN_DIR"] 
        self.train.valid_dir = self.config["train"]["VALID_DIR"] 

        self.train.use_valid_data = self.config["train"]["USE_VALID_DATA"] 

        # HYPERPARAMETERS
        self.train.epochs =     self.config["train"]["EPOCHS"] 
        self.train.batch_size = self.config["train"]["BATCH_SIZE"]
        self.train.init_lr =    self.config["train"]["INIT_LR"]
        self.train.lr_decay =   self.config["train"]["LR_DECAY"]
        self.train.train_split= self.config["train"]["TRAIN_SPLIT"]

        # MODEL
        self.train.coord_idx =  self.config["train"]["COORD_IDX"]
        self.train.feat_idx =   self.config["train"]["FEAT_IDX"]
        self.train.label_idx =  self.config["train"]["LABEL_IDX"]
        self.train.output_classes = self.config["train"]["OUTPUT_CLASSES"]
        self.set_activation_fn(self.config["train"]["ACTIVATION_FUNCTION"].__str__())
        self.train.voxel_size = self.config["train"]["VOXEL_SIZE"]

        # DATA STUFF
        self.train.normalize =  self.config["train"]["NORMALIZE"]
        self.train.binary =     self.config["train"]["BINARY"]
        self.train.add_range =  self.config["train"]["ADD_RANGE"]
        self.train.compute_weights = self.config["train"]["COMPUTE_WEIGHTS"]

        # TRAINING
        self.train.set_device(self.config["train"]["DEVICE"].__str__())
        self.train.optimizer =  self.config["train"]["OPTIMIZER"]
        self.set_loss_fn( self.config["train"]["LOSS_FN"].__str__() )
        self.train.lr_scheduler = self.config["train"]["LR_SCHEDULER"]

        self.train.threshold_method =   self.config["train"]["THRESHOLD_METHOD"]
        self.train.termination_criteria = self.config["train"]["TERMINATION_CRITERIA"]
        self.train.epoch_timeout =      self.config["train"]["EPOCH_TIMEOUT"]

        self.train.output_dir = self.config["train"]["OUTPUT_DIR"]
        
        # ---------------------------------------------------------------------#
        # TEST CONFIGURATION
        self.test = test()

        # DATA PATHS
        self.test.test_dir = self.config["test"]["TEST_DIR"]
        
        # HYPERPARAMETERS
        self.test.batch_size = self.config["test"]["BATCH_SIZE"]

        # TEST
        self.test.set_device(self.config["test"]["DEVICE"].__str__())
        self.test.save_pred_clouds = self.config["test"]["SAVE_PRED_CLOUDS"]


    def set_activation_fn(self, _fn):
        if _fn == 'relu':
            self.train.activation_fn = torch.nn.ReLU()
        elif _fn == 'sigmoid':
            self.train.activation_fn = torch.nn.Sigmoid()
        elif _fn == 'softmax':
            self.train.activation_fn = torch.nn.Softmax()


    def set_loss_fn(self, _loss_fn):
        if _loss_fn == 'celoss':
            self.train.loss_fn = torch.nn.CrossEntropyLoss()
        elif _loss_fn == 'bceloss':
            self.train.loss_fn = torch.nn.BCELoss()
    
    # def set_lr_scheduler(self, _scheduler):
    #     if _scheduler == "step":
    #         self.train.lr_scheduler = torch.optim.lr_scheduler.StepLR(self.train.optimizer, step_size=30, gamma=0.1)
    #     elif _scheduler == "plateau":
    #         self.train.lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.train.optimizer, mode='min', factor=0.1, patience=10, threshold=0.0001, threshold_mode='rel', cooldown=0, min_lr=0, eps=1e-08, verbose=False)
    #     else:   
    #         self.train.lr_scheduler = None



                