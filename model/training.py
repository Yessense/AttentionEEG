import sys
from argparse import ArgumentParser

from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader

sys.path.append('..')
sys.path.append('.')
from dataset.dataset import DatasetCreator, Physionet
from model.AttentionEEG import AttentionEEG

# --------------------------------------------------
# -- Arguments
# --------------------------------------------------

parser = ArgumentParser()

# add program level args
dataset_parser = parser.add_argument_group('Dataset')
dataset_parser.add_argument("--dataset_path", type=str,
                            default="/home/yessense/PycharmProjects/eeg_project/data_physionet")

experiment_parser = parser.add_argument_group('Experiment')
experiment_parser.add_argument("--shift", type=int, default=128)
experiment_parser.add_argument("--dt", type=int, default=256)

experiment_parser.add_argument("--batch_size", type=int, default=64)
experiment_parser.add_argument("--train_test_split_max", type=int, default=4)

parser = AttentionEEG.add_model_specific_args(parent_parser=parser)
parser = pl.Trainer.add_argparse_args(parser)
args = parser.parse_args()

# --------------------------------------------------
# -- Dataloaders
# --------------------------------------------------
very_good_persons = [7, 69, 60, 27, 86, 93, 52, 83, 75, 2, 98, 55, 85, 72, 29, 62, 44]
good_persons = [103, 33, 89, 1, 73, 25, 42, 57, 8, 61, 56, 78, 108, 30, 34, 94, 41, 12]
mediocre_persons = [101, 32, 68, 35, 20, 74, 90, 19, 46, 26, 104, 58, 4]

train, test = train_test_split(list(range(1, args.train_test_split_max)), test_size=0.3, random_state=42)
# train = [i for i in train if i in very_good_persons + good_persons]
args.n_persons = 109
# Train data
dataset_creator = DatasetCreator(args.dataset_path, dt=args.dt,
                                 val_exp_numbers=[2, 5])

train_dataset = Physionet(*dataset_creator.create_dataset(train, args.shift))

train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

# Validation data
validation_dataset = Physionet(*dataset_creator.create_dataset(train, args.shift, validation=True))
validation_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, drop_last=True)
#
# Test data
test_dataset = Physionet(*dataset_creator.create_dataset(test, args.shift))  # args.shift
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size)

# --------------------------------------------------
# -- Trainer
# --------------------------------------------------

dict_args = vars(args)
classifier = AttentionEEG(**dict_args)

wandb_logger = WandbLogger(project='eeg_attention', log_model=False)

monitor = 'Val Loss/dataloader_idx_0'
profiler = None

if args.gpus is not None:
    gpus = [args.gpus]
else:
    gpus = None

# checkpoint
save_top_k = 2
checkpoint_callback = ModelCheckpoint(monitor=monitor, save_top_k=save_top_k)

trainer = pl.Trainer(gpus=gpus,
                     max_epochs=args.max_epochs,
                     logger=wandb_logger,
                     profiler=profiler,
                     log_every_n_steps=20)

logger = wandb_logger
trainer.fit(model=classifier,
            train_dataloaders=train_dataloader,
            val_dataloaders=[validation_dataloader, test_dataloader])

# trainer.test(model=classifier,
#              dataloaders=test_dataloader)
