import argparse
import os
from datetime import datetime

from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import LightningLoggerBase, TensorBoardLogger, CSVLogger

from pet_model import PET_Model

import warnings
warnings.filterwarnings("ignore")

#запуск модели
def main(hparams) -> None:

#     # Инициализируем модель
    model = PET_Model(hparams)
    
    #  Модуль для раннего останова
    
    early_stop_callback = EarlyStopping(
        monitor=hparams.monitor,
        min_delta = 0.01,
        patience = hparams.patience,
        verbose = True,
        mode = hparams.metric_mode,
    )

    # Логгер для тензорборда

    tb_logger = TensorBoardLogger(
        save_dir="./experiments",
        version="version_" + datetime.now().strftime("%d-%m-%Y--%H-%M-%S"),
        name="",
    )
    
    ckpt_path = os.path.join(
        "./experiments/", tb_logger.version, "checkpoints",
    )
    
    csv_logger = CSVLogger('./', name='csv_logs', version='v2')

    # Чекпоинты
    
    checkpoint_callback = ModelCheckpoint(
        save_top_k = hparams.save_top_k,
        verbose = True,
        monitor = hparams.monitor,
        mode = hparams.metric_mode,
        save_weights_only = True
    )

    # Инициализация pl класса трэйнера
    #callbacks= early_stop_callback,

    trainer = Trainer(
        logger=[tb_logger, csv_logger],
        checkpoint_callback=True,
        gpus=[0, 3],
        log_gpu_memory="all",
        deterministic=True,
        fast_dev_run=False,
        precision = 16,
        accumulate_grad_batches=hparams.accumulate_grad_batches,
        max_epochs=hparams.max_epochs,
        min_epochs=hparams.min_epochs,
        val_check_interval=hparams.val_check_interval,
        strategy="ddp",
    )
    
#     #  запуск обучения
    print('START TRAINING')
    trainer.fit(model, model.data)
    print('START TESTING')
#     chk_path = "./experiments/version_01-06-2021--18-09-58/checkpoints/epoch=2-step=315.ckpt"
#     model_saved = model.load_from_checkpoint(chk_path)
    trainer.test(model, datamodule = model.data, verbose=True)


if __name__ == "__main__":
    # ------------------------
    # TRAINING ARGUMENTS
    # ------------------------
    # these are project-wide arguments
    parser = argparse.ArgumentParser(
        description="Minimalist Transformer Classifier",
        add_help=True,
    )
    parser.add_argument("--seed", type=int, default=3, help="Training seed.")
    parser.add_argument(
        "--save_top_k",
        default=1,
        type=int,
        help="The best k models according to the quantity monitored will be saved.",
    )
    # Early Stopping
    parser.add_argument(
        "--monitor", default="val_loss", type=str, help="Quantity to monitor."
    )
    parser.add_argument(
        "--metric_mode",
        default="max",
        type=str,
        help="If we want to min/max the monitored quantity.",
        choices=["auto", "min", "max"],
    )
    parser.add_argument(
        "--patience",
        default=3,
        type=int,
        help=(
            "Number of epochs with no improvement "
            "after which training will be stopped."
        ),
    )
    parser.add_argument(
        "--min_epochs",
        default=1,
        type=int,
        help="Limits training to a minimum number of epochs",
    )
    parser.add_argument(
        "--max_epochs",
        default=20,
        type=int,
        help="Limits training to a max number number of epochs",
    )

    # Batching
    parser.add_argument(
        "--batch_size", default=6, type=int, help="Batch size to be used."
    )
    parser.add_argument(
        "--accumulate_grad_batches",
        default=1,
        type=int,
        help=(
            "Accumulated gradients runs K small batches of size N before "
            "doing a backwards pass."
        ),
    )

    # gpu args
    parser.add_argument("--gpus", type = list, default = [0, 3], help = "How many gpus")
    parser.add_argument(
        "--val_check_interval",
        default = 2.0,
        type=float,
        help=(
            "If you don't want to use the entire dev set (for debugging or "
            "if it's huge), set how much of the dev set you want to use with this flag."
        ),
    )

    # each LightningModule defines arguments relevant to it
    parser = PET_Model.add_model_specific_args(parser)
    hparams = parser.parse_args()

    # ---------------------
    # RUN TRAINING
    # ---------------------
    main(hparams)