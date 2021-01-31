import logging
import gin
from ray import tune
from input_pipeline import datasets
from train import Trainer
from utils import utils_params, utils_misc
from models.architectures import DenseNet121


def train_func(config):
    # Hyperparameters
    bindings = []
    for key, value in config.items():
        bindings.append(f'{key}={value}')

    # generate folder structures
    run_paths = utils_params.gen_run_folder(bindings[2])

    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    # gin dir should be replaced by your own dir
    gin.parse_config_files_and_bindings([r'D:\Uni Stuttgart\Deep learning lab\Diabetic Retinopathy Detection\dl-lab-2020-team08\diabetic_retinopathy\configs\config.gin'],
                                        bindings)
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    train_ds, valid_ds, test_ds = datasets.load()

    # model
    model = DenseNet121(IMG_SIZE=256)

    trainer = Trainer(model=model, ds_train=train_ds, ds_val=test_ds, run_paths=run_paths)
    for val_accuracy in trainer.train():
        tune.report(val_accuracy=val_accuracy)
# some question about tune api

analysis = tune.run(
    train_func, num_samples=100, resources_per_trial={'gpu': 1, 'cpu': 2},
    config={
        "Trainer.total_steps": tune.grid_search([5000]),
        "Trainer.total_steps_ft": tune.randint(300, 1500),
        "Trainer.lr": tune.loguniform(1e-3, 1e-2),
        "Trainer.lr_ft": tune.loguniform(1e-6, 1e-4),
        "Trainer.ft_layer_idx": tune.randint(100, 300),
        "DenseNet121.dense_units": tune.randint(2, 64),
        "DenseNet121.dropout_rate": tune.uniform(0, 0.9),
        "DenseNet121.idx_layer": tune.randint(200, 400)
    })

print("Best config: ", analysis.get_best_config(metric="val_accuracy", mode="max"))

# Get a dataframe for analyzing trial results.
df = analysis.dataframe()
