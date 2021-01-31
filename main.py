import gin
import logging
from absl import app, flags
from train import Trainer
from evaluation.eval import evaluate
from input_pipeline import datasets
from utils import utils_params, utils_misc
from models.architectures import DenseNet121
import tensorflow as tf



FLAGS = flags.FLAGS
flags.DEFINE_boolean('train', True, 'Specify whether to train or evaluate a model.') # change train flas to decied train or evaluation

def main(argv):

    # generate folder structures
    run_paths = utils_params.gen_run_folder()
    # set loggers
    utils_misc.set_loggers(run_paths['path_logs_train'], logging.INFO)

    # gin-config
    gin.parse_config_files_and_bindings([r'D:\Uni Stuttgart\Deep learning lab\Diabetic Retinopathy Detection\dl-lab-2020-team08\diabetic_retinopathy\configs\config.gin'], [])
    utils_params.save_config(run_paths['path_gin'], gin.config_str())

    # setup pipeline
    train_ds, valid_ds, test_ds = datasets.load()

    # training including fine tuning
    if FLAGS.train:
        # model
        if FLAGS.train:
            model = DenseNet121(IMG_SIZE=256)
            model.summary()

            # training and fine tuning
            trainer = Trainer(model=model, ds_train=train_ds, ds_val=valid_ds, run_paths=run_paths)
            for _ in trainer.train():
                continue

    else:
        # evaluation
        # model dir should be replaced by saved model dir
        model_dir = r"\diabetic_retinopathy\logs\20201221-225335\saved_model_ft"
        model = tf.keras.models.load_model(model_dir)
        evaluate(model, valid_ds)


if __name__ == "__main__":
    app.run(main)
