import gin
import tensorflow as tf
import logging
import datetime
import os

@gin.configurable()
class Trainer(object):
    def __init__(self, model,
              ds_train,
              ds_val,
              lr,
              lr_ft,
              ft_layer_idx,
              run_paths,
              total_steps,
              total_steps_ft,
              log_interval,
              ckpt_interval,
              fine_tune=True):
        # Summary Writer
        self.train_summary_writer = tf.summary.create_file_writer
        self.valid_summary_writer = tf.summary.create_file_writer
        # Checkpoint Manager
        self.ckpt = tf.train.Checkpoint
        self.manager = tf.train.CheckpointManager

        self.model = model
        self.ds_train = ds_train
        self.ds_val = ds_val
        self.lr = lr
        self.lr_ft = lr_ft
        self.ft_layer_idx = ft_layer_idx
        self.fine_tune = fine_tune
        self.run_paths = run_paths
        self.total_steps = total_steps
        self.total_steps_ft = total_steps_ft
        self.log_interval = log_interval
        self.ckpt_interval = ckpt_interval

        # Loss objective
        self.loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.lr)
        self.optimizer_ft = tf.keras.optimizers.Adam(learning_rate=self.lr_ft)

        # Metrics
        self.train_loss = tf.keras.metrics.Mean(name='train_loss')
        self.train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')

        self.test_loss = tf.keras.metrics.Mean(name='test_loss')
        self.test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')



    @tf.function
    def train_step(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    @tf.function
    def test_step(self, images, labels):
        # training=False is only needed if there are layers with different
        # behavior during training versus inference (e.g. Dropout).
        predictions = self.model(images, training=False)
        t_loss = self.loss_object(labels, predictions)

        self.test_loss(t_loss)
        self.test_accuracy(labels, predictions)

    @tf.function
    def train_step_ft(self, images, labels):
        with tf.GradientTape() as tape:
            # training=True is only needed if there are layers with different
            # behavior during training versus inference (e.g. Dropout).
            predictions = self.model(images, training=True)
            loss = self.loss_object(labels, predictions)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer_ft.apply_gradients(zip(gradients, self.model.trainable_variables))

        self.train_loss(loss)
        self.train_accuracy(labels, predictions)

    def train(self):
        print("start training")
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        train_log_dir = 'logs/' + current_time + '/train'
        valid_log_dir = 'logs/' + current_time + '/valid'
        model_log_dir = 'logs/' + current_time + '/saved_model'
        train_summary_writer = self.train_summary_writer(train_log_dir)
        valid_summary_writer = self.valid_summary_writer(valid_log_dir)
        ckpt = self.ckpt(step=tf.Variable(1), optimizer=self.optimizer, net=self.model)
        manager = self.manager(ckpt, self.run_paths["path_ckpts_train"], max_to_keep=10)
        tf.profiler.experimental.start('logs/'+ current_time)

        for idx, (images, labels) in enumerate(self.ds_train):

            step = idx + 1
            self.train_step(images, labels)

            # Profiler of first 20 step
            if step == 20:
                tf.profiler.experimental.stop()

            if step % self.log_interval == 0:

                # Reset test metrics
                self.test_loss.reset_states()
                self.test_accuracy.reset_states()

                for val_images, val_labels in self.ds_val:
                    self.test_step(val_images, val_labels)

                template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'

                logging.info(template.format(step,
                                self.train_loss.result(),
                                self.train_accuracy.result() * 100,
                                self.test_loss.result(),
                                self.test_accuracy.result() * 100))

                # Reset train metrics
                self.train_loss.reset_states()
                self.train_accuracy.reset_states()

                # Write summary to tensorboard
                with train_summary_writer.as_default():
                    tf.summary.scalar('loss', self.train_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step)
                with valid_summary_writer.as_default():
                    tf.summary.scalar('loss', self.test_loss.result(), step=step)
                    tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step)

                #yield self.test_accuracy.result().numpy()

            if step % self.ckpt_interval == 0:
                logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                # Save checkpoint
                manager.save()

            if step % self.total_steps == 0:
                logging.info(f'Finished training after {step} steps.')
                # save the whole model
                self.model.save(model_log_dir)
                print("Saved model for step {}: {}".format(step, model_log_dir))
                break

        if self.fine_tune == True:
            print("start fine tuning")
            ckpt = self.ckpt(step=tf.Variable(1), optimizer=self.optimizer_ft, net=self.model)
            manager = self.manager(ckpt, self.run_paths["path_ckpts_train"], max_to_keep=10)
            for layer in self.model.layers[self.ft_layer_idx:]:
                layer.trainable = True

            for idx, (images, labels) in enumerate(self.ds_train):
                step_ft = step + idx + 1
                self.train_step_ft(images, labels)

                if step_ft % (self.log_interval//10) == 0:

                    # Reset test metrics
                    self.test_loss.reset_states()
                    self.test_accuracy.reset_states()

                    for val_images, val_labels in self.ds_val:
                        self.test_step(val_images, val_labels)

                    template = 'Step {}, Loss: {}, Accuracy: {}, Validation Loss: {}, Validation Accuracy: {}'

                    logging.info(template.format(step_ft,
                                    self.train_loss.result(),
                                    self.train_accuracy.result() * 100,
                                    self.test_loss.result(),
                                    self.test_accuracy.result() * 100))

                    # Reset train metrics
                    self.train_loss.reset_states()
                    self.train_accuracy.reset_states()

                    # Write summary to tensorboard
                    with train_summary_writer.as_default():
                        tf.summary.scalar('loss', self.train_loss.result(), step=step_ft)
                        tf.summary.scalar('accuracy', self.train_accuracy.result(), step=step_ft)
                    with valid_summary_writer.as_default():
                        tf.summary.scalar('loss', self.test_loss.result(), step=step_ft)
                        tf.summary.scalar('accuracy', self.test_accuracy.result(), step=step_ft)
                    yield self.test_accuracy.result().numpy()

                if step_ft % (self.ckpt_interval//10) == 0:
                    logging.info(f'Saving checkpoint to {self.run_paths["path_ckpts_train"]}.')
                    # Save checkpoint
                    manager.save()


                if step_ft % (self.total_steps_ft + self.total_steps) == 0:
                    logging.info(f'Finished fine tuning after {step} steps.')
                    # save the whole model
                    model_log_dir = 'logs/' + current_time + '/saved_model_ft'
                    self.model.save(model_log_dir)
                    print("Saved model for step {}: {}".format(step, model_log_dir))
                    return self.test_accuracy.result().numpy()
