from collections import deque
from datetime import datetime
import logging
import os
import pprint as pp
from importlib import reload
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
# from LIB.tf_utils import shape ,get_streaming_metrics
from sys import stdout

class TFBaseModel(object):

    """Interface : containing some boilerplate code for training tensorflow models.
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    !!for validation  label=self.target , logits=self.predictions ,accuracy =self.accuracy     !
    !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    if you will use loss you should inverse the comparison

    Subclassing models must implement self.calculate_loss(), which returns a tensor for the batch loss.
    Code for the training loop, parameter updates, checkpointing, and inference are implemented here and
    subclasses are mainly responsible for building the computational graph beginning with the placeholders
    and ending with the loss tensor.

    Args:
        reader: Class with attributes train_batch_generator, val_batch_generator, and test_batch_generator
            that yield dictionaries mapping tf.placeholder names (as strings) to batch data (numpy arrays).
        batch_size: Minibatch size.
        learning_rate: Learning rate.
        optimizer: 'rms' for RMSProp, 'adam' for Adam, 'sgd' for SGD
        grad_clip: Clip gradients elementwise to have norm at most equal to grad_clip.
        regularization_constant:  Regularization constant applied to all trainable parameters.
        keep_prob: 1 - p, where p is the dropout probability
        early_stopping_epochs:  Number of steps to continue training after validation loss has
            stopped decreasing.
        warm_start_init_step:  If nonzero, model will resume training a restored model beginning
            at warm_start_init_step.
        num_restarts:  After validation loss plateaus, the best checkpoint will be restored and the
            learning rate will be halved.  This process will repeat num_restarts times.
        enable_parameter_averaging:  If true, model saves exponential weighted averages of parameters
            to separate checkpoint file.
        min_steps_to_checkpoint:  Model only saves after min_steps_to_checkpoint training steps
            have passed.
        log_interval:  Train and validation accuracies are logged every log_interval training steps.
        loss_averaging_window:  Train/validation losses are averaged over the last loss_averaging_window
            training steps.
        num_validation_batches:  Number of batches to be used in validation evaluation at each step.
        log_dir: Directory where logs are written.
        checkpoint_dir: Directory where checkpoints are saved.
        prediction_dir: Directory where predictions/outputs are saved.
    """

    def __init__(
        self,
        reader,
        num_epoch=1000,
        batch_size=128,
        num_training_steps=20000,
        learning_rate=.001,
        optimizer='adam',
        grad_clip=5,
        regularization_constant=0.0,
        keep_prob=1.0,
        early_stopping_epochs=3000,
        warm_start_init_step=0,
        num_restarts=None,
        enable_parameter_averaging=False,
        min_epoch_to_checkpoint=100,
        log_interval=20,
        loss_averaging_window=100,
        num_validation_batches=1,
        log_dir='logs',
        checkpoint_dir='checkpoints',
        prediction_dir='predictions',
        Tensorbord_dir="Graph",
        log_metrics=True,
        decay_steps=40000,
        decay_rate=0.95


        ):

        self.reader = reader
        self.batch_size = batch_size
        self.num_training_steps = num_training_steps
        self.learning_rate = learning_rate
        self.optimizer = optimizer
        self.grad_clip = grad_clip
        self.regularization_constant = regularization_constant
        self.warm_start_init_step = warm_start_init_step
        self.early_stopping_epochs = early_stopping_epochs
        self.keep_prob_scalar = keep_prob
        self.enable_parameter_averaging = enable_parameter_averaging
        self.num_restarts = num_restarts
        self.min_epoch_to_checkpoint = min_epoch_to_checkpoint
        self.log_interval = log_interval
        self.num_validation_batches = num_validation_batches
        self.loss_averaging_window = loss_averaging_window
        self.log_metrics=log_metrics
        self.log_dir = log_dir
        self.prediction_dir = prediction_dir
        self.checkpoint_dir = checkpoint_dir
        self.num_epoch=num_epoch
        self.Tensorbord_dir=Tensorbord_dir
        self.decay_steps=decay_steps
        self.decay_rate=decay_rate

        if self.enable_parameter_averaging:
            self.checkpoint_dir_averaged = checkpoint_dir + '_avg'

        self.init_logging(self.log_dir)
        logging.info('\nnew run with parameters:\n{}'.format(pp.pformat(self.__dict__)))
        config = tf.ConfigProto()
        config.allow_soft_placement = True
        config.gpu_options.allow_growth = True
        self.graph = self.build_graph()
        self.session = tf.Session(graph=self.graph,config=config)
        print('built graph')

    def calculate_loss(self):
        raise NotImplementedError('subclass must implement this')



    def fit(self):

        with self.session.as_default():

            if self.warm_start_init_step:
                self.restore(self.warm_start_init_step)
                self.epoch= self.warm_start_init_step
            else:
                self.session.run(self.init)
                self.session.run(self.init_l)
                self.epoch=0

            train_generator = self.reader.train_epoch_generator(self.num_epoch)
            validation_generator = self.reader.val_epoch_generator(self.num_epoch)
            #self.num_validation_batches*self.batch_size

            train_loss_history = []
            train_accuracy_history = []

            validation_loss_history =   []
            validation_accuracy_history = []


            self.best_validation_score_down, self.best_validation_tstep = float('inf'), 0
            self.best_validation_score, self.best_validation_tstep = float('-inf'), 0
            self.restarts = 0
            self.step_train=0

            while self.epoch < self.num_epoch:
                train_batches=next(train_generator)
                validation_batches=next(validation_generator)
                train_loss_epoch=[]
                train_accuracy_epoch=[]

                validation_loss_epoch=[]
                validation_accuracy_epoch=[]
            ############################Train#####################################
                step_display=0

                for i, train_batch_df in enumerate(train_batches.batch_generator(self.batch_size)):
                    step_display=step_display+1
                    train_feed_dict = {
                        getattr(self, placeholder_name, None): data
                        for placeholder_name, data in train_batch_df if hasattr(self, placeholder_name)
                    }

                    train_feed_dict.update({self.learning_rate_var: self.learning_rate})

                    if hasattr(self, 'keep_prob'):
                        train_feed_dict.update({self.keep_prob: self.keep_prob_scalar})
                    if hasattr(self, 'is_training'):
                        train_feed_dict.update({self.is_training: True})

                    train_loss, grad,summary,ame= self.session.run(
                        fetches=[self.loss, self.step, self.merge_Sum,self.ame],
                        feed_dict=train_feed_dict
                    )
                    self.step_train+=1
                    self.train_writer.add_summary(summary,self.step_train)
                    train_loss_epoch.append(train_loss)
                    train_accuracy_epoch.append(ame)
                    if step_display == 30 :

                        metric_log = (
                                        "[[step {:>8}]]   "
                                        "[[train]]   loss: {:<8}  -ame :{:<8}"



                                    ).format(self.epoch, round(np.mean(train_loss_epoch), 8),round(np.mean(train_accuracy_epoch), 8))
                        stdout.write("\r%s" % metric_log )
                        step_display=0





                train_mean_loss=np.mean(train_loss_epoch)
                train_mean_accuracy=np.mean(train_accuracy_epoch)

                train_loss_history.append(train_mean_loss)
                train_accuracy_history.append(train_mean_accuracy)
                ############################Validation#####################################
                y_true=[]
                y_pred=[]
                for i, validation_batch_df in enumerate(validation_batches.batch_generator(self.num_validation_batches*self.batch_size)):

                    validation_feed_dict = {
                    getattr(self, placeholder_name, None): data
                    for placeholder_name, data in validation_batch_df if hasattr(self, placeholder_name)
                    }
                    if hasattr(self, 'keep_prob'):
                        validation_feed_dict.update({self.keep_prob: 1.0})
                    if hasattr(self, 'is_training'):
                        validation_feed_dict.update({self.is_training: False})


                    validation_loss,ame,summary= self.session.run(
                        fetches=[self.loss,self.ame,self.merge_Sum],
                        feed_dict=validation_feed_dict
                        )
                    self.test_writer.add_summary(summary,self.step_train)
                    validation_loss_epoch.append(validation_loss)
                    validation_accuracy_epoch.append(ame)




                validation_mean_loss=np.mean(validation_loss_epoch)
                validation_mean_accuracy=np.mean(validation_accuracy_epoch)
                validation_loss_history.append(validation_mean_loss)
                validation_accuracy_history.append(validation_mean_accuracy)
                ######################## DISPLAY ##########################################################
                stdout.write("\n")

                metric_log = (
                                        "[[epoch {:>8}]]     "
                                        "[[train]]     loss: {:<12}-ame :{:<12} \n"
                                        "[[val]]     loss: {:<12}-ame :{:<12} "


                                    ).format(self.epoch, round(train_mean_loss, 8),round(train_mean_accuracy, 8),
                                    round(validation_mean_loss, 8),round(validation_mean_accuracy, 8))
                logging.info(metric_log)


                if(self.save_with_best_score_down(validation_mean_accuracy)):break

                self.epoch += 1

            if self.epoch <= self.min_epoch_to_checkpoint:
                self.best_validation_tstep = self.epoch
                self.save(self.epoch)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

                logging.info('num_training_steps reached - ending training')
#########################################Predict function ####################################
    def predict(self,data, chunk_size=1024):

        if hasattr(self, 'prediction_tensors'):
            if(data=="test"):
                test_generator = self.reader.test_epoch_generator(1)
                test_batches=next(test_generator)
            elif (data=="train") :
                test_generator = self.reader.train_epoch_generator(10)
                test_batches=next(test_generator)
            elif (data=="val") :
                test_generator = self.reader.val_epoch_generator(1)
                test_batches=next(test_generator)
            else  :
                print("Ther is  no avaible data with name "+str(data))
                return
            prediction_dict = {tensor_name: [] for tensor_name in self.prediction_tensors}
            print(test_batches.__len__())


            for i, test_batch_df in enumerate(test_batches.batch_generator(chunk_size)):

                test_feed_dict = {
                getattr(self, placeholder_name, None): data
                for placeholder_name, data in test_batch_df if hasattr(self, placeholder_name)
                }
                if hasattr(self, 'keep_prob'):
                    test_feed_dict.update({self.keep_prob: 1.0})
                if hasattr(self, 'is_training'):
                    test_feed_dict.update({self.is_training: False})

                tensor_names, tf_tensors = zip(*self.prediction_tensors.items())
                np_tensors = self.session.run(
                    fetches=tf_tensors,
                    feed_dict=test_feed_dict
                )
                for tensor_name, tensor in zip(tensor_names, np_tensors):
                    prediction_dict[tensor_name].extend(tensor.tolist())


            return prediction_dict
        else :
            print("defin dic of prediction_tensors { name:tensor} ")

    def get_parameter_tensors (self):
        prediction_dict = {tensor_name: [] for tensor_name in self.parameter_tensors}
        tensor_names, tf_tensors = zip(*self.parameter_tensors.items())
        np_tensors = self.session.run(
            fetches=tf_tensors
        )
        for tensor_name, tensor in zip(tensor_names, np_tensors):
            prediction_dict[tensor_name]=tensor
        return prediction_dict
    def save(self, step, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not os.path.isdir(checkpoint_dir):
            logging.info('creating checkpoint directory {}'.format(checkpoint_dir))
            os.mkdir(checkpoint_dir)

        model_path = os.path.join(checkpoint_dir, 'model')
        logging.info('saving model to {}'.format(model_path))
        saver.save(self.session, model_path, global_step=step)
    def restore(self, step=None, averaged=False):
        saver = self.saver_averaged if averaged else self.saver
        checkpoint_dir = self.checkpoint_dir_averaged if averaged else self.checkpoint_dir
        if not step:
            model_path = tf.train.latest_checkpoint(checkpoint_dir)
            logging.info('restoring model parameters from {}'.format(model_path))
            saver.restore(self.session, model_path)
        else:
            model_path = os.path.join(
                checkpoint_dir, 'model{}-{}'.format('_avg' if averaged else '', step)
            )
            logging.info('restoring model from {}'.format(model_path))
            saver.restore(self.session, model_path)
    def init_logging(self, log_dir):
        if not os.path.isdir(log_dir):
            os.makedirs(log_dir)

        date_str = datetime.now().strftime('%Y-%m-%d_%H-%M')
        log_file = 'log_{}.txt'.format(date_str)

        reload(logging)  # bad
        logging.basicConfig(
            filename=os.path.join(log_dir, log_file),
            level=logging.INFO,
            format='[[%(asctime)s]] %(message)s',
            datefmt='%m/%d/%Y %I:%M:%S %p'
        )
        logging.getLogger().addHandler(logging.StreamHandler())
    def update_parameters(self, loss):
        self.global_step = tf.Variable(0, trainable=False)
        self.learning_rate_var = tf.Variable(0.0, trainable=False)

        if self.regularization_constant != 0:
            l2_norm = tf.reduce_sum([tf.sqrt(tf.reduce_sum(tf.square(param))) for param in tf.trainable_variables()])
            loss = loss + self.regularization_constant*l2_norm
        with tf.variable_scope("Optimizer"):
            self.learning_rate_deacy= tf.train.exponential_decay(self.learning_rate, self.global_step,
                                           self.decay_steps, self.decay_rate, staircase=True)
            optimizer = self.get_optimizer(self.learning_rate_deacy)
            grads = optimizer.compute_gradients(loss)
            clipped = [(tf.clip_by_value(g, -5., 5.), v_) for g, v_ in grads]
            #         self.log_y=clipped
            step = optimizer.apply_gradients(clipped, global_step=self.global_step)

        if self.enable_parameter_averaging:
            maintain_averages_op = self.ema.apply(tf.trainable_variables())
            with tf.control_dependencies([step]):
                self.step = tf.group(maintain_averages_op)
        else:
            self.step = step

        #         logging.info('all parameters:')
        #         logging.info(pp.pformat([(var.name, shape(var)) for var in tf.global_variables()]))

        #         logging.info('trainable parameters:')
        #         logging.info(pp.pformat([(var.name, shape(var)) for var in tf.trainable_variables()]))

        #         logging.info('trainable parameter count:')
        #         logging.info(str(np.sum(np.prod(shape(var)) for var in tf.trainable_variables())))
    def get_optimizer(self, learning_rate):
        if self.optimizer == 'adam':
            return tf.train.AdamOptimizer(learning_rate)
        elif self.optimizer == 'gd':
            return tf.train.GradientDescentOptimizer(learning_rate)
        elif self.optimizer == 'rms':
            return tf.train.RMSPropOptimizer(learning_rate, decay=0.95, momentum=0.9)
        else:
            assert False, 'optimizer must be adam, gd, or rms'
    def build_graph(self):
        with tf.Graph().as_default() as graph:
            self.ema = tf.train.ExponentialMovingAverage(decay=0.995)

            self.loss = self.calculate_loss()
            self.update_parameters(self.loss)
           # self.merge_Sum=self.tensorbord(graph)
            self.saver = tf.train.Saver(max_to_keep=1)
            if self.enable_parameter_averaging:
                self.saver_averaged = tf.train.Saver(self.ema.variables_to_restore(), max_to_keep=1)

            self.init = tf.global_variables_initializer()
            self.init_l = tf.local_variables_initializer()
            self.merge_Sum=self.tensorbord(graph)


            return graph
    def get_grpah(self):
        return self.graph
    def metrics(self,X,Y):
        raise NotImplementedError('subclass must implement this')

    def tensorbord(self,graph):
        import shutil
        with tf.name_scope('Tensorboard'):
            
#             if  os.path.isdir(self.Tensorbord_dir):
#                 try:
#                     shutil.rmtree(self.Tensorbord_dir)
#                     os.mkdir(self.Tensorbord_dir)
#                 except OSError:
#                     print ("Error")
# #                 os.remove(self.Tensorbord_dir)
#             logging.info('creating TensorBord  directory {}'.format(self.Tensorbord_dir))
            
            print("creating TensorBord  directory")
            self.train_writer = tf.summary.FileWriter(self.checkpoint_dir +"/Train",graph)
            self.test_writer = tf.summary.FileWriter(self.checkpoint_dir+"/Test")
            for tensor_name , tensor in self.Dict_Tensorboard.items() :
                tf.summary.scalar(tensor_name,tensor)
            tf.summary.scalar("lr",self.learning_rate_deacy)
            self.projector()
            #self.confusion_update ,confusion_image=get_streaming_metrics()
            #tf.summary.image('confusion_image',self.confusion_image)
            return tf.summary.merge_all()

    def projector(self):
        config = projector.ProjectorConfig()
        # One can add multiple embeddings.
        for name , tensor  in self.projectorTensor.items() :
            setattr(self,name,config.embeddings.add())
            getattr(self,name).tensor_name=tensor.name
        checkpoint_dir =  self.checkpoint_dir
        projector.visualize_embeddings(tf.summary.FileWriter(checkpoint_dir), config)

    def save_with_best_score_down(self,new_score):
        if self.epoch >= self.min_epoch_to_checkpoint:
            if new_score  < self.best_validation_score_down:

                self.best_validation_score_down = new_score
                self.best_validation_tstep = self.epoch

                logging.info('best validation score  of {} at test step {}'.format(
                    self.best_validation_score_down, self.best_validation_tstep))
                self.save(self.epoch)
                if self.enable_parameter_averaging:
                    self.save(step, averaged=True)

        if self.epoch - self.best_validation_tstep > self.early_stopping_epochs:

            if self.num_restarts is None or self.restarts >= self.num_restarts:
                logging.info('best validation score of {} at training step {}'.format(
                    self.best_validation_score_down, self.best_validation_tstep))
                logging.info('early stopping - ending training.')
                return True

            if self.restarts < self.num_restarts:
                self.restore(self.best_validation_tstep)
                logging.info('halving learning rate')
                self.learning_rate /= 2.0
                #                 self.early_stopping_epochs /= 2
                self.epoch = self.best_validation_tstep
                self.restarts += 1
