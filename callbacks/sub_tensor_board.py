from keras.callbacks import TensorBoard


class SubTensorBoard(TensorBoard):
    def __init__(self, logdir):
        super(SubTensorBoard, self).__init__(log_dir=logdir)

    def on_episode_end(self, epoch, logs):
        timesteps = logs["nb_steps"]
        super(SubTensorBoard, self).on_epoch_end(timesteps, logs)