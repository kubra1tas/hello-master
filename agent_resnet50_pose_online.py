import math
import shutil
import time
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import style
import numpy as np
import pyquaternion as pq
import torch
import torch.optim as optim
from scipy import io
from torch import nn
from torch.backends import cudnn

from agents.base import BaseAgent
from datasets.H5DataLoader import H5Dataloader
from graphs.models import model_resnet50_pose
from utils.metrics import AverageMeter
from utils.train_utils import adjust_learning_rate
from utils.transformation import Transformation
import sys

cudnn.benchmark = True


class agent_resnet50_pose_online(BaseAgent):

    def __init__(self, config):
        super().__init__(config)

        # define models
        self.model = model_resnet50_pose.Model_Resnet50_Pose(self.config)

        # print("Current Model")
        # print(self.model.model)
        # print("")

        # print("Model Overview")
        # print("{:40} {:40}".format("Layer name", "data shape"))
        # print("="*80)
        # for name, params in self.model.model.named_parameters():
        #     # print(name, "\t\t\t\t", params.data.shape)
        #     print("{:40} {:40}".format(str(name), str(params.data.shape)))
        # print("="*80)

        # define data_loader
        self.data_loader = H5Dataloader(config=config)

        # define loss
        self.loss = nn.MSELoss()

        # define optimizers for both generator and discriminator
        # TODO change optimizer to ADAM
        # self.optimizer = optim.SGD(self.model.parameters(),
        #                            lr=self.config.sgd_learning_rate,
        #                            momentum=float(self.config.sgd_momentum),
        #                            weight_decay=self.config.sgd_weight_decay,
        #                            dampening=float(self.config.sgd_damping),
        #                            nesterov=self.config.sgd_nesterov
        #                            )

        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        # Define Scheduler
        # lambda1 = lambda epoch: pow((1 - ((epoch - 1) / self.config.max_epoch)), 0.9)
        # self.scheduler = lr_scheduler.LambdaLR(self.optimizer, lr_lambda=lambda1)

        # initialize counter
        self.current_epoch = 0
        self.current_iteration = 0
        self.best_valid_mean_loss = 100.0

        # set cuda flag
        self.is_cuda = torch.cuda.is_available()
        if self.is_cuda and not self.config.cuda:
            self.logger.info("WARNING: You have a CUDA device, so you should probably enable CUDA")

        self.cuda = self.is_cuda & self.config.cuda

        # set the manual seed for torch
        self.manual_seed = self.config.seed
        if self.cuda:
            torch.cuda.manual_seed_all(self.manual_seed)
            torch.cuda.set_device(self.config.gpu_device)
            self.device = torch.device("cuda")
            self.model = self.model.cuda()
            self.loss = self.loss.cuda()
            self.logger.info("Program will run on *****GPU-CUDA***** ")
        # print_cuda_statistics()
        else:
            self.device = torch.device("cpu")
            torch.manual_seed(self.manual_seed)
            self.logger.info("Program will run on *****CPU*****\n")

        self.model = self.model.to(self.device).double()
        self.loss = self.loss.to(self.device)
        # Model Loading from the latest checkpoint if not found start from scratch.
        self.load_checkpoint(self.config.checkpoint_file)
        # Summary Writer
        self.summary_writer = True

        # print summary of model
        # summary(self.model.model, (1, 224, 224))
        self.transformation = Transformation(config)

        # timing
        self.start_time = None
        self.duration = None

        # dict for saving logging data
        self.logging_dict = {}
        self.logging_dict["learning_rate"] = []
        self.logging_dict["train_loss"] = []
        self.logging_dict["train_err_rotation"] = []
        self.logging_dict["train_err_translation"] = []
        self.logging_dict["train_err_joints"] = []
        self.logging_dict["valid_loss"] = []
        self.logging_dict["valid_err_rotation"] = []
        self.logging_dict["valid_err_translation"] = []
        self.logging_dict["valid_err_joints"] = []
        self.logging_dict["training_duration_sec"] = []

        # animation window
        # self.fig = plt.figure()
        # self.ax1 = self.fig.add_subplot(211)
        # self.ax2 = self.fig.add_subplot(212)

        self.iter = 0
        self.q_dist_mean = []
        self.trans_mean = []
        self.joints_mean = []

    def save_checkpoint(self, filename='checkpoint.pth.tar', is_best=0):
        """
        Saving the latest checkpoint of the training
        :param filename: filename which will contain the state
        :param is_best: flag is it is the best model
        :return:
        """
        state = {
            'epoch': self.current_epoch + 1,
            'iteration': self.current_iteration,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }
        # Save the state
        torch.save(state, self.config.checkpoint_dir + filename)
        # If it is the best copy it to another file 'model_best.pth.tar'
        if is_best:
            shutil.copyfile(self.config.checkpoint_dir + filename,
                            self.config.checkpoint_dir + 'model_best.pth.tar')

    def load_checkpoint(self, filename):
        filename = self.config.checkpoint_dir + filename
        try:
            self.logger.info("Loading checkpoint '{}'".format(filename))
            checkpoint = torch.load(filename)

            self.current_epoch = checkpoint['epoch']
            self.current_iteration = checkpoint['iteration']
            self.model.load_state_dict(checkpoint['state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])

            self.logger.info("Checkpoint loaded successfully from '{}' at (epoch {}) at (iteration {})\n"
                             .format(self.config.checkpoint_dir, checkpoint['epoch'], checkpoint['iteration']))
        except OSError as e:
            self.logger.info("No checkpoint exists from '{}'. Skipping...".format(self.config.checkpoint_dir))
            self.logger.info("**First time to train**")

    def run(self):
        """
        The main operator
        :return:
        """
        assert self.config.mode in ['train', 'test', 'deploy']
        try:
            if self.config.mode == 'test':
                self.validate()
            elif self.config.mode == 'train':
                self.train()
            elif self.config.mode == 'deploy':
                self.deploy()

        except KeyboardInterrupt:
            self.logger.info("You have entered CTRL+C.. Wait to finalize")

    def deploy(self):
        # load a sample image
        example_image, example_label = next(iter(self.data_loader.train_loader))
        print(example_image)
        example_image = example_image.to(device=torch.device("cpu"), dtype=torch.double)
        self.model = self.model.to(torch.device("cpu"))
        self.model = self.model.double()
        print(example_image)
        # run the tracing
        traced_script_module = torch.jit.trace(self.model, example_image)

        # save converted model
        traced_script_module.save(self.config.checkpoint_dir + self.config.exp_name + ".pt")

    def monitor_joints(self, i):
        xs = []
        ys = []
        for line in self.joints_mean:
            xs.append(line[0])
            ys.append(line[1])
        self.ax1.clear()
        self.ax1.plot(xs, ys)
        self.ax1.set_title("Joint Error")
        self.ax1.set_ylabel("Degree")
        self.ax1.set_xlabel("Epoch")


    def monitor(self, i):
        xs = []
        ys = []
        for line in self.q_dist_mean:
            # x, y = line.split(',')
            xs.append(line[0])
            ys.append(line[1])
        self.ax1.clear()
        self.ax1.plot(xs, ys)
        self.ax1.set_title("Relative Pose Monitor - Rotation")
        self.ax1.set_ylabel("Degree")
        self.ax1.set_xlabel("Epoch")

        x = []
        y = []
        for line in self.trans_mean:
            x.append(line[0])
            y.append(line[1])
        self.ax2.clear()
        self.ax2.plot(x, y)
        self.ax2.set_title("Relative Pose Monitor - Translation")
        self.ax2.set_ylabel("Translation [mm]")
        self.ax2.set_xlabel("Epoch")

    def train(self):
        """
        Main training loop
        :return:
        """
        # self.current_epoch -= 5
        self.start_time = time.time()

        for epoch in range(self.current_epoch, self.config.max_epoch):
            self.current_epoch = epoch
            # self.scheduler.step(epoch)
            self.train_one_epoch()

            # monitor training data
            # style.use('fivethirtyeight')
            # ani = animation.FuncAnimation(self.fig, self.monitor, interval=100)
            # ani = animation.FuncAnimation(self.fig, self.monitor_joints, interval=100)
            # plt.pause(0.05)
            # plt.show()

            valid_loss = self.validate()
            # self.validate()
            # self.scheduler.step(valid_loss)

            # TODO save checkpoint if a better validation error has been achieved
            is_best = valid_loss < self.best_valid_mean_loss
            if is_best:
                self.best_valid_mean_loss = valid_loss

            # save dict for logging
            self.logging_dict["training_duration_sec"].append(time.time()-self.start_time)
            io.savemat("./experiments/" + self.config.exp_name + "/out/" + self.config.exp_name + "_logging_dict.mat",
                       self.logging_dict, appendmat=False)

            # print("Logging dict mat")
            # for key in self.logging_dict:
            #     print("Key: {:20} mean: {:3.4e}".format(key, np.mean(self.logging_dict[key])))

            # is_best = True
            self.save_checkpoint(is_best=is_best)

    def train_one_epoch(self):
        """
        One epoch training function
        """

        # Set the model to be in training mode
        self.model.train()
        # Initialize your average meters
        train_loss = AverageMeter()
        train_err_joints = AverageMeter()
        train_err_rotation = AverageMeter()
        train_err_translation = AverageMeter()

        total_batch = len(self.data_loader.train_loader)

        print("Starting Epoch Training with batch size: {:d}".format(total_batch))
        print("Batch Size: {:d}".format(self.config.batch_size))
        current_batch = 1

        # times = {}

        q_dist_list = []
        trans_list = []
        joint_list = []

        tic = time.time()

        mean_speed = 0
        total_loss_sum = 0

        samples = 300
        np_batch_bench = np.zeros([samples, 4])



        for x, y in self.data_loader.train_loader:
            batch_start = time.time()

            if self.cuda:
                x = x.to(device=self.device, dtype=torch.long)
                y = y.to(device=self.device, dtype=torch.long)

            progress = float(self.current_epoch * self.data_loader.train_iterations + current_batch) / (
                    self.config.max_epoch * self.data_loader.train_iterations)

            # adjust learning rate
            lr = adjust_learning_rate(self.optimizer, self.current_epoch, self.config, batch=current_batch,
                                      nBatch=self.data_loader.train_iterations)

            # model

            pred = self.model((x))

            if self.config.data_output_type == "joints_absolute":
                loss_joints = self.loss(pred, y)
                total_loss = loss_joints
                train_loss.update(total_loss.item())
                train_err_joints.update(total_loss.item())
            elif self.config.data_output_type == "q_trans_simple":
                loss_q_trans_simple = self.loss(pred, y)
                total_loss = loss_q_trans_simple
            elif self.config.data_output_type == "pose_relative":
                # loss for rotation
                # select rotation indices from the prediction tensor
                indices = torch.tensor([3, 4, 5, 6])
                indices = indices.to(self.device)
                rotation = torch.index_select(pred, 1, indices)
                # select rotation indices from the label tensor
                y_rot = torch.index_select(y, 1, indices)
                # calc MSE loss for rotation
                # loss_rotation = self.loss(rotation, y_rot)

                # trans_list.append(loss_rotation[0].item().numpy())
                # print(loss_rotation.item())

                # penalty loss from facebook paper posenet
                # penalty_loss = self.config.rot_reg * torch.mean((torch.sum(quater ** 2, dim=1) - 1) ** 2)
                penalty_loss = 0

                q_pred = pq.Quaternion(rotation[0].cpu().detach().numpy())
                q_rot = pq.Quaternion(y_rot[0].cpu().detach().numpy())
                q_dist = math.degrees(pq.Quaternion.distance(q_pred, q_rot))
                q_dist_list.append(q_dist)

                # loss for translation
                # select translation indices from the prediction tensor
                indices = torch.tensor([0, 1, 2])
                indices = indices.to(self.device)
                translation = torch.index_select(pred, 1, indices)
                # select translation indices from the label tensor
                y_trans = torch.index_select(y, 1, indices)

                # calc MSE loss for translation
                loss_translation = self.loss(translation, y_trans)
                trans_list.append(loss_translation.item())

                # total_loss = penalty_loss + loss_rotation + loss_translation
                # use simple loss
                total_loss = self.loss(pred.double(), y.double())

                # calc translation MSE
                q_pred = pq.Quaternion(rotation[0].cpu().detach().numpy())
                q_rot = pq.Quaternion(y_rot[0].cpu().detach().numpy())
                q_dist = math.degrees(pq.Quaternion.distance(q_pred, q_rot))
                q_dist_list.append(q_dist)
                trans_pred = translation[0].cpu().detach().numpy()
                trans_label = y_trans[0].cpu().detach().numpy()
                mse_trans = (np.square(trans_pred - trans_label)).mean()
                train_err_translation.update(mse_trans)
                train_err_rotation.update(q_dist)

            elif self.config.data_output_type == "pose_absolute":
                # select rotation indices from the prediction tensor
                indices = torch.tensor([3, 4, 5, 6])
                indices = indices.to(self.device)
                rotation = torch.index_select(pred, 1, indices)
                # select rotation indices from the label tensor
                y_rot = torch.index_select(y, 1, indices)

                q_pred = pq.Quaternion(rotation[0].cpu().detach().numpy())
                q_rot = pq.Quaternion(y_rot[0].cpu().detach().numpy())
                q_dist = math.degrees(pq.Quaternion.distance(q_pred, q_rot))
                q_dist_list.append(q_dist)

                # loss for translation
                # select translation indices from the prediction tensor
                indices = torch.tensor([0, 1, 2])
                indices = indices.to(self.device)
                translation = torch.index_select(pred, 1, indices)
                # select translation indices from the label tensor
                y_trans = torch.index_select(y, 1, indices)

                trans_pred = translation[0].cpu().detach().numpy()
                trans_label = y_trans[0].cpu().detach().numpy()

                # calc MSE loss for translation
                loss_translation = self.loss(translation, y_trans)
                trans_list.append(loss_translation.item())

                # use simple loss
                total_loss = self.loss(pred, y)

                # calc translation MSE
                mse_trans = (np.square(trans_pred - trans_label)).mean()
                train_err_translation.update(mse_trans)
                train_err_rotation.update(q_dist)

            elif self.config.data_output_type == "joints_relative":
                total_loss = self.loss(pred, y)
                train_err_joints.update(total_loss.item())
                # print("Train loss {:f}".format(total_loss.item()))
                joint_list.append(total_loss.item())
            else:
                raise Exception("Wrong data output type chosen.")

            if np.isnan(float(total_loss.item())):
                raise ValueError('Loss is nan during training...')

            # optimizer
            self.optimizer.zero_grad()
            total_loss.backward()
            self.optimizer.step()

            train_loss.update(total_loss.item())

            self.current_iteration += 1

            batch_duration = time.time() - batch_start
            mean_speed += batch_duration
            speed = float(mean_speed/current_batch)
            remaining_sec = speed * (total_batch-current_batch) * (self.config.max_epoch-self.current_epoch)
            batch_progress = float(current_batch/total_batch) * 100
            # print(int(batch_progress) % 5)

            total_loss_sum += total_loss.item()
            avg_total_loss = float(total_loss_sum / current_batch)
            #
            # if avg_total_loss <= self.config.min_avg_loss:
            #     print("Loss is {:.3e} <= {:.3e}".format(avg_total_loss, self.config.min_avg_loss))
            # else:
            #     print("Loss is {:.3e} > {:.3e}".format(avg_total_loss, self.config.min_avg_loss))

            if self.config.DEBUG_TRAINING_DURATION: # and int(math.floor(batch_progress)) % 25 == 0:
                print("Current Batch {:d} {:d} {:2.1%} {:.2f} s Avg {:.2f} s/batch Loss {:.3e} Remaining {:s}".format(
                    current_batch,
                    total_batch,
                    float(current_batch / total_batch),
                    batch_duration,
                    speed,
                    avg_total_loss,
                    time.strftime('Days %d Time %H:%M:%S', time.gmtime(remaining_sec)))
                )

            if current_batch > samples:
                break

            print(np_batch_bench.shape)
            print(current_batch)
            np_batch_bench[current_batch-1][0] = current_batch
            np_batch_bench[current_batch-1][1] = total_batch
            np_batch_bench[current_batch-1][2] = batch_duration
            np_batch_bench[current_batch-1][3] = speed



            current_batch += 1




        # save mean of q_dist_list into bigger array
        mean = np.mean(np.asarray(q_dist_list))
        # print("Q mean {:3.2f} deg".format(mean))
        mean_t = np.mean(np.asarray(trans_list))
        mean_joints = np.mean(np.asarray(joint_list))

        self.trans_mean.append([self.iter, mean_t])
        self.q_dist_mean.append([self.iter, mean])
        self.joints_mean.append([self.iter, mean_joints])
        self.iter += 1

        # update logging dict
        self.logging_dict["learning_rate"].append(lr)
        self.logging_dict["train_loss"].append(train_loss.val)
        self.logging_dict["train_err_rotation"].append(train_err_rotation.val)
        self.logging_dict["train_err_translation"].append(train_err_translation.val)
        self.logging_dict["train_err_joints"].append(train_err_joints.val)



        # print progress
        progress = float((self.current_epoch + 1) / self.config.max_epoch)
        duration_epoch = time.time() - tic
        if self.current_epoch % self.config.display_step == 0 or self.current_epoch % 1 == 0:
            self.duration = time.time() - self.start_time
            self.logger.info(
                "Train Epoch: {:>4d} | Total: {:>4d} | Progress: {:>3.2%} | Loss: {:>3.2e} | Translation [mm]: {:>3.2e} |"
                " Rotation [deg] {:>3.2e} | Joints [deg] {:>3.2e} | ({:02d}:{:02d}:{:02d}) ".format(
                    self.current_epoch + 1,
                    self.config.max_epoch,
                    progress,
                    train_loss.val,
                    train_err_translation.val,
                    train_err_rotation.val,
                    train_err_joints.val,
                    int(self.duration / 3600),
                    int(np.mod(self.duration, 3600) / 60),
                    int(np.mod(np.mod(self.duration, 3600), 60))) + time.strftime("%d.%m.%y %H:%M:%S",
                                                                                  time.localtime()))

        ds = pd.DataFrame(np_batch_bench)
        print(ds)
        path = "/home/speerponar/pytorch_models/evaluation/"
        ds.to_csv(path+"test_"+str(self.config.batch_size)+"w_"+str(self.config.data_loader_workers)+".csv")
        print("Save csv file")

    def validate(self):
        """
        One epoch validation
        :return:
        """
        # set the model in validation mode
        self.model.eval()
        val_losses = []

        valid_loss = AverageMeter()
        valid_err_translation = AverageMeter()
        valid_err_rotation = AverageMeter()
        valid_err_joints = AverageMeter()

        t = Transformation(config=self.config)

        val_tic = time.time()

        with torch.no_grad():
            for x, y in self.data_loader.valid_loader:
                # validate on gpu
                if self.cuda:
                    x = x.to(device=self.device, dtype=torch.float)
                    y = y.to(device=self.device, dtype=torch.float)

                # model
                pred = self.model(x.type(torch.FloatTensor))

                if self.config.data_output_type == "joints_absolute":
                    loss_joints = self.loss(pred, y)
                    total_loss = loss_joints
                    valid_loss.update(total_loss.item())
                    valid_err_joints.update(total_loss.item())
                elif self.config.data_output_type == "q_trans_simple":
                    loss_q_trans_simple = self.loss(pred, y)
                    total_loss = loss_q_trans_simple
                elif self.config.data_output_type == "joints_relative":
                    total_loss = self.loss(pred, y)
                    valid_loss.update(total_loss.item())
                    # print("Validation loss {:f}".format(total_loss.item()))
                elif self.config.data_output_type == "pose_relative":
                    # loss for rotation
                    # select rotation indices from the prediction tensor
                    indices = torch.tensor([3, 4, 5, 6])
                    indices = indices.to(self.device)
                    rotation = torch.index_select(pred, 1, indices)
                    # select rotation indices from the label tensor
                    y_rot = torch.index_select(y, 1, indices)
                    # calc MSE loss for rotation
                    # print("Rotation Pred", rotation[0])
                    # print("Rotation Label", y_rot[0])
                    # q_distance = pq.Quaternion.distance(rotation)

                    loss_rotation = self.loss(rotation, y_rot)
                    # penalty loss from facebook paper posenet
                    # penalty_loss = self.config.rot_reg * torch.mean((torch.sum(quater ** 2, dim=1) - 1) ** 2)
                    penalty_loss = 0

                    # loss for translation
                    # select translation indices from the prediction tensor
                    indices = torch.tensor([0, 1, 2])
                    indices = indices.to(self.device)
                    translation = torch.index_select(pred, 1, indices)
                    # select translation indices from the label tensor
                    y_trans = torch.index_select(y, 1, indices)

                    # calc MSE loss for translation
                    loss_translation = self.loss(translation, y_trans)
                    total_loss = penalty_loss + loss_rotation + loss_translation

                    q_pred = pq.Quaternion(rotation[0].cpu().detach().numpy())
                    q_rot = pq.Quaternion(y_rot[0].cpu().detach().numpy())
                    q_dist = math.degrees(pq.Quaternion.distance(q_pred, q_rot))
                    valid_err_rotation.update(q_dist)

                    # loss for translation
                    # select translation indices from the prediction tensor
                    indices = torch.tensor([0, 1, 2])
                    indices = indices.to(self.device)
                    translation = torch.index_select(pred, 1, indices)
                    # select translation indices from the label tensor
                    y_trans = torch.index_select(y, 1, indices)

                    trans_pred = translation[0].cpu().detach().numpy()
                    trans_label = y_trans[0].cpu().detach().numpy()

                    # calc translation MSE
                    mse_trans = (np.square(trans_pred - trans_label)).mean()
                    valid_err_translation.update(mse_trans)

                    # use simple loss
                    total_loss = self.loss(pred, y)
                    valid_loss.update(total_loss.item())

                elif self.config.data_output_type == "pose_absolute":
                    # select rotation indices from the prediction tensor
                    indices = torch.tensor([3, 4, 5, 6])
                    indices = indices.to(self.device)
                    rotation = torch.index_select(pred, 1, indices)
                    # select rotation indices from the label tensor
                    y_rot = torch.index_select(y, 1, indices)
                    q_pred = pq.Quaternion(rotation[0].cpu().detach().numpy())
                    q_rot = pq.Quaternion(y_rot[0].cpu().detach().numpy())
                    q_dist = math.degrees(pq.Quaternion.distance(q_pred, q_rot))
                    valid_err_rotation.update(q_dist)

                    # loss for translation
                    # select translation indices from the prediction tensor
                    indices = torch.tensor([0, 1, 2])
                    indices = indices.to(self.device)
                    translation = torch.index_select(pred, 1, indices)
                    # select translation indices from the label tensor
                    y_trans = torch.index_select(y, 1, indices)

                    trans_pred = translation[0].cpu().detach().numpy()
                    trans_label = y_trans[0].cpu().detach().numpy()

                    # calc translation MSE
                    mse_trans = (np.square(trans_pred - trans_label)).mean()
                    valid_err_translation.update(mse_trans)

                    # use simple loss
                    total_loss = self.loss(pred, y)
                    valid_loss.update(total_loss.item())
                else:
                    raise Exception("Wrong data output type chosen.")

                if np.isnan(float(total_loss.item())):
                    raise ValueError('Loss is nan during Validation.')
                val_losses.append(total_loss.item())


        # update logging dict
        # self.logging_dict["loss_validation_mse"].append(np.mean(val_losses))

        self.logging_dict["valid_loss"].append(valid_loss.val)
        self.logging_dict["valid_err_rotation"].append(valid_err_rotation.val)
        self.logging_dict["valid_err_translation"].append(valid_err_translation.val)
        self.logging_dict["valid_err_joints"].append(valid_err_joints.val)

        progress = float((self.current_epoch + 1) / self.config.max_epoch)
        val_duration = time.time() - val_tic
        self.logger.info(
            "Valid Epoch: {:>4d} | Total: {:>4d} | Progress: {:2.2%} | Loss: {:>3.2e} | Translation [mm]: {:>3.2e} |"
            " Rotation [deg] {:>3.2e} | Joints [deg] {:>3.2e} ({:02d}:{:02d}:{:02d}) ".format(
                self.current_epoch + 1,
                self.config.max_epoch,
                progress,
                valid_loss.val,
                valid_err_translation.val,
                valid_err_rotation.val,
                valid_err_joints.val,
                int(val_duration / 3600),
                int(np.mod(val_duration, 3600) / 60),
                int(np.mod(np.mod(val_duration, 3600), 60))) + time.strftime("%d.%m.%y %H:%M:%S",
                                                                              time.localtime()))
        return np.mean(val_losses)

    def finalize(self):
        """
        Finalize all the operations of the 2 Main classes of the process the operator and the data loader
        :return:
        """
        print("Please wait while finalizing the operation.. Thank you")
        self.save_checkpoint()
        self.data_loader.finalize()
