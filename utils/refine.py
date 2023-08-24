import sys

from models import create_model
import glob

import os
from utils import utils
from tqdm import tqdm
import math
import torch.nn.functional as F

import numpy as np
import pickle as pkl
import torch


def get_yaw_pitch(grasp_approach_vector_normalize, yaw_resolution=8, pitch_resolution=4):
    vx = grasp_approach_vector_normalize[:, 0].cpu().numpy()
    vy = grasp_approach_vector_normalize[:, 1].cpu().numpy()
    vz = grasp_approach_vector_normalize[:, 2].cpu().numpy()

    r_ap_pitch = np.arccos(vz)  # 2.69, 2.949, 2.8807
    # r_ap_yaw = np.atan2(vy, vx) + math.pi
    r_ap_yaw = np.arctan2(vy, vx) + np.pi

    yaw_label = np.floor(r_ap_yaw / (2 * np.pi / yaw_resolution))  # 6.0979 0.04 6.00 5.48 5.359
    pitch_label = np.floor(r_ap_pitch / (np.pi / pitch_resolution))
    return yaw_label, pitch_label


def load_network_from_file(checkpoint_folder, gpu_ids, checkpoints_dir=None, evaluate=True):
    network_arguments = utils.read_checkpoint_args(checkpoint_folder)
    network_arguments.is_train = False

    if checkpoints_dir != None:
        network_arguments.checkpoints_dir = checkpoints_dir

    if evaluate is True:
        network_arguments.gpu_ids = [gpu_ids]
    else:
        network_arguments.gpu_ids[0] = gpu_ids
    model = create_model(network_arguments)
    return model

class RefineNN:
    '''
    Do refinement with euler angle representation instead of qt representation

    '''

    def __init__(self, grasp_scoring_network, batch_max_size=200):
        self.grasp_scoring_network = grasp_scoring_network
        self.batch_max_size = batch_max_size
        self.pc = None
        self.device = torch.device('cuda:0')

    def improve_grasps_sampling_based_unconstrained(self,
                                                    pcs,
                                                    grasp_qt,
                                                    num_refine_steps, delta_translation=0.02):

        batch_size = grasp_qt.shape[0]
        object_pc_numpy = np.tile(pcs, (batch_size, 1, 1))
        pcs = torch.from_numpy(object_pc_numpy).cuda().to(torch.float32)
        grasp_qt = torch.from_numpy(grasp_qt).cuda().to(torch.float32)

        # pcs = np.asarray(pcs.points, dtype=np.float32)
        # object_pc_numpy = np.tile(pcs, (batch_size, 1, 1))
        # pcs = torch.from_numpy(object_pc_numpy).cuda().to(torch.float32)
        # grasp_qt = torch.from_numpy(grasp_qt).cuda().to(torch.float32)

        with torch.no_grad():
            grasp_seq = []
            score_seq = []
            grasps = grasp_qt.clone()
            grasp_pcs = utils.transform_control_points(grasps,
                                                       grasp_qt.shape[0],
                                                       device=self.device)
            last_success = self.grasp_scoring_network.evaluate_grasps(pcs, grasp_pcs)
            grasp_seq.append(grasp_qt.cpu().numpy())
            score_seq.append(last_success.cpu().numpy())

            # last_success = torch.sigmoid(last_success)
            for _ in range(num_refine_steps):
                delta_t = 2 * (
                        torch.rand(grasps[:, 4:].shape).to(self.device) - 0.5)
                delta_t *= delta_translation
                rand_w = 2 * (torch.rand(grasps[:, :3].shape).to(self.device) -
                              0.5)
                rand_w *= 0.1
                norm_w = torch.norm(rand_w, p=2, dim=-1).to(self.device)
                exp_w = torch.zeros((grasps.shape[0], 4)).to(self.device)
                exp_w[:, 0] = torch.cos(norm_w)
                exp_w[:,
                1:] = (1 / norm_w * torch.sin(norm_w))[:, None] * rand_w
                perturbed_qt = grasps.clone()
                perturbed_qt[:, 4:] += delta_t
                perturbed_qt[:, :4] = utils.quaternion_mult(exp_w, perturbed_qt[:, :4],
                                                            numpy=False)
                grasp_pcs = utils.transform_control_points(
                    perturbed_qt, perturbed_qt.shape[0], device=self.device)

                perturbed_success = self.grasp_scoring_network.evaluate_grasps(pcs, grasp_pcs)
                # perturbed_success = torch.sigmoid(perturbed_success)
                ratio = perturbed_success / torch.max(
                    last_success,
                    torch.tensor(0.0001).to(self.device))
                mask = torch.rand(ratio.shape).to(self.device) <= ratio

                ind = torch.where(mask)[0]
                last_success[ind] = perturbed_success[ind]
                grasps.data[ind] = perturbed_qt.data[ind]

                grasp_seq.append(grasps.cpu().numpy())
                score_seq.append(last_success.cpu().numpy())

            grasp_seq = np.array(grasp_seq)
            score_seq = np.array(score_seq)

            ind_list = np.argmax(score_seq, axis=0)

            rows, cols = np.indices(ind_list.shape)
            last_success = score_seq[ind_list, rows, cols]
            grasps = grasp_seq[ind_list, rows].squeeze(axis=1)

            grasps[:, :4] = utils.flip(grasps[:, :4])
            return grasps, last_success.squeeze()
            # return grasp_seq, score_seq.squeeze()

    def improve_grasps_sampling_based_constrained(self,
                                                  pcs,
                                                  grasp_qt, grasp_app_condition, grasp_angle,
                                                  num_refine_steps, delta_translation=0.02):
        # pcs: (1024,3) numpy
        # grasp_qt: (batchsize, 7) numpy
        # grasp_app_condition: (3,) numpy

        batch_size = grasp_qt.shape[0]
        object_pc_numpy = np.tile(pcs, (batch_size, 1, 1))
        pcs = torch.from_numpy(object_pc_numpy).cuda().to(torch.float32)
        grasp_qt = torch.from_numpy(grasp_qt).cuda().to(torch.float32)
        grasp_app_condition = torch.tensor(np.tile(grasp_app_condition, (batch_size, 1, 1))).cuda().to(torch.float32)

        with torch.no_grad():
            grasp_seq = []
            score_seq = []
            grasps = grasp_qt.clone()
            grasp_pcs = utils.transform_control_points(grasps,
                                                       grasp_qt.shape[0],
                                                       device=self.device)
            last_success = self.grasp_scoring_network.evaluate_grasps(pcs, grasp_pcs)
            grasp_seq.append(grasp_qt.cpu().numpy())
            score_seq.append(last_success.cpu().numpy())

            # last_success = torch.sigmoid(last_success)
            for _ in range(num_refine_steps):
                delta_t = 2 * (
                        torch.rand(grasps[:, 4:].shape).to(self.device) - 0.5)
                delta_t *= delta_translation
                rand_w = 2 * (torch.rand(grasps[:, :3].shape).to(self.device) -
                              0.5)
                rand_w *= 0.1
                norm_w = torch.norm(rand_w, p=2, dim=-1).to(self.device)
                exp_w = torch.zeros((grasps.shape[0], 4)).to(self.device)
                exp_w[:, 0] = torch.cos(norm_w)
                exp_w[:,
                1:] = (1 / norm_w * torch.sin(norm_w))[:, None] * rand_w
                perturbed_qt = grasps.clone()
                perturbed_qt[:, 4:] += delta_t
                perturbed_qt[:, :4] = utils.quaternion_mult(exp_w, perturbed_qt[:, :4],
                                                            numpy=False)
                grasp_pcs = utils.transform_control_points(
                    perturbed_qt, perturbed_qt.shape[0], device=self.device)

                perturbed_success = self.grasp_scoring_network.evaluate_grasps(pcs, grasp_pcs)

                constrained_success = self.evaluate_grasps_constrain(grasp_pcs, grasp_app_condition, grasp_angle)


                perturbed_success = perturbed_success * constrained_success
                # perturbed_success = torch.sigmoid(perturbed_success)
                ratio = perturbed_success / torch.max(
                    last_success,
                    torch.tensor(0.0001).to(self.device))
                mask = torch.rand(ratio.shape).to(self.device) <= ratio

                ind = torch.where(mask)[0]
                last_success[ind] = perturbed_success[ind]
                grasps.data[ind] = perturbed_qt.data[ind]

                grasp_seq.append(grasps.cpu().numpy())

                score_seq.append(last_success.cpu().numpy())

            grasp_seq = np.array(grasp_seq)
            score_seq = np.array(score_seq)
            print(grasp_seq.shape)

            ind_list = np.argmax(score_seq, axis=0)

            rows, cols = np.indices(ind_list.shape)
            last_success = score_seq[ind_list, rows, cols]
            grasps = grasp_seq[ind_list, rows].squeeze(axis=1)

            grasps[:, :4] = utils.flip(grasps[:, :4])
            return grasps, last_success.squeeze()

    def evaluate_grasps_constrain(self, grasps_pc, grasp_app_condition, grasp_angle):
        grasp_app = (grasps_pc[:, 2] + grasps_pc[:, 3]) / 2 - grasps_pc[:, 0]
        grasp_app_normalize = grasp_app / torch.linalg.norm(grasp_app, axis=1).reshape(-1, 1)

        grasp_app_condition = grasp_app_condition.squeeze()
        cosscore = F.cosine_similarity(grasp_app_condition, grasp_app_normalize)
        grasp_angle_thres = math.cos(math.radians(grasp_angle))
        accept_probability = cosscore > grasp_angle_thres

        return accept_probability.float().reshape(grasps_pc.shape[0], 1)