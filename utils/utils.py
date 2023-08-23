import numpy as np
import os
import sys
import math
import trimesh.transformations as tra
# from constrained_6dof_graspnet.utils import sample

# sys.path.append("/media/zehang/LaCie/zehang/ubuntu/project/GoNet-X/constrained_6dof_graspnet")
from utils import sample
# from constrained_6dof_graspnet.utils import sample
import torch
import yaml
from easydict import EasyDict as edict
import h5py
import open3d as o3d
from scipy.spatial.transform import Rotation as R
from scipy.spatial import distance_matrix
import random
import pickle
import os
import copy

gripper_model_path = os.path.dirname(os.path.realpath(__file__)) + "/../"

GRIPPER_PC = np.load(gripper_model_path+'gripper_models/panda_pc.npy',
                     allow_pickle=True).item()['points']
GRIPPER_PC[:, 3] = 1.

seed = 0

# orien_label_dict = {
#     0: [0,0],
#     2: [1,0],
#     4: [2,0],
#     6: [3,0],
#     1: [0,1],
#     3: [1,1],
#     5: [2,1],
#     7: [3,1]
# }

orien_label_dict = {
    0: [0,0],
    1: [1,0],
    2: [2,0],
    3: [-1,-1],
    4: [-1,-1],
    5: [-1,-1],
    6: [-1,-1],
    7: [-1,-1]
}

def farthest_points(data,
                    nclusters,
                    dist_func,
                    return_center_indexes=False,
                    return_distances=False,
                    verbose=False):
    """
      Performs farthest point sampling on data points.
      Args:
        data: numpy array of the data points.
        nclusters: int, number of clusters.
        dist_dunc: distance function that is used to compare two data points.
        return_center_indexes: bool, If True, returns the indexes of the center of 
          clusters.
        return_distances: bool, If True, return distances of each point from centers.

      Returns clusters, [centers, distances]:
        clusters: numpy array containing the cluster index for each element in 
          data.
        centers: numpy array containing the integer index of each center.
        distances: numpy array of [npoints] that contains the closest distance of 
          each point to any of the cluster centers.
    """
    if nclusters >= data.shape[0]:
        if return_center_indexes:
            return np.arange(data.shape[0],
                             dtype=np.int32), np.arange(data.shape[0],
                                                        dtype=np.int32)

        return np.arange(data.shape[0], dtype=np.int32)

    clusters = np.ones((data.shape[0], ), dtype=np.int32) * -1
    distances = np.ones((data.shape[0], ), dtype=np.float32) * 1e7
    centers = []
    for iter in range(nclusters):
        index = np.argmax(distances)
        centers.append(index)
        shape = list(data.shape)
        for i in range(1, len(shape)):
            shape[i] = 1

        broadcasted_data = np.tile(np.expand_dims(data[index], 0), shape)
        new_distances = dist_func(broadcasted_data, data)
        distances = np.minimum(distances, new_distances)
        clusters[distances == new_distances] = iter
        if verbose:
            print('farthest points max distance : {}'.format(
                np.max(distances)))

    if return_center_indexes:
        if return_distances:
            return clusters, np.asarray(centers, dtype=np.int32), distances
        return clusters, np.asarray(centers, dtype=np.int32)

    return clusters


def distance_by_translation_grasp(p1, p2):
    """
      Gets two nx4x4 numpy arrays and computes the translation of all the
      grasps.
    """
    t1 = p1[:, :3, 3]
    t2 = p2[:, :3, 3]
    return np.sqrt(np.sum(np.square(t1 - t2), axis=-1))


def distance_by_translation_point(p1, p2):
    """
      Gets two nx3 points and computes the distance between point p1 and p2.
    """
    return np.sqrt(np.sum(np.square(p1 - p2), axis=-1))


def regularize_pc_point_count(pc, npoints, use_farthest_point=False):
    """
      If point cloud pc has less points than npoints, it oversamples.
      Otherwise, it downsample the input pc to have npoint points.
      use_farthest_point: indicates whether to use farthest point sampling
      to downsample the points. Farthest point sampling version runs slower.
    """

    # np.random.seed(seed)
    # random.seed(seed)
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    if pc.shape[0] > npoints:
        if use_farthest_point:
            _, center_indexes = farthest_points(pc,
                                                npoints,
                                                distance_by_translation_point,
                                                return_center_indexes=True)
        else:
            center_indexes = np.random.choice(range(pc.shape[0]),
                                              size=npoints,
                                              replace=False)
        pc = pc[center_indexes, :]
    else:
        required = npoints - pc.shape[0]
        if required > 0:
            index = np.random.choice(range(pc.shape[0]), size=required)
            pc = np.concatenate((pc, pc[index, :]), axis=0)
    return pc


def perturb_grasp(grasp, num, min_translation, max_translation, min_rotation,
                  max_rotation):
    """
      Self explanatory.
    """
    output_grasps = []
    for _ in range(num):
        sampled_translation = [
            np.random.uniform(lb, ub)
            for lb, ub in zip(min_translation, max_translation)
        ]
        sampled_rotation = [
            np.random.uniform(lb, ub)
            for lb, ub in zip(min_rotation, max_rotation)
        ]
        grasp_transformation = tra.euler_matrix(*sampled_rotation)
        grasp_transformation[:3, 3] = sampled_translation
        output_grasps.append(np.matmul(grasp, grasp_transformation))

    return output_grasps


def evaluate_grasps(grasp_tfs, obj_mesh):
    """
        Check the collision of the grasps and also heuristic quality for each
        grasp.
    """
    collisions, _ = sample.in_collision_with_gripper(
        obj_mesh,
        grasp_tfs,
        gripper_name='panda',
        silent=True,
    )
    qualities = sample.grasp_quality_point_contacts(
        grasp_tfs,
        collisions,
        object_mesh=obj_mesh,
        gripper_name='panda',
        silent=True,
    )

    return np.asarray(collisions), np.asarray(qualities)


def inverse_transform(trans):
    """
      Computes the inverse of 4x4 transform.
    """
    rot = trans[:3, :3]
    t = trans[:3, 3]
    rot = np.transpose(rot)
    t = -np.matmul(rot, t)
    output = np.zeros((4, 4), dtype=np.float32)
    output[3][3] = 1
    output[:3, :3] = rot
    output[:3, 3] = t

    return output


def uniform_quaternions():
    quaternions = [
        l[:-1].split('\t') for l in open(
            '../uniform_quaternions/data2_4608.qua', 'r').readlines()
    ]

    quaternions = [[float(t[0]),
                    float(t[1]),
                    float(t[2]),
                    float(t[3])] for t in quaternions]
    quaternions = np.asarray(quaternions)
    quaternions = np.roll(quaternions, 1, axis=1)
    return [tra.quaternion_matrix(q) for q in quaternions]


def nonuniform_quaternions():
    all_poses = []
    for az in np.linspace(0, np.pi * 2, 30):
        for el in np.linspace(-np.pi / 2, np.pi / 2, 30):
            all_poses.append(tra.euler_matrix(el, az, 0))
    return all_poses


def print_network(net):
    """Print the total number of parameters in the network
    Parameters:
        network
    """
    print('---------- Network initialized -------------')
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('[Network] Total number of parameters : %.3f M' % (num_params / 1e6))
    print('-----------------------------------------------')


def merge_pc_and_gripper_pc(pc,
                            gripper_pc,
                            instance_mode=0,
                            pc_latent=None,
                            gripper_pc_latent=None):
    """
    Merges the object point cloud and gripper point cloud and
    adds a binary auxilary feature that indicates whether each point
    belongs to the object or to the gripper.
    """

    pc_shape = pc.shape
    gripper_shape = gripper_pc.shape
    assert (len(pc_shape) == 3)
    assert (len(gripper_shape) == 3)
    assert (pc_shape[0] == gripper_shape[0])

    npoints = pc.shape[1]
    batch_size = pc.shape[0]

    if instance_mode == 1:
        assert pc_shape[-1] == 3
        latent_dist = [pc_latent, gripper_pc_latent]
        latent_dist = torch.cat(latent_dist, 1)

    l0_xyz = torch.cat((pc, gripper_pc), 1)
    labels = [
        torch.ones((pc.shape[1], 1), dtype=torch.float32),
        torch.zeros((gripper_pc.shape[1], 1), dtype=torch.float32)
    ]
    labels = torch.cat(labels, 0)
    labels = torch.expand_dims(labels, 0)
    labels = torch.tile(labels, [batch_size, 1, 1])

    if instance_mode == 1:
        l0_points = torch.cat([l0_xyz, latent_dist, labels], -1)
    else:
        l0_points = torch.cat([l0_xyz, labels], -1)

    return l0_xyz, l0_points


def get_gripper_pc(batch_size, npoints, use_torch=True):
    """
      Returns a numpy array or a tensor of shape (batch_size x npoints x 4).
      Represents gripper with the sepcified number of points.
      use_tf: switches between output tensor or numpy array.
    """
    output = np.copy(GRIPPER_PC)
    if npoints != -1:
        assert (npoints > 0 and npoints <= output.shape[0]
                ), 'gripper_pc_npoint is too large {} > {}'.format(
                    npoints, output.shape[0])
        output = output[:npoints]
        output = np.expand_dims(output, 0)
    else:
        raise ValueError('npoints should not be -1.')

    if use_torch:
        output = torch.tensor(output, torch.float32)
        output = output.repeat(batch, size, 1, 1)
        return output
    else:
        output = np.tile(output, [batch_size, 1, 1])

    return output


def create_ranges(start, stop, N, endpoint=True):
    if endpoint == 1:
        divisor = N - 1
    else:
        divisor = N
    steps = (1.0 / divisor) * (stop - start)
    return steps[:, None] * np.arange(N) + start[:, None]

def get_equidistant_points(p1, p2, parts=10):
    try:
        p_dense = np.linspace((p1[0], p1[1], p1[2]), (p2[0], p2[1], p2[2]), parts+1)
    except:
        p_dense = create_ranges(p1, p2, parts+1).T
    return p_dense

def get_control_point_tensor(batch_size, use_torch=True, dense=False, device="cpu"):
    """
      Outputs a tensor of shape (batch_size x 6 x 3).
      use_tf: switches between outputing a tensor and outputing a numpy array.
    """
    control_points = np.load(gripper_model_path+'gripper_control_points/panda.npy')[:, :3]
    if dense:
        points_between_fingers = get_equidistant_points(control_points[0, :], control_points[1, :])
        points_from_base_to_gripper = get_equidistant_points(np.array([0, 0, 0]), (control_points[0, :]+control_points[1, :])/2.0)
        control_points = np.concatenate(([[0, 0, 0], [0, 0, 0]], points_between_fingers, points_from_base_to_gripper, control_points))

    else:
        control_points = [[0, 0, 0], [0, 0, 0], control_points[0, :],
                          control_points[1, :], control_points[-2, :],
                          control_points[-1, :]]
    control_points = np.asarray(control_points, dtype=np.float32)
    control_points = np.tile(np.expand_dims(control_points, 0),
                             [batch_size, 1, 1])

    if use_torch:
        return torch.tensor(control_points).to(device)

    return control_points


def transform_control_points(gt_grasps, batch_size, mode='qt', device="cpu", dense=False):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is catenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert (mode == 'qt' or mode == 'rt'), mode
    grasp_shape = gt_grasps.shape
    if mode == 'qt':
        assert (len(grasp_shape) == 2), grasp_shape
        assert (grasp_shape[-1] == 7), grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device, dense=dense)
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps

        gt_grasps = torch.unsqueeze(input_gt_grasps,
                                    1).repeat(1, num_control_points, 1)

        gt_q = gt_grasps[:, :, :4]
        gt_t = gt_grasps[:, :, 4:]
        gt_control_points = qrot(gt_q, control_points)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert (len(grasp_shape) == 3), grasp_shape
        assert (grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        control_points = get_control_point_tensor(batch_size, device=device, dense=dense)
        shape = control_points.shape
        ones = torch.ones((shape[0], shape[1], 1), dtype=torch.float32)
        control_points = torch.cat((control_points, ones), -1)
        return torch.matmul(control_points, gt_grasps.permute(0, 2, 1))


def transform_control_points_numpy(gt_grasps, batch_size, mode='qt', dense=False):
    """
      Transforms canonical points using gt_grasps.
      mode = 'qt' expects gt_grasps to have (batch_size x 7) where each 
        element is a concatenation of quaternion and translation for each
        grasps.
      mode = 'rt': expects to have shape (batch_size x 4 x 4) where
        each element is 4x4 transformation matrix of each grasp.
    """
    assert (mode == 'qt' or mode == 'rt'), mode
    grasp_shape = gt_grasps.shape
    if mode == 'qt':
        assert (len(grasp_shape) == 2), grasp_shape
        assert (grasp_shape[-1] == 7), grasp_shape
        control_points = get_control_point_tensor(batch_size, use_torch=False, dense=dense)
        num_control_points = control_points.shape[1]
        input_gt_grasps = gt_grasps
        gt_grasps = np.expand_dims(input_gt_grasps,
                                   1).repeat(num_control_points, axis=1)
        gt_q = gt_grasps[:, :, :4]
        gt_t = gt_grasps[:, :, 4:]

        gt_control_points = rotate_point_by_quaternion(control_points, gt_q, numpy=True)
        gt_control_points += gt_t

        return gt_control_points
    else:
        assert (len(grasp_shape) == 3), grasp_shape
        assert (grasp_shape[1] == 4 and grasp_shape[2] == 4), grasp_shape
        control_points = get_control_point_tensor(batch_size, use_torch=False, dense=dense)
        shape = control_points.shape
        ones = np.ones((shape[0], shape[1], 1), dtype=np.float32)
        control_points = np.concatenate((control_points, ones), -1)
        return np.matmul(control_points, np.transpose(gt_grasps, (0, 2, 1)))[:, :, :3]


def quaternion_mult(q, r, numpy=False):
    """
    Multiply quaternion(s) q with quaternion(s) r.
    Expects two equally-sized tensors of shape (*, 4), where * denotes any number of dimensions.
    Returns q*r as a tensor of shape (*, 4).
    """
    assert q.shape[-1] == 4
    assert r.shape[-1] == 4

    original_shape = q.shape

    if numpy:
        terms = np.matmul(r.reshape(-1, 4, 1), q.reshape(-1, 1, 4))
    else:
        # Compute outer product
        terms = torch.bmm(r.view(-1, 4, 1), q.view(-1, 1, 4))
    w = terms[:, 0, 0] - terms[:, 1, 1] - terms[:, 2, 2] - terms[:, 3, 3]
    x = terms[:, 0, 1] + terms[:, 1, 0] - terms[:, 2, 3] + terms[:, 3, 2]
    y = terms[:, 0, 2] + terms[:, 1, 3] + terms[:, 2, 0] - terms[:, 3, 1]
    z = terms[:, 0, 3] - terms[:, 1, 2] + terms[:, 2, 1] + terms[:, 3, 0]
    if numpy:
        return np.stack((w, x, y, z), axis=1).reshape(original_shape)
    else:
        return torch.stack((w, x, y, z), dim=1).view(original_shape)


def conj_quaternion(q):
    """
      Conjugate of quaternion q.
    """
    if torch.is_tensor(q):
        q_conj = q.clone()
    else:
        q_conj = np.copy(q)
    q_conj[:, :, 1:] *= -1
    return q_conj


def rotate_point_by_quaternion(point, q, device="cpu", numpy=False):
    """
      Takes in points with shape of (batch_size x n x 3) and quaternions with
      shape of (batch_size x n x 4) and returns a tensor with shape of 
      (batch_size x n x 3) which is the rotation of the point with quaternion
      q. 
    """
    shape = point.shape
    q_shape = q.shape

    if len(shape) == 2:
        if numpy:
            point = np.expand_dims(point, 0)
        else:
            point = point.unsqueeze(0)
        shape = point.shape

    if len(q_shape) == 2:
        if numpy:
            q = np.expand_dims(q, 0)
            q = np.repeat(q, point.shape[1], axis=1)
        else:
            q = q.unsqueeze(0)
            q = q.repeat([1, point.shape[1], 1])
        q_shape = q.shape

    assert (len(shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (shape[-1] == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (len(q_shape) == 3), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[-1] == 4), 'point shape = {} q shape = {}'.format(
        shape, q_shape)
    assert (q_shape[1] == shape[1]), 'point shape = {} q shape = {}'.format(
        shape, q_shape)

    q_conj = conj_quaternion(q)
    if numpy:
        r = np.concatenate([
            np.zeros(
                (shape[0], shape[1], 1), dtype=point.dtype), point
        ],
            axis=-1)

    else:
        r = torch.cat([
            torch.zeros(
                (shape[0], shape[1], 1), dtype=point.dtype).to(device), point
        ],
            dim=-1)
    final_point = quaternion_mult(quaternion_mult(q, r, numpy), q_conj, numpy)
    final_output = final_point[:, :,
                               1:]  # torch.slice(final_point, [0, 0, 1], shape)
    return final_output


def tc_rotation_matrix(az, el, th, batched=False):
    if batched:

        cx = torch.cos(torch.reshape(az, [-1, 1]))
        cy = torch.cos(torch.reshape(el, [-1, 1]))
        cz = torch.cos(torch.reshape(th, [-1, 1]))
        sx = torch.sin(torch.reshape(az, [-1, 1]))
        sy = torch.sin(torch.reshape(el, [-1, 1]))
        sz = torch.sin(torch.reshape(th, [-1, 1]))

        ones = torch.ones_like(cx)
        zeros = torch.zeros_like(cx)

        rx = torch.cat([ones, zeros, zeros, zeros, cx, -sx, zeros, sx, cx],
                       dim=-1)
        ry = torch.cat([cy, zeros, sy, zeros, ones, zeros, -sy, zeros, cy],
                       dim=-1)
        rz = torch.cat([cz, -sz, zeros, sz, cz, zeros, zeros, zeros, ones],
                       dim=-1)

        rx = torch.reshape(rx, [-1, 3, 3])
        ry = torch.reshape(ry, [-1, 3, 3])
        rz = torch.reshape(rz, [-1, 3, 3])

        return torch.matmul(rz, torch.matmul(ry, rx))
    else:
        cx = torch.cos(az)
        cy = torch.cos(el)
        cz = torch.cos(th)
        sx = torch.sin(az)
        sy = torch.sin(el)
        sz = torch.sin(th)

        rx = torch.stack([[1., 0., 0.], [0, cx, -sx], [0, sx, cx]], dim=0)
        ry = torch.stack([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dim=0)
        rz = torch.stack([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dim=0)

        return torch.matmul(rz, torch.matmul(ry, rx))


def mkdir(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def control_points_from_rot_and_trans(grasp_eulers,
                                      grasp_translations,
                                      device="cpu"):
    rot = tc_rotation_matrix(grasp_eulers[:, 0],
                             grasp_eulers[:, 1],
                             grasp_eulers[:, 2],
                             batched=True)
    grasp_pc = get_control_point_tensor(grasp_eulers.shape[0], device=device)
    grasp_pc = torch.matmul(grasp_pc, rot.permute(0, 2, 1))
    grasp_pc += grasp_translations.unsqueeze(1).expand(-1, grasp_pc.shape[1],
                                                       -1)
    return grasp_pc


def rot_and_trans_to_grasps(euler_angles, translations, selection_mask):
    grasps = []
    refine_indexes, sample_indexes = np.where(selection_mask)
    for refine_index, sample_index in zip(refine_indexes, sample_indexes):
        rt = tra.euler_matrix(*euler_angles[refine_index, sample_index, :])
        rt[:3, 3] = translations[refine_index, sample_index, :]
        grasps.append(rt)
    return grasps


def convert_qt_to_rt(grasps_qt, order="zyx"):
    grasps = copy.deepcopy(grasps_qt)
    Ts = grasps[:, 4:]
    Rs = qeuler(grasps[:, :4], order)
    return Rs, Ts


def convert_rt_to_qt(R_input, T_input, device):
    R = copy.deepcopy(R_input)
    T = copy.deepcopy(T_input)
    qt = torch.empty((R.shape[0], 7))
    qt[:, 4:] = T
    qt[:, :4] = quaternion_from_euler(R[:, 0], R[:, 1], R[:, 2], device)
    return qt


def qeuler(q, order, epsilon=0):
    """
    Convert quaternion(s) q to Euler angles.
    Expects a tensor of shape (*, 4), where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4

    original_shape = list(q.shape)
    original_shape[-1] = 3
    q = q.view(-1, 4)

    q0 = q[:, 0]
    q1 = q[:, 1]
    q2 = q[:, 2]
    q3 = q[:, 3]

    if order == 'xyz':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q1 * q3 + q0 * q2), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    elif order == 'yzx':
        x = torch.atan2(2 * (q0 * q1 - q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q1 * q2 + q0 * q3), -1 + epsilon, 1 - epsilon))
    elif order == 'zxy':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 + q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q0 * q2 - q1 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q0 * q3 - q1 * q2), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'xzy':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
        y = torch.atan2(2 * (q0 * q2 + q1 * q3), 1 - 2 * (q2 * q2 + q3 * q3))
        z = torch.asin(
            torch.clamp(2 * (q0 * q3 - q1 * q2), -1 + epsilon, 1 - epsilon))
    elif order == 'yxz':
        x = torch.asin(
            torch.clamp(2 * (q0 * q1 - q2 * q3), -1 + epsilon, 1 - epsilon))
        y = torch.atan2(2 * (q1 * q3 + q0 * q2), 1 - 2 * (q1 * q1 + q2 * q2))
        z = torch.atan2(2 * (q1 * q2 + q0 * q3), 1 - 2 * (q1 * q1 + q3 * q3))
    elif order == 'zyx':
        x = torch.atan2(2 * (q0 * q1 + q2 * q3), 1 - 2 * (q1 * q1 + q2 * q2))
        y = torch.asin(
            torch.clamp(2 * (q0 * q2 - q1 * q3), -1 + epsilon, 1 - epsilon))
        z = torch.atan2(2 * (q0 * q3 + q1 * q2), 1 - 2 * (q2 * q2 + q3 * q3))
    else:
        raise ValueError("Invalid order " + order)

    return torch.stack((x, y, z), dim=1).view(original_shape)


def read_checkpoint_args(folder_path):
    return edict(yaml.safe_load(open(os.path.join(folder_path, 'opt.yaml'))))


def choose_grasps_better_than_threshold(eulers,
                                        translations,
                                        probs,
                                        threshold=0.7):
    """
      Chooses the grasps that have scores higher than the input threshold.
    """
    print('choose_better_than_threshold threshold=', threshold)
    return np.asarray(probs >= threshold, dtype=np.float32)


def choose_grasps_better_than_threshold_in_sequence(eulers,
                                                    translations,
                                                    probs,
                                                    threshold=0.7):
    """
      Chooses the grasps with the maximum score in the sequence of grasp refinements.
    """
    output = np.zeros(probs.shape, dtype=np.float32)
    max_index = np.argmax(probs, 0)
    max_value = np.max(probs, 0)
    for i in range(probs.shape[1]):
        if max_value[i] > threshold:
            output[max_index[i]][i] = 1.
    return output


def denormalize_grasps(grasps, mean=0, std=1):
    for grasp in grasps:
        grasp[:3, 3] = (std * grasp[:3, 3] + mean)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    Args:
        quat: first three coeff of quaternion of rotation. fourth is then computed to have a norm of 1 -- size = [B, 3]
    Returns:
        Rotation matrix corresponding to the quaternion -- size = [B, 3, 3]
    """
    norm_quat = torch.cat([quat[:, :1].detach() * 0 + 1, quat], dim=1)
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:,
                                                             2], norm_quat[:,
                                                                           3]
    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([
        w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz, 2 * wz + 2 * xy,
        w2 - x2 + y2 - z2, 2 * yz - 2 * wx, 2 * xz - 2 * wy, 2 * wx + 2 * yz,
        w2 - x2 - y2 + z2
    ],
        dim=1).reshape(B, 3, 3)
    return rotMat


def qrot(q, v):
    """
    Rotate vector(s) v about the rotation described by quaternion(s) q.
    Expects a tensor of shape (*, 4) for q and a tensor of shape (*, 3) for v,
    where * denotes any number of dimensions.
    Returns a tensor of shape (*, 3).
    """
    assert q.shape[-1] == 4
    assert v.shape[-1] == 3
    assert q.shape[:-1] == v.shape[:-1]

    original_shape = list(v.shape)
    q = q.view(-1, 4)
    v = v.view(-1, 3)

    qvec = q[:, 1:]
    uv = torch.cross(qvec, v, dim=1)
    uuv = torch.cross(qvec, uv, dim=1)
    return (v + 2 * (q[:, :1] * uv + uuv)).view(original_shape)


def get_inlier_grasp_indices(grasp_list, query_point, threshold=1.0, device="cpu"):
    """This function returns all grasps whose distance between the mid of the finger tips and the query point is less than the threshold value. 

    Arguments:
        grasps are given as a list of [B,7] where B is the number of grasps and the other
        7 values represent teh quaternion and translation.
        query_point is a 1x3 point in 3D space.
        threshold represents the maximum distance between a grasp and the query_point
    """
    indices_to_keep = []
    for grasps in grasp_list:
        grasp_cps = transform_control_points(grasps,
                                             grasps.shape[0],
                                             device=device)
        mid_points = get_mid_of_contact_points(grasp_cps)
        dist = torch.norm(mid_points - query_point, 2, dim=-1)
        indices_to_keep.append(torch.where(dist <= threshold))
    return indices_to_keep


def get_inlier_grasp_indices_with_control_points(grasps, query_point=torch.tensor([[0.0, 0.0, 0.0]]), threshold=0.4, device="cpu"):
    """This function returns all grasps whose distance between the mid of the finger tips and the query point is less than the threshold value. 

    Arguments:
        grasps are given as a tensor of [B,6,3] where B is the number of grasps, 6 is the number of control points, and 3 is their position
        in euclidean space.
        query_point is a 1x3 point in 3D space.
        threshold represents the maximum distance between a grasp and the query_point
    """
    grasp_centers = grasps.mean(1)
    query_point = query_point.to(device)
    distances = torch.cdist(grasp_centers, query_point)
    indices_with_distances_smaller_than_threshold = distances < threshold
    return grasps[indices_with_distances_smaller_than_threshold[:, 0]]


def get_mid_of_contact_points(grasp_cps):
    mid = (grasp_cps[:, 0, :] + grasp_cps[:, 1, :]) / 2.0
    return mid


def euclid_dist(point1, point2):
    return np.linalg.norm(point1 - point2)


def partition_array_into_subarrays(array, sub_array_size):
    subarrays = []
    for i in range(0, math.ceil(array.shape[0] / sub_array_size)):
        subarrays.append(array[i * sub_array_size:(i + 1) * sub_array_size])
    return subarrays


def read_h5_file(file):
    data = h5py.File(file, 'r')
    return data


def add_to_file(file, data):
    if not os.path.isfile(file):
        with open(file, 'w') as f:
            f.write(str(data) + "\n")
    else:
        with open(file, 'a+') as f:
            f.write(str(data) + "\n")


def create_gripper_marker(color=[0, 0, 1], tube_radius=0.001, sections=6):
    import trimesh
    """Create a 3D mesh visualizing a parallel yaw gripper. It consists of four cylinders.
    Args:
        color (list, optional): RGB values of marker. Defaults to [0, 0, 255].
        tube_radius (float, optional): Radius of cylinders. Defaults to 0.001.
        sections (int, optional): Number of sections of each cylinder. Defaults to 6.
    Returns:
        trimesh.Trimesh: A mesh that represents a simple parallel yaw gripper.
    """
    cfl = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        height=0,
        segment=[
            [4.10000000e-02, -7.27595772e-12, 6.59999996e-02],
            [4.10000000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cfr = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        height=0,
        segment=[
            [-4.100000e-02, -7.27595772e-12, 6.59999996e-02],
            [-4.100000e-02, -7.27595772e-12, 1.12169998e-01],
        ],
    )
    cb1 = trimesh.creation.cylinder(
        radius=0.002, sections=sections, segment=[[0, 0, 0], [0, 0, 6.59999996e-02]], height=0

    )
    cb2 = trimesh.creation.cylinder(
        radius=0.002,
        sections=sections,
        segment=[[-4.100000e-02, 0, 6.59999996e-02], [4.100000e-02, 0, 6.59999996e-02]],
        height=0
    )
    gripper_mesh = trimesh_to_open3d(cfl)
    gripper_mesh += trimesh_to_open3d(cfr)
    gripper_mesh += trimesh_to_open3d(cb1)
    gripper_mesh += trimesh_to_open3d(cb2)
    gripper_mesh = gripper_mesh.paint_uniform_color(color)

    return gripper_mesh


def trimesh_to_open3d(mesh):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices))
    o3d_mesh.triangles = o3d.utility.Vector3iVector(np.asarray(mesh.faces))
    return o3d_mesh


def find_neighboring_points(points, query_points, distance_threshold_for_neighbors=0.01, max_num_neighbors=1024):
    distances = euclid_dist_non_equal_length(points, query_points)
    indexes = distances < distance_threshold_for_neighbors
    neighbors_per_cluster = []
    for i in range(indexes.shape[0]):
        neighbors = np.argwhere(indexes[i]).squeeze(axis=-1)
        if neighbors.size > max_num_neighbors:
            center_indexes = np.random.choice(range(neighbors.shape[0]),
                                              size=max_num_neighbors,
                                              replace=False)
            neighbors = neighbors[center_indexes]
        neighbors_per_cluster.append(neighbors)
    return neighbors_per_cluster


def farthest_point_sample(point, npoint=1000):
    """
    Input:
        xyz: pointcloud data, [N, D]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [npoint, D]
    """
    N, _ = point.shape
    xyz = point[:, :3]
    centroids = np.zeros((npoint,))
    distance = np.ones((N,)) * 1e10
    farthest = np.random.randint(0, N)
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = np.argmax(distance, -1)
    centroid = centroids.astype(np.int32)
    farthest_points = point[centroids.astype(np.int32)]
    points_left = np.delete(point, centroid, 0)
    return farthest_points, points_left


def euclid_dist_non_equal_length(list_a, list_b):
    distances = distance_matrix(list_a, list_b)
    return distances


def convert_qt_to_transformation_matrix(qt, numpy=False):
    """
    qt is an (*,7) matrix where the first four columns represent the quaternion as an w, x, y, z and the last
    three columns represent the translation
    """
    if len(qt.shape) == 1:
        if numpy or not torch.is_tensor(qt):
            qt = np.expand_dims(qt, 0)
        else:
            qt = qt.unsqueeze(0)
    if torch.is_tensor(qt):
        quat = qt[:, :4].cpu().numpy()
    else:
        quat = qt[:, :4]
    Ts = qt[:, 4:]
    quat = np.roll(quat, -1, axis=1)
    Rs = R.from_quat(quat)
    if numpy:
        transformation_matrix = np.eye(4)
        transformation_matrix = transformation_matrix.reshape(1, 4, 4)
        transformation_matrix = np.repeat(transformation_matrix, qt.shape[0], 0)
        transformation_matrix[:, :3, :3] = Rs.as_matrix()
        transformation_matrix[:, :3, 3] = Ts
    else:
        transformation_matrix = torch.eye(4)
        transformation_matrix = transformation_matrix.reshape(1, 4, 4)
        transformation_matrix = transformation_matrix.repeat(qt.shape[0], 1, 1)
        transformation_matrix[:, :3, :3] = torch.tensor(Rs.as_matrix())
        transformation_matrix[:, :3, 3] = torch.tensor(Ts)
    return transformation_matrix

def convert_transformation_matrix_to_qt(transformation_matrix, numpy=False):
    """
        in the form: Bx4x4
        Bx[ a b c d
        e f g h
        i j k l
        0 0 0 1]
    """
    if len(transformation_matrix.shape) == 2:
        if numpy or not torch.is_tensor(transformation_matrix):
            transformation_matrix = np.expand_dims(transformation_matrix, 0)
        else:
            transformation_matrix = transformation_matrix.unsqueeze(0)
    if torch.is_tensor(transformation_matrix):
        transformation_matrix_numpy = transformation_matrix.cpu().numpy()
    else:
        transformation_matrix_numpy = transformation_matrix
        # quat = transformation_matrix[:, :4].cpu().numpy()

    t = transformation_matrix_numpy[:,:3,3]
    Rs = transformation_matrix_numpy[:,:3,:3]

    R_rotmat = R.from_matrix(Rs)
    q = R_rotmat.as_quat()
    q = np.roll(q, 1, axis=1)

    if numpy:
        qt = np.zeros((1,7))
        qt = np.repeat(qt, transformation_matrix.shape[0], 0)
        qt[:,:4] = q
        qt[:,4:] = t
    else:
        qt = torch.zeros((1,7))
        qt = qt.repeat(transformation_matrix.shape[0], 1)
        qt[:,:4] = torch.tensor(q)
        qt[:,4:] = torch.tensor(t)
    return qt

def create_gripper_meshes_from_transformation_matrices(grasp_transformations, colors=[[1, 0, 0]]):
    grippers = []
    for i, grasp_transformation in enumerate(grasp_transformations):
        if len(colors) > 1:
            color = colors[i]
        else:
            color = colors[0]
        gripper = create_gripper_marker(color)
        T = convert_qt_to_transformation_matrix(grasp_transformation)[0]
        grippers.append(gripper.transform(T))
    return grippers


def create_gripper_point_clouds_from_transformation_matrices(grasp_transformations, color=[1, 0, 0]):
    point_clouds = []
    for grasp_transformation in grasp_transformations:
        gripper = create_gripper_marker(color)
        T = convert_qt_to_transformation_matrix(grasp_transformation)[0]
        gripper = gripper.transform(T)
        point_cloud = gripper.sample_points_uniformly()

        point_clouds.append(point_cloud)
    return point_clouds


def create_o3d_point_cloud(pc, color=np.asarray([[1, 0, 0]]).T):
    grasp_point_cloud_o3d = o3d.geometry.PointCloud()
    grasp_point_cloud_o3d.points = o3d.utility.Vector3dVector(pc)
    if color is not None:
        grasp_point_cloud_o3d.paint_uniform_color(color)
    return grasp_point_cloud_o3d


def load_mesh_from_file(mesh_file, scale=1):
    mesh = o3d.io.read_triangle_mesh(mesh_file)
    mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices)*scale)
    mesh.paint_uniform_color([0.768, 0.768, 0.768])
    return mesh


def get_canonical_grasp_center_point():
    right_gripper_top = np.asarray([[4.10000000e-02, -7.27595772e-12, 1.12169998e-01]])
    left_gripper_top = np.asarray([[-4.10000000e-02, -7.27595772e-12,  1.12169998e-01]])
    right_gripper_base = np.asarray([[4.10000000e-02, -7.27595772e-12, 6.59999996e-02]])
    left_gripper_base = np.asarray([[-4.10000000e-02, -7.27595772e-12,  6.59999996e-02]])

    center_point = (right_gripper_top+left_gripper_top+right_gripper_base+left_gripper_base)/4.0
    return center_point


def transform_vector(vector, quaternion_and_translations):
    new_vector = np.ones((vector.shape[0], 4))
    new_vector[:, :3] = vector
    T = convert_qt_to_transformation_matrix(quaternion_and_translations, numpy=True)
    transformed_vector = T.dot(new_vector.T).squeeze()
    if transformed_vector.shape[0] == 4:
        transformed_vector = transformed_vector[:3, :].T
    elif transformed_vector.shape[1] == 4:
        transformed_vector = transformed_vector[:, :3]
    return transformed_vector


def file_exists(folder, file):
    return os.path.exists(folder+file)


def flip(q):
    """Flip a quaternion to the real positive hemisphere if needed."""
    q[q[:, 0] < 0, :] *= -1
    return q


def quaternion_from_euler(ai, aj, ak, device="cpu"):
    """Return quaternion from Euler angles and axis sequence.
    ai, aj, ak : Euler's roll, pitch and yaw angles
    axes : One of 24 axis sequences as string or encoded tuple
    >>> q = quaternion_from_euler(1, 2, 3, 'ryxz')
    >>> np.allclose(q, [0.435953, 0.310622, -0.718287, 0.444435])
    True
    """

    ai /= 2.0
    aj /= 2.0
    ak /= 2.0
    ci = torch.cos(ai)
    si = torch.sin(ai)
    cj = torch.cos(aj)
    sj = torch.sin(aj)
    ck = torch.cos(ak)
    sk = torch.sin(ak)
    cc = ci * ck
    cs = ci * sk
    sc = si * ck
    ss = si * sk
    q = torch.empty((ai.shape[0], 4)).to(device)
    q[:, 0] = cj * cc + sj * ss
    q[:, 1] = cj * sc - sj * cs
    q[:, 2] = cj * ss + sj * cc
    q[:, 3] = cj * cs - sj * sc
    q *= -1.0

    return q


def get_graph_feature(x,
                      k=20,
                      device="cuda",
                      idx=None,
                      dim9=False,
                      rotation_invariant=False):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        if dim9 == False:
            idx = knn(x, k=k)  # (batch_size, num_points, k)
        else:
            idx = knn(x[:, 6:], k=k)
    device = torch.device(device)

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1,
                                                               1) * num_points
    idx = idx + idx_base

    idx = idx.view(-1)

    _, num_dims, _ = x.size()
    x = x.transpose(2, 1).contiguous(
    )  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)

    try:
        feature = x.view(batch_size * num_points, -1)[idx, :]
        feature = feature.view(batch_size, num_points, k, num_dims)
        x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    except:
        print(type(x))
        print(x.shape)
        print(batch_size)
        print(num_points)
        print(batch_size * num_points)
        input()
        print(idx)
        input()
    if rotation_invariant:
        # temp = torch.rand((batch_size, 4, num_points, k)).to(device)
        try:
            temp = RRI(x, feature,
                       device).permute(0, 3, 1,
                                       2)  # (batch_size, 4, num_points, k)
        except:
            print(x)
            print("SDFA")
            input()

        return temp
    else:
        return torch.cat(
            (feature - x, x),
            dim=3).permute(0, 3, 1,
                           2)  # (batch_size, 2*num_dims, num_points, k)


def RRI(x, knn, device):
    rx = torch.norm(x, p=2, dim=-1, keepdim=True)
    rxy = torch.norm(knn, p=2, dim=-1, keepdim=True)
    normal_vec = x / rx
    pik = knn / rxy
    if torch.any(rx == 0) or torch.any(rxy == 0):
        print("ADFAS")
    temp = torch.sum(normal_vec * pik, dim=-1)
    temp = temp[:, :, 1:]
    temp[temp > 1] = 1
    temp[temp < -1] = -1
    theta_xy = torch.acos(temp)

    proj_yx = pik[:, :, 1:, :] - temp[:, :, :, None] * normal_vec[:, :, 1:, :]
    proj_yx_norm = torch.norm(proj_yx, p=2, dim=-1, keepdim=True)

    proj_yx = proj_yx / proj_yx_norm
    cos_phi = torch.sum(proj_yx[:, :, :, None, :] * proj_yx[:, :, None, :, :],
                        dim=-1)
    sin_phi = torch.zeros(cos_phi.shape).to(device)
    for i in range(knn.shape[2] - 1):
        for j in range(i + 1, knn.shape[2] - 1):
            temp = torch.cross(proj_yx[0, 0, i], proj_yx[0, 0, j])
            res = torch.sum(temp * normal_vec[0, 0, i], dim=-1)
            sin_phi[:, :, i, j] = res
            sin_phi[:, :, j, i] = -1 * res
    phi = torch.atan2(sin_phi, cos_phi)
    phi[phi < 0] += 2 * math.pi

    dim = knn.shape
    min_phi = torch.min(phi[phi > 0].view(dim[0], dim[1], dim[2] - 1,
                                          dim[2] - 2),
                        dim=-1)[0]
    features = torch.zeros((dim[0], dim[1], dim[2] - 1, 4)).to(device)
    features[:, :, :, 0] = rx[:, :, 1:, 0]
    features[:, :, :, 1] = rxy[:, :, 1:, 0]
    features[:, :, :, 2] = theta_xy[:, :, :]
    features[:, :, :, 3] = min_phi[:, :, :]
    return features


def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx

def save_obj_pickle(dataset_individual, recordname, numobj=None):
    grasp_file_list_all = dataset_individual.dataset.paths
    indices = dataset_individual.indices
    grasp_file_list = [grasp_file_list_all[index] for index in indices]
    obj_info_dict = {}
    for grasp_file in grasp_file_list:
        with open(grasp_file, "rb") as f:
            data = pickle.load(f)
            obj_scale = data['mesh/scale']
        obj_name = os.path.basename(grasp_file).split('_')[2] + ".obj"
        obj_info_dict[obj_name] = obj_scale
    with open(recordname, "wb") as f:
        pickle.dump(obj_info_dict, f)

def computeRotMat(v):
    '''
        input: v with size (N,3)
        return: rotation matrix with size (N, 3, 3)
    '''

    # TODO: colinear

    # get rotation axis vector
    # the vector norm represent the rotation angle
    v = v / np.linalg.norm(v, axis=1).reshape(-1, 1)
    yaxis = np.array([0, -1, 0])
    theta = np.pi-np.arccos(v[:, 1])
    # rotaxis = np.cross(v, yaxis)
    rotaxis = np.cross(v, yaxis)
    # np.linalg.norm(rotaxis, axis=1) < 1e-8

    rotaxis = rotaxis * (theta.reshape(-1, 1)) / np.linalg.norm(rotaxis, axis=1).reshape(-1, 1)

    # initialize the rotation object
    r = R.from_rotvec(rotaxis)

    return r

def get_cartesian_center_from_yaw_pitch_label(yaw_label, pitch_label, yaw_resolution, pitch_resolution):
    '''
    Given the discretized yaw, pitch labels with a specific resolution,
    we construct a sector and compute the center vector of it
    '''

    r_ap_yaw = (yaw_label + 1 / 2) * 2 * math.pi / yaw_resolution
    r_ap_pitch = (pitch_label + 1 / 2) * math.pi / pitch_resolution
    r_ap_yaw -= math.pi
    vx = math.sin(r_ap_pitch) * math.cos(r_ap_yaw)
    vy = math.sin(r_ap_pitch) * math.sin(r_ap_yaw)
    vz = math.cos(r_ap_pitch)
    return np.array([vx, vy, vz])

def align_pc_approach(pc_input, approach_vec, pcmode="single"):
    approach_vec = approach_vec.reshape(-1, 3)
    alignmat = computeRotMat(approach_vec).as_matrix()[0]
    pc_input_copy = copy.deepcopy(pc_input)
    if pcmode == "multi":
        for i in range(pc_input_copy.shape[0]):
            pc_input_copy[i] = np.dot(alignmat, pc_input_copy[i].T).T
    elif pcmode == "single":
        pc_input_copy = np.dot(alignmat, pc_input_copy.T).T

    return pc_input_copy

