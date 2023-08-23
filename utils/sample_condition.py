import os
import sys

import numpy as np
import time
import numpy as np
import matplotlib.pyplot as plt
import pickle as pkl
from utils import utils
import math
from scipy.interpolate import interp1d
from scipy.integrate import quad
# from constrained_6dof_graspnet.utils import utils
from mpl_toolkits.mplot3d import Axes3D


def random_unit_vectors(N):
    V = np.random.uniform(-1, 1, (N, 3))
    norms = np.linalg.norm(V, axis=1)
    V /= norms[:, None]
    return V

def adjust_collinear_vectors(R, V1):
    dot_products = np.einsum('ij,ij->i', R, V1)
    collinear_indices = dot_products > 0.999
    R[collinear_indices] += np.random.uniform(-0.1, 0.1, 3)
    return R

def random_vectors_around(V1, theta, N):
    R = random_unit_vectors(N)
    R = adjust_collinear_vectors(R, V1)

    W = np.cross(V1, R)
    U_W = W / np.linalg.norm(W, axis=1)[:, None]

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    V2 = cos_theta[:, None] * V1 + sin_theta[:, None] * U_W
    V2 /= np.linalg.norm(V2, axis=1)[:, None]

    return V2


def sample_point_on_sphere(radius=1):
    while True:
        # Generate two random numbers in the range [-1, 1]
        x1, x2 = np.random.uniform(-1, 1, 2)

        # Calculate the sum of squares
        s = x1 ** 2 + x2 ** 2

        # Check if s is inside the unit circle
        if s <= 1:
            break

    # Calculate scaling factor
    scale = 2 * math.sqrt(1 - s)

    # Calculate the 3D coordinates of the point on the sphere
    x = radius * x1 * scale
    y = radius * x2 * scale
    z = radius * (1 - 2 * s)

    return x, y, z

def angle_between_normalized_vectors(a, b):
    # Calculate the dot product of normalized vectors a and b
    dot_product = np.dot(a, b)

    # Ensure the value is in the range [-1, 1] to avoid errors in the arccos function
    dot_product = np.clip(dot_product, -1, 1)

    # Calculate the angle in radians
    angle_rad = np.arccos(dot_product)

    # # Convert the angle to degrees
    angle_deg = np.degrees(angle_rad)

    return angle_deg

def rotation_matrix(a, b):
    # Helper function to calculate the rotation matrix to align vector a with vector b
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    R = np.eye(3) + kmat + kmat @ kmat * ((1 - c) / s**2)
    return R

def rotate_to_align(points, v):
    # Helper function to rotate points to align with vector v
    return np.dot(points, rotation_matrix(np.array([0, 0, 1]), v).T)

def sample_spherical_cap(N, v, r):
    # Sample N points on the surface of the spherical cap with unit vectors v and angle r
    phi = np.arccos(np.random.uniform(np.cos(r), 1, N))
    theta = np.random.uniform(0, 2 * np.pi, N)
    R = np.sin(phi)
    x = R * np.cos(theta)
    y = R * np.sin(theta)
    z = np.cos(phi)
    points = np.stack([x, y, z], axis=-1)
    rotated_points = np.array([rotate_to_align(p, v_i) for p, v_i in zip(points, v)])
    return rotated_points

# def angle_between(v1, v2):
#     # Helper function to compute the angle between two vectors in radians
#     return np.arccos(np.clip(np.dot(v1, v2.T), -1.0, 1.0))

def angle_between(v1, v2):
    # Ensure v1 and v2 are numpy arrays
    v1 = np.array(v1)
    v2 = np.array(v2)

    # Compute the dot product along the last axis (axis=-1)
    dot_product = np.sum(v1 * v2, axis=-1)

    # Compute the magnitudes of v1 and v2 along the last axis
    v1_magnitude = np.linalg.norm(v1, axis=-1)
    v2_magnitude = np.linalg.norm(v2, axis=-1)

    # Compute the cosine of the angles by element-wise division
    cos_angles = np.clip(dot_product / (v1_magnitude * v2_magnitude), -1.0, 1.0)

    # Compute the angles in radians using arccosine
    angles = np.arccos(cos_angles)

    return angles

def get_unit_approach_vec_batch(grasp_qt, debug=False):
    '''
    Get approach vector of a batch
    '''
    if debug is True:
        if grasp_qt is None:
            approach_vec = np.random.randn(600, 3)
            approach_vec[:200] = approach_vec[0]
            approach_vec[200:400] = approach_vec[200]
            approach_vec[400:] = approach_vec[400]
        else:
            grasp_pc = utils.transform_control_points_numpy(
                grasp_qt,
                batch_size=grasp_qt.shape[0],
                mode='qt', dense=False)
            grasp_pc[1:] = grasp_pc[0]
            grasp_approach_vector = (grasp_pc[:, 2] + grasp_pc[:, 3]) / 2 - grasp_pc[:, 0]
            approach_vec = grasp_approach_vector / np.linalg.norm(
                grasp_approach_vector, axis=1).reshape(-1, 1)

            approach_vec[1:] = approach_vec[0]

    else:
        grasp_pc = utils.transform_control_points_numpy(
            grasp_qt,
            batch_size=grasp_qt.shape[0],
            mode='qt', dense=False)

        grasp_approach_vector = (grasp_pc[:, 2] + grasp_pc[:,3]) / 2 - grasp_pc[:, 0]
        approach_vec = grasp_approach_vector / np.linalg.norm(
            grasp_approach_vector, axis=1).reshape(-1, 1)


    # approach_vec = None # (num_grasps, 3)
    approach_vec = approach_vec / np.linalg.norm(approach_vec, axis=1).reshape(-1, 1)
    return approach_vec

#
# def pdf(r, R):
#     if np.isscalar(r):
#         with np.errstate(divide='ignore', invalid='ignore'):
#             result = 1 / (2 * np.pi * R**2 * np.sin(r))
#         if np.isnan(result) or np.isinf(result):
#             return 0
#         else:
#             return result
#     else:
#         with np.errstate(divide='ignore', invalid='ignore'):
#             result = 1 / (2 * np.pi * R**2 * np.sin(r))
#         result[np.isnan(result)] = 0
#         return result
#
#
# def generate_r(n_samples, R, max_iterations=1000):
#     def pdf(r, R):
#         with np.errstate(divide='ignore', invalid='ignore'):
#             result = 1 / (2 * np.pi * R**2 * np.sin(r))
#         result[np.isnan(result)] = 0
#         result[np.isinf(result)] = np.finfo(np.float64).max
#         return result
#
#     r_values = np.linspace(0, np.pi/2, n_samples)
#     max_pdf = np.max(pdf(r_values, R))
#
#     samples = []
#     iterations = 0
#     while len(samples) < n_samples and iterations < max_iterations:
#         r_proposal = np.random.uniform(0, np.pi/2)
#         pdf_proposal = pdf(r_proposal, R)
#         u = np.random.uniform(0, max_pdf)
#
#         if u <= pdf_proposal:
#             samples.append(r_proposal)
#
#         iterations += 1
#
#     if len(samples) < n_samples:
#         print(f"Warning: Maximum iterations reached. Only {len(samples)} samples were generated.")
#
#     return np.array(samples)

from sklearn.mixture import GaussianMixture


def fit_gmm(R, n_components=3, eps=1e-8):
    """
    Fit a Gaussian Mixture Model to the data generated by the pdf function.

    Parameters:
    R (float): Radius of the sphere
    n_components (int): Number of components in the GMM
    eps (float): Small constant to avoid division by zero

    Returns:
    gmm (GaussianMixture): Fitted GMM
    """
    # Generate a large number of points from the pdf
    theta = np.linspace(0, np.pi, 1000)
    p_theta = 1 / (2 * np.pi * R ** 2 * (np.sin(theta) + eps))

    # Fit a GMM
    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(np.column_stack((theta, p_theta)))

    return gmm


def generate_r_gmm(n, R, gmm=None):
    """
    Generate n samples of r using a Gaussian Mixture Model.

    Parameters:
    n (int): Number of samples
    R (float): Radius of the sphere
    gmm (GaussianMixture, optional): Pre-fitted GMM. If None, a GMM will be fitted.

    Returns:
    r (numpy.array): Generated samples
    """
    if gmm is None:
        gmm = fit_gmm(R)

    samples = gmm.sample(n)[0]
    r = samples[:, 0]

    return r



def get_condition_label(approach_vec, deg_radius, mode="single", num_condition=10):
    '''
    sample conditions based on the approach vector direction
    '''
    condition_label = None

    if mode == "single":
        raise NotImplementedError

        assert approach_vec.shape == (3,)
        v = approach_vec.reshape(3)
        v_norm = np.linalg.norm(v)
        v_hat = v / v_norm

        sample_num = num_condition

        deg = deg_radius  # 0 to 360
        theta_range = np.pi / 2 - np.radians(deg), np.pi / 2 + np.radians(deg)
        theta_sample = np.random.uniform(low=theta_range[0], high=theta_range[1], size=sample_num).reshape(1, -1)
        perp_v = np.cross(v_hat, np.array([0, 0, 1]))
        if np.linalg.norm(perp_v) < 1e-6:
            # If v_hat is parallel to the z-axis, use the x-axis instead
            perp_v = np.cross(v_hat, np.array([1, 0, 0]))

        phi_sample = np.random.uniform(low=0, high=2 * np.pi, size=sample_num).reshape(1, -1)
        v_comm = np.cross(v_hat, perp_v).reshape(3, 1)
        perp_v = perp_v.reshape(3, 1)
        v_hat = v_hat.reshape(3, 1)



        rot_v_sample = np.dot(perp_v, np.cos(phi_sample)) + np.dot(v_comm, np.sin(phi_sample))
        # p = rot_v * np.cos(theta) + v_hat * np.sin(theta)
        condition_label = rot_v_sample * np.repeat(np.cos(theta_sample), repeats=3, axis=0) + np.dot(v_hat,
                                                                                                np.sin(theta_sample))

        condition_label = condition_label.transpose()
        print("hello")

    elif mode == "multi":
        '''
        inside a batch, for individual approach direction, we only sample one condition

        input: approach_vec: (N,3)
        deg_radius :(N,) or int
        num_condition=1

        output: sample conditions: (N,4) -> x,y,z, radius, can be directly concatenated after the grasps qt representation
        '''
        if num_condition != 1:
            raise NotImplementedError

        if type(deg_radius) is float or type(deg_radius) is int:
            deg_radius = deg_radius * np.ones(approach_vec.shape[0])
        rad_radius = np.radians(deg_radius)
        condition_label = sample_spherical_cap(approach_vec.shape[0], approach_vec, rad_radius)
        angdist = angle_between(condition_label, approach_vec)

    return condition_label, rad_radius, angdist # (num_grasps, 3) # 3 for xyz, 4 for quantanion representation

def convert_3D_yawpitch(grasp_approach_vector_normalize):
    vx = grasp_approach_vector_normalize[:,0]
    vy = grasp_approach_vector_normalize[:,1]
    vz = grasp_approach_vector_normalize[:,2]


    r_ap_pitch = np.arccos(vz)
    r_ap_yaw = np.arctan2(vy, vx) + math.pi

    yawpitchlabel = np.vstack([r_ap_yaw, r_ap_pitch, np.zeros_like(r_ap_yaw)]).T
    return yawpitchlabel


def get_approach_condition_batch(grasp_qt, deg_radius=None, mode="single", debug=True, gen_label=True, yawpitch=True):
    '''
    input a grasp_qt, return a minibatch of grasp orientation conditinos
    '''
    # get unit approach vector sets of the grasps
    approach_vec = get_unit_approach_vec_batch(grasp_qt, debug=debug)
    condition_label_batch_yp = None
    condition_label_batch_coord = None

    # get the sampled condition in SO(3) space based on approach direction
    if mode == "single":
        raise NotImplementedError
    elif mode == "multi":
        condition_label_batch, deg_radius_batch, angdist = get_condition_label(approach_vec, deg_radius = deg_radius, mode=mode, num_condition=1)
    else:
        raise NotImplementedError

    if gen_label is True:
        if yawpitch is False:
            condition_label_batch = np.hstack((condition_label_batch, deg_radius_batch.reshape(-1,1)))
        else:
            condition_label_batch_yp = convert_3D_yawpitch(condition_label_batch)
            condition_label_batch_yp = np.hstack((condition_label_batch_yp, deg_radius_batch.reshape(-1,1)))

        condition_label_batch_coord = condition_label_batch
    return condition_label_batch_yp, approach_vec, angdist, condition_label_batch_coord # need to reconstruct the gonet condition