import torch.utils.data as data
import numpy as np
import pickle
import os
import copy
import json
from utils.sample import Object
from utils import utils
import glob
# from renderer.online_object_renderer import OnlineObjectRenderer
import threading
# np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)


class NoPositiveGraspsException(Exception):
    """raised when there's no positive grasps for an object."""
    pass


class BaseDataset(data.Dataset):
    def __init__(self,
                 opt,
                 shared_dict={},
                 caching=True,
                 prior_orien=0.0,
                 yaw_resolution=4.0,
                 pitch_resolution=1.0,
                 collision_hard_neg_min_translation=(-0.03, -0.03, -0.03),
                 collision_hard_neg_max_translation=(0.03, 0.03, 0.03),
                 collision_hard_neg_min_rotation=(-0.6, -0.2, -0.6),
                 collision_hard_neg_max_rotation=(+0.6, +0.2, +0.6),
                 collision_hard_neg_num_perturbations=10):
        super(BaseDataset, self).__init__()
        self.opt = opt
        self.mean = 0
        self.std = 1
        self.ninput_channels = None
        self.current_pc = None
        self.caching = caching
        self.prior_orien = prior_orien
        self.yaw_resolution = yaw_resolution
        self.pitch_resolution = pitch_resolution
        self.cache = shared_dict
        self.collision_hard_neg_min_translation = collision_hard_neg_min_translation
        self.collision_hard_neg_max_translation = collision_hard_neg_max_translation
        self.collision_hard_neg_min_rotation = collision_hard_neg_min_rotation
        self.collision_hard_neg_max_rotation = collision_hard_neg_max_rotation
        self.collision_hard_neg_num_perturbations = collision_hard_neg_num_perturbations
        self.lock = threading.Lock()
        for i in range(3):
            assert (collision_hard_neg_min_rotation[i] <=
                    collision_hard_neg_max_rotation[i])
            assert (collision_hard_neg_min_translation[i] <=
                    collision_hard_neg_max_translation[i])

        # self.renderer = OnlineObjectRenderer(caching=True)

        if opt.use_uniform_quaternions:
            self.all_poses = utils.uniform_quaternions()
        else:
            self.all_poses = utils.nonuniform_quaternions()

        self.eval_files = [
            json.load(open(f)) for f in glob.glob(
                os.path.join(self.opt.dataset_root_folder, 'splits', '*.json'))
        ]

    def apply_dropout(self, pc):
        if self.opt.occlusion_nclusters == 0 or self.opt.occlusion_dropout_rate == 0.:
            return np.copy(pc)

        labels = utils.farthest_points(pc, self.opt.occlusion_nclusters,
                                       utils.distance_by_translation_point)

        removed_labels = np.unique(labels)
        removed_labels = removed_labels[np.random.rand(removed_labels.shape[0])
                                        < self.opt.occlusion_dropout_rate]
        if removed_labels.shape[0] == 0:
            return np.copy(pc)
        mask = np.ones(labels.shape, labels.dtype)
        for l in removed_labels:
            mask = np.logical_and(mask, labels != l)
        return pc[mask]

    def render_random_scene(self, camera_pose=None):
        """
          Renders a random view and return (pc, camera_pose, object_pose). 
          object_pose is None for single object per scene.
        """
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self.all_poses))
            camera_pose = self.all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self.renderer.render(in_camera_pose)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, self.opt.npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

    def change_object_and_render(self,
                                 cad_path,
                                 cad_scale,
                                 camera_pose=None,
                                 thread_id=0):
        if camera_pose is None:
            viewing_index = np.random.randint(0, high=len(self.all_poses))
            camera_pose = self.all_poses[viewing_index]

        in_camera_pose = copy.deepcopy(camera_pose)
        _, _, pc, camera_pose = self.renderer.change_and_render(
            cad_path, cad_scale, in_camera_pose, thread_id)
        pc = self.apply_dropout(pc)
        pc = utils.regularize_pc_point_count(pc, self.opt.npoints)
        pc_mean = np.mean(pc, 0, keepdims=True)
        pc[:, :3] -= pc_mean[:, :3]
        camera_pose[:3, 3] -= pc_mean[0, :3]

        return pc, camera_pose, in_camera_pose

    def change_object(self, cad_path, cad_scale):
        self.renderer.change_object(cad_path, cad_scale)

    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache.keys():
            pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds

        pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds = self.read_object_grasp_data(
            path,
            return_all_grasps=return_all_grasps)

        if self.caching:
            self.cache[file_name] = (pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds)
            return copy.deepcopy(self.cache[file_name])

        return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds

    def read_object_grasp_data(self,
                               data_path,
                               return_all_grasps=True):
        """
        Reads the grasps from the json path and loads the mesh and all the 
        grasps.
        """
        num_clusters = self.opt.num_grasp_clusters

        if num_clusters <= 0:
            raise NoPositiveGraspsException

        json_dict = pickle.load(open(data_path, "rb"))

        object_model = Object(json_dict['mesh/file'])
        object_model.rescale(json_dict['mesh/scale'])
        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)

        object_model.vertices -= object_mean

        grasps = np.asarray(json_dict['grasps/transformations'])
        point_clouds = np.asarray(json_dict['rendering/point_clouds'])
        camera_poses_for_prerendered_point_clouds = np.asarray(json_dict['rendering/camera_poses'])
        flex_qualities = np.asarray(json_dict['grasps/successes'])

        successful_mask = (flex_qualities == 1)

        positive_grasp_indexes = np.where(successful_mask)[0]
        negative_grasp_indexes = np.where(~successful_mask)[0]

        positive_grasps = grasps[positive_grasp_indexes, :, :]
        negative_grasps = grasps[negative_grasp_indexes, :, :]

        def cluster_grasps(grasps):
            cluster_indexes = np.asarray(
                utils.farthest_points(grasps, num_clusters,
                                      utils.distance_by_translation_grasp))
            output_grasps = []

            for i in range(num_clusters):
                indexes = np.where(cluster_indexes == i)[0]
                output_grasps.append(grasps[indexes, :, :])

            output_grasps = np.asarray(output_grasps)

            return output_grasps

        if not return_all_grasps:
            positive_grasps = cluster_grasps(
                positive_grasps)
            negative_grasps = cluster_grasps(
                negative_grasps)
        return positive_grasps, negative_grasps, object_model, point_clouds, camera_poses_for_prerendered_point_clouds

    def sample_grasp_indexes(self, n, grasps):
        """
          Stratified sampling of the grasps.
        """
        nonzero_rows = [i for i in range(len(grasps)) if len(grasps[i]) > 0]
        num_clusters = len(nonzero_rows)
        replace = n > num_clusters
        if num_clusters == 0:
            raise NoPositiveGraspsException

        grasp_rows = np.random.choice(range(num_clusters),
                                      size=n,
                                      replace=replace).astype(np.int32)
        grasp_rows = [nonzero_rows[i] for i in grasp_rows]
        grasp_cols = []
        for grasp_row in grasp_rows:
            if len(grasps[grasp_rows]) == 0:
                raise ValueError('grasps cannot be empty')

            grasp_cols.append(np.random.randint(len(grasps[grasp_row])))

        grasp_cols = np.asarray(grasp_cols, dtype=np.int32)
        return np.vstack((grasp_rows, grasp_cols)).T

    def get_mean_std(self):
        """ Computes Mean and Standard Deviation from Training Data
        If mean/std file doesn't exist, will compute one
        :returns
        mean: N-dimensional mean
        std: N-dimensional standard deviation
        ninput_channels: N
        (here N=5)
        """

        mean_std_cache = os.path.join(self.opt.dataset_root_folder,
                                      'mean_std_cache.p')
        if not os.path.isfile(mean_std_cache):
            print('computing mean std from train data...')
            # doesn't run augmentation during m/std computation
            num_aug = self.opt.num_aug
            self.opt.num_aug = 1
            mean, std = np.array(0), np.array(0)
            for i, data in enumerate(self):
                if i % 500 == 0:
                    print('{} of {}'.format(i, self.size))
                features = data['edge_features']
                mean = mean + features.mean(axis=1)
                std = std + features.std(axis=1)
            mean = mean / (i + 1)
            std = std / (i + 1)
            transform_dict = {
                'mean': mean[:, np.newaxis],
                'std': std[:, np.newaxis],
                'ninput_channels': len(mean)
            }
            with open(mean_std_cache, 'wb') as f:
                pickle.dump(transform_dict, f)
            print('saved: ', mean_std_cache)
            self.opt.num_aug = num_aug
        # open mean / std from file
        with open(mean_std_cache, 'rb') as f:
            transform_dict = pickle.load(f)
            print('loaded mean / std from cache')
            self.mean = transform_dict['mean']
            self.std = transform_dict['std']
            self.ninput_channels = transform_dict['ninput_channels']

    def make_dataset(self):
        files = glob.glob(self.opt.dataset_root_folder+"/*")
        return files


def collate_fn(batch):
    """Creates mini-batch tensors
    We should build custom collate_fn rather than using default collate_fn
    """
    batch = list(filter(lambda x: x is not None, batch))  #
    meta = {}
    keys = batch[0].keys()
    for key in keys:
        meta.update({key: np.concatenate([d[key] for d in batch])})
    return meta
