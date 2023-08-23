import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
import glob
import copy
import pickle
import os

from utils.sample_condition import get_approach_condition_batch

class GraspSamplerData(BaseDataset):
    def __init__(self, opt, caching=True):
        BaseDataset.__init__(self, opt, caching=caching)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        opt.input_nc = self.ninput_channels
        self.i = 0
        self.validate = opt.validate
        self.obj_rotation = None
        self.caching = False

    def __getitem__(self, index):
        path = self.paths[index]

        meta = {}
        try:
            point_clouds, clusters_with_grasps_per_point_cloud, grasps_per_clusters = self.read_grasp_file(
                path, self.opt.debug)
            try:
                if grasps_per_clusters[0][0].shape[0] < 64:
                    raise NoPositiveGraspsException
            except:
                raise NoPositiveGraspsException

            if self.validate:
                sampled_point_cloud_indices, sampled_query_point_indices_with_grasps_per_point_cloud, sampled_grasps_per_query_point = self.sample_clusters_query_points_and_grasps(
                    1, point_clouds, clusters_with_grasps_per_point_cloud, grasps_per_clusters)
            else:
                sampled_point_cloud_indices, sampled_query_point_indices_with_grasps_per_point_cloud, sampled_grasps_per_query_point \
                    = self.sample_clusters_query_points_and_grasps(
                    self.opt.num_grasps_per_object, point_clouds, clusters_with_grasps_per_point_cloud, grasps_per_clusters)

        except NoPositiveGraspsException:
            if self.opt.skip_error:
                return None
            else:
                return self.__getitem__(np.random.randint(0, self.size))

        pc = copy.deepcopy(point_clouds[sampled_point_cloud_indices])

        # # Adjust point cloud to match depth image
        pc[:,:,1] = -pc[:,:,1]
        pc[:,:,2] = -pc[:,:,2]

        # Handle grasp flipping for acronym dataset
        sampled_grasps_per_query_flip_list = []
        sampled_grasps_per_query_flip_qt_list = []
        yzopp_mat = np.eye(4)
        yzopp_mat[1, 1] = -1
        yzopp_mat[2, 2] = -1

        if not self.validate:
            for grasp_id in range(sampled_grasps_per_query_point.shape[0]):
                sampled_grasps_per_query_point_flip = np.matmul(yzopp_mat, sampled_grasps_per_query_point[grasp_id])
                sampled_grasps_per_query_point_flip_qt = utils.convert_transformation_matrix_to_qt(sampled_grasps_per_query_point_flip, numpy=True)
                sampled_grasps_per_query_flip_list.append(sampled_grasps_per_query_point_flip)
                sampled_grasps_per_query_flip_qt_list.append(sampled_grasps_per_query_point_flip_qt)
            sampled_grasps_per_query_point = np.array(sampled_grasps_per_query_flip_list)
            sampled_grasps_per_query_point_qt = np.array(sampled_grasps_per_query_flip_qt_list)

            sampled_grasps_per_query_point_qt_squeeze = sampled_grasps_per_query_point_qt.squeeze(axis=1)

            deg_radius = np.random.uniform(low=0, high=90, size=sampled_grasps_per_query_point_qt_squeeze.shape[0])

            sample_grasps_continuous_label, _, angdist, sample_grasps_continuous_label_coord = get_approach_condition_batch(sampled_grasps_per_query_point_qt_squeeze, deg_radius=deg_radius, mode="multi",
                                                                     debug=False, gen_label=True, yawpitch=True)

            features = self.create_constrained_features_orien_label_continuous_equivariant(
                pc.shape[0], pc.shape[1], sampled_query_point_indices_with_grasps_per_point_cloud,
                sample_grasps_continuous_label)


            r_matrix = utils.computeRotMat(sample_grasps_continuous_label_coord).as_matrix()
            for i in range(r_matrix.shape[0]):
                rotmat = r_matrix[i]
                pc[i] = np.dot(rotmat, pc[i].T).T

                # transform the grasp into the approach space
                alignmat = np.eye(4)
                alignmat[:3,:3] = rotmat
                sampled_grasps_per_query_point[i] = np.dot(alignmat, sampled_grasps_per_query_point[i])
                sampled_grasps_per_query_point_qt[i] = utils.convert_transformation_matrix_to_qt(sampled_grasps_per_query_point[i], numpy=True)

            gt_control_points = utils.transform_control_points_numpy(
                sampled_grasps_per_query_point, self.opt.num_grasps_per_object, mode='rt')[:, :, :3]

            meta['pc'] = pc
            meta['features'] = features

            meta['grasp_rt'] = sampled_grasps_per_query_point.reshape(
                len(sampled_grasps_per_query_point), -1)
            meta['target_cps'] = np.array(gt_control_points[:, :, :3])
            meta['grasp_qt'] = sampled_grasps_per_query_point_qt.reshape(
            len(sampled_grasps_per_query_point_qt), -1)
        else:
            features = np.random.rand(pc.shape[0], pc.shape[1], 1)
            pc = np.tile(pc, (self.opt.num_grasps_per_object, 1, 1))
            features = np.tile(features, (self.opt.num_grasps_per_object, 1, 1))
            meta['pc'] = pc
            meta['features'] = features

            meta['grasp_rt'] = sampled_grasps_per_query_point.reshape(
                len(sampled_grasps_per_query_point), -1)

            meta['target_cps'] = []  # np.array(gt_control_points[:, :, :3])
            meta['grasp_qt'] = meta['grasp_rt']

        return meta

    def __len__(self):
        return self.size

    def create_constrained_features_orien_label_continuous_equivariant(self, batch_size, points_per_pc, query_point_indices_to_grasp,
                                                   sampled_grasps_per_query_point_orien_label):
        # The first channel is reserved for positional constraint, but not used at this moment
        features = np.zeros((batch_size, points_per_pc, 2))

        for i in range(batch_size):
            if not self.validate:
                features[i,:,1:] = sampled_grasps_per_query_point_orien_label[i,3]
            else:
                features[i,:,1:] = 0.78

        # Currently, we ignore the positional flag
        features = features[:, :, 1:]

        return features

    def sample_clusters_query_points_and_grasps(self, num_samples, point_clouds, clusters_with_grasps, grasps_per_cluster):
        num_point_clouds = len(point_clouds)
        if len(point_clouds) == 0:
            raise NoPositiveGraspsException

        replace = num_samples > num_point_clouds
        point_cloud_indices = np.random.choice(range(num_point_clouds),
                                               size=num_samples,
                                               replace=replace).astype(np.int32)

        random_cluster_indices_with_grasps = []
        random_grasps_per_cluster = []


        for point_cloud_index in point_cloud_indices:
            if not self.validate:
                num_cluters_with_grasp_for_current_point_cloud = len(clusters_with_grasps[point_cloud_index])
                random_cluster_index_with_grasp = np.random.randint(0, num_cluters_with_grasp_for_current_point_cloud)
                random_cluster_indices_with_grasps.append(
                    clusters_with_grasps[point_cloud_index][random_cluster_index_with_grasp])

                num_grasps_in_random_cluster = len(grasps_per_cluster[point_cloud_index][random_cluster_index_with_grasp])
                random_grasp_index_for_random_query_point = np.random.randint(0, num_grasps_in_random_cluster)

                random_grasps_per_cluster.append(
                    grasps_per_cluster[point_cloud_index][random_cluster_index_with_grasp][random_grasp_index_for_random_query_point])

            else:
                num_cluters_with_grasp_for_current_point_cloud = len(clusters_with_grasps[point_cloud_index])
                random_cluster_index_with_grasp = np.random.randint(0, num_cluters_with_grasp_for_current_point_cloud)
                random_cluster_indices_with_grasps.append(
                    clusters_with_grasps[point_cloud_index][random_cluster_index_with_grasp])

                random_grasps_per_cluster.append([])

        random_cluster_indices_with_grasps = np.asarray(random_cluster_indices_with_grasps)
        random_grasps_per_cluster = np.asarray(random_grasps_per_cluster)
        return point_cloud_indices, random_cluster_indices_with_grasps, random_grasps_per_cluster

    def make_dataset(self):
        """Retrieve all files from the dataset root folder."""
        files = glob.glob(self.opt.dataset_root_folder+"/*")
        return files

    def read_grasp_file(self, path, debug=False):
        file_name = path
        if self.caching and file_name in self.cache:
            point_clouds, query_points_with_grasps, grasps_per_query_points = copy.deepcopy(
                self.cache[file_name])
            return point_clouds,  query_points_with_grasps, grasps_per_query_points

        point_clouds, query_points_with_grasps, grasps_per_query_points = self.read_object_grasp_data(path, debug=debug)

        if self.caching:
            self.cache[file_name] = (point_clouds, query_points_with_grasps, grasps_per_query_points)
            return copy.deepcopy(self.cache[file_name])

        return point_clouds, query_points_with_grasps, grasps_per_query_points

    def read_object_grasp_data(self,
                               file_path, debug=False
                               ):
        """
        Reads the grasps from the json path and loads the mesh and all the
        grasps.
        """
        num_clusters = self.opt.num_grasp_clusters

        if num_clusters <= 0:
            raise NoPositiveGraspsException

        grasps, pre_rendered_point_clouds, camera_poses_for_prerendered_point_clouds, all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud = self.load_data_from_file(
            file_path, debug=debug)

        clusters_with_grasps_for_each_point_cloud, grasps_for_each_cluster_per_point_cloud, point_clouds_to_keep = self.get_query_points_with_grasps_and_all_grasps(
            all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud, grasps, camera_poses_for_prerendered_point_clouds)
        pre_rendered_point_clouds = pre_rendered_point_clouds[point_clouds_to_keep]
        # , mesh_file, mesh_scale, camera_poses_for_prerendered_point_clouds
        return pre_rendered_point_clouds, clusters_with_grasps_for_each_point_cloud, grasps_for_each_cluster_per_point_cloud


    def load_data_from_file(self, file_path, debug=False):
        if file_path.endswith("h5"):
            return self.load_from_h5py_file(file_path)
        elif file_path.endswith("pickle"):
            return self.load_from_pickle_file(file_path, debug=debug)
        else:
            raise ValueError("Cannot read file with extension ", file_path.split(".")[-1])


    def load_from_h5py_file(self, h5py_file):
        h5_dict = utils.read_h5_file(h5py_file)
        grasps = h5_dict['grasps/transformations'][()]
        pre_rendered_point_clouds = h5_dict['rendering/point_clouds'][()]
        camera_poses_for_prerendered_point_clouds = h5_dict['rendering/camera_poses'][()]
        all_query_points_per_point_cloud = h5_dict["query_points/points_with_grasps_on_each_rendered_point_cloud"][()]
        grasp_indices_for_every_query_point_on_each_rendered_point_cloud = h5_dict[
            "query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"][()]

        return grasps, pre_rendered_point_clouds, camera_poses_for_prerendered_point_clouds, all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud

    def load_from_pickle_file(self, file, debug=False):
        with open(file, "rb") as f:
            data = pickle.load(f)
            grasps = data['grasps/transformations']
            pre_rendered_point_clouds = np.asarray(data['rendering/point_clouds'])
            camera_poses_for_prerendered_point_clouds = np.asarray(data['rendering/camera_poses'])
            all_query_points_per_point_cloud = np.asarray(data["query_points/points_with_grasps_on_each_rendered_point_cloud"])
            grasp_indices_for_every_query_point_on_each_rendered_point_cloud = np.asarray(data[
                "query_points/grasp_indices_for_each_point_with_grasp_on_each_rendered_point_cloud"])

        if debug == True:
            camposeid = 5
            return grasps, pre_rendered_point_clouds[None, camposeid], camera_poses_for_prerendered_point_clouds[None, camposeid], \
                   all_query_points_per_point_cloud[None, camposeid], grasp_indices_for_every_query_point_on_each_rendered_point_cloud[None, camposeid]

        return grasps, pre_rendered_point_clouds, camera_poses_for_prerendered_point_clouds, all_query_points_per_point_cloud, grasp_indices_for_every_query_point_on_each_rendered_point_cloud


    def get_query_points_with_grasps_and_all_grasps(self, all_clusters,
                                                    grasp_indices_for_every_cluster_on_each_rendered_point_cloud,
                                                    grasps, camera_poses_for_prerendered_point_clouds):

        all_clusters_per_point_cloud = []
        all_grasps_per_cluster_per_point_cloud = []
        point_clouds_to_keep = []
        for point_cloud_index in range(all_clusters.shape[0]):
            all_clusters_with_grasps_per_point_cloud = all_clusters[point_cloud_index]
            if len(all_clusters_with_grasps_per_point_cloud) == 0:
                continue
            point_clouds_to_keep.append(point_cloud_index)
            all_clusters_per_point_cloud.append(all_clusters_with_grasps_per_point_cloud)

            all_grasps_per_cluster = []
            grasp_indices_for_all_clusters = grasp_indices_for_every_cluster_on_each_rendered_point_cloud[point_cloud_index]

            camera_pose_for_current_point_cloud_index = camera_poses_for_prerendered_point_clouds[point_cloud_index]
            for grasp_indices_per_cluster in grasp_indices_for_all_clusters:
                transformed_grasps = np.matmul(camera_pose_for_current_point_cloud_index,
                                               grasps[grasp_indices_per_cluster, :, :])
                all_grasps_per_cluster.append(transformed_grasps)
            all_grasps_per_cluster_per_point_cloud.append(all_grasps_per_cluster)

        return all_clusters_per_point_cloud, all_grasps_per_cluster_per_point_cloud, point_clouds_to_keep