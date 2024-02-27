import os
import torch
from data.base_dataset import BaseDataset, NoPositiveGraspsException
import numpy as np
from utils import utils
import random
from multiprocessing import Manager
import copy
import pickle

from utils.sample import Object


class GraspEvaluatorData(BaseDataset):
    def __init__(self, opt, ratio_positive=0.3, ratio_hardnegative=0.4):
        manager = Manager()
        shared_dict_cache = manager.dict()
        BaseDataset.__init__(self, opt, shared_dict_cache)
        self.opt = opt
        self.device = torch.device('cuda:{}'.format(
            opt.gpu_ids[0])) if opt.gpu_ids else torch.device('cpu')
        self.root = opt.dataset_root_folder
        self.paths = self.make_dataset()
        self.size = len(self.paths)
        self.collision_hard_neg_queue = manager.dict()
        # self.get_mean_std()
        opt.input_nc = self.ninput_channels
        self.ratio_positive = ratio_positive  # self.set_ratios(ratio_positive)
        self.ratio_hardnegative = ratio_hardnegative  # self.set_ratios(ratio_hardnegative)
        self.num_positive = int(self.opt.num_grasps_per_object * self.ratio_positive)
        self.num_hard_negative = int(self.opt.num_grasps_per_object * self.ratio_hardnegative)
        self.num_flex_negative = self.opt.num_grasps_per_object - self.num_positive - self.num_hard_negative
        self.equivariant = opt.equivariant

    def __getitem__(self, index):
        path = self.paths[index]
        
        try:
            data = self.get_uniform_evaluator_data(path)
        except NoPositiveGraspsException:
            if self.opt.skip_error:
                return None
            else:
                return self.__getitem__(np.random.randint(0, self.size))


        sampled_grasps_per_query_flip_list = []
        sampled_grasps_per_query_flip_qt_list = []
        yzopp_mat = np.eye(4)
        yzopp_mat[1, 1] = -1
        yzopp_mat[2, 2] = -1
        # Do grasp flipping here for acronym dataset
        for grasp_id in range(data[1].shape[0]):
            sampled_grasps_per_query_point_flip = np.matmul(yzopp_mat, data[1][grasp_id])
            sampled_grasps_per_query_point_flip_qt = utils.convert_transformation_matrix_to_qt(sampled_grasps_per_query_point_flip, numpy=True)
            sampled_grasps_per_query_flip_list.append(sampled_grasps_per_query_point_flip)
            sampled_grasps_per_query_flip_qt_list.append(sampled_grasps_per_query_point_flip_qt)
        sampled_grasps_per_query_point = np.array(sampled_grasps_per_query_flip_list)
        # sampled_grasps_per_query_point_qt = np.array(sampled_grasps_per_query_flip_qt_list)

        gt_control_points = utils.transform_control_points_numpy(
            sampled_grasps_per_query_point, self.opt.num_grasps_per_object, mode='rt', dense=False)
        
        pc = copy.deepcopy(data[0])
        pc[:,:,1] = -pc[:,:,1]
        pc[:,:,2] = -pc[:,:,2]


        # # Also make teh evaluator to be equivariant. This will also facilitate the learning process. You could turn it on by uncommenting
        # if self.equivariant:
        #     # align the point clouds and grasp pose such that the grasp approach direction is aligned with the y axis
        #     gt_grasp_approach_vector = (gt_control_points[:, 2] + gt_control_points[
        #                                                           :,
        #                                                           3]) / 2 - gt_control_points[
        #                                                                     :, 0]
        #     gt_grasp_approach_vector_normalize = gt_grasp_approach_vector / np.linalg.norm(gt_grasp_approach_vector,
        #                                                                                    axis=1).reshape(-1, 1)
        #     # compute the rotation matrix
        #     for grasp_id in range(data[1].shape[0]):
        #         # compute the grasp approach direction:
        #
        #         pc[grasp_id] = utils.align_pc_approach(pc[grasp_id], gt_grasp_approach_vector_normalize[grasp_id], pcmode="single")
        #         gt_control_points[grasp_id] = utils.align_pc_approach(gt_control_points[grasp_id], gt_grasp_approach_vector_normalize[grasp_id], pcmode="single")


        meta = {}
        meta['pc'] = pc
        meta['grasp_qt'] = gt_control_points[:, :, :3]
        meta['labels'] = data[2]
        meta['deg_weight'] = np.array([])
        return meta

    def __len__(self):
        return self.size

    def get_uniform_evaluator_data(self, path):

        pos_grasps, neg_grasps, obj_mesh, point_clouds, camera_poses_for_prerendered_point_clouds, hard_negative_grasps = self.read_grasp_file(
            path)

        output_pcs = []
        output_grasps = []
        output_labels = []
        positive_clusters = self.sample_grasp_indexes(self.num_positive, pos_grasps
                                                      )
        negative_clusters = self.sample_grasp_indexes(self.num_flex_negative,
                                                      neg_grasps)
        hard_neg_candidates = []
        # fill in positive examples.

        if hard_negative_grasps.size == 0:
            # If queue does not have enough data, fill it up with hard negative examples from the positives.
            if path not in self.collision_hard_neg_queue.keys() or len(
                    self.collision_hard_neg_queue[path]) < self.num_hard_negative:
                hard_neg_candidates = self.get_hard_neg_grasps(positive_clusters, negative_clusters, pos_grasps, neg_grasps)
                actually_hard_neg_candidates = []
                if path not in self.collision_hard_neg_queue.keys():
                    self.collision_hard_neg_queue[path] = []
                # hard negatives are perturbations of correct grasps.
                collisions, heuristic_qualities = utils.evaluate_grasps(
                    hard_neg_candidates, obj_mesh)
                hard_neg_mask = collisions | (heuristic_qualities < 0.001)
                hard_neg_indexes = np.where(hard_neg_mask)[0].tolist()
                np.random.shuffle(hard_neg_indexes)
                for index in hard_neg_indexes:
                    actually_hard_neg_candidates.append(
                        hard_neg_candidates[index])
                actually_hard_neg_candidates += self.collision_hard_neg_queue[path]
                self.collision_hard_neg_queue[path] = actually_hard_neg_candidates
                random.shuffle(self.collision_hard_neg_queue[path])
            # Adding hard neg
            for i in range(self.num_hard_negative):
                grasp = self.collision_hard_neg_queue[path][i]
                output_grasps.append(grasp)
                output_labels.append(0)

            self.collision_hard_neg_queue[path] = self.collision_hard_neg_queue[
                path][self.num_hard_negative:]

        else:
            np.random.shuffle(hard_negative_grasps)
            for i in range(self.num_hard_negative):
                grasp = hard_negative_grasps[i]
                output_grasps.append(grasp)
                output_labels.append(0)

        # Adding positive grasps
        for positive_cluster in positive_clusters:
            selected_grasp = pos_grasps[positive_cluster[0]][
                positive_cluster[1]]
            output_grasps.append(selected_grasp)
            output_labels.append(1)

        # Adding flex neg
        if len(negative_clusters) != self.num_flex_negative:
            raise ValueError(
                'negative clusters should have the same length as num_flex_negative {} != {}'
                .format(len(negative_clusters), self.num_flex_negative))

        for negative_cluster in negative_clusters:
            selected_grasp = neg_grasps[negative_cluster[0]][
                negative_cluster[1]]
            output_grasps.append(selected_grasp)
            output_labels.append(0)
        point_cloud_indices = self.sample_point_clouds(self.opt.num_grasps_per_object, point_clouds)
        output_grasps = np.matmul(camera_poses_for_prerendered_point_clouds[point_cloud_indices], output_grasps)

        output_pcs = point_clouds[point_cloud_indices]  # np.asarray(output_pcs, dtype=np.float32)
        output_grasps = np.asarray(output_grasps, dtype=np.float32)
        output_labels = np.asarray(output_labels, dtype=np.int32)
        return output_pcs, output_grasps, output_labels

    def sample_point_clouds(self, num_samples, point_clouds):
        num_point_clouds = len(point_clouds)
        if num_point_clouds == 0:
            raise NoPositiveGraspsException

        replace_point_cloud_indices = num_samples > num_point_clouds
        point_cloud_indices = np.random.choice(range(num_point_clouds),
                                               size=num_samples,
                                               replace=replace_point_cloud_indices).astype(np.int32)

        return point_cloud_indices

    def get_hard_neg_grasps(self, positive_clusters, negative_clusters, pos_grasps, neg_grasps):
        hard_neg_candidates = []
        # fill in positive examples.

        for clusters, grasps in zip(
                [positive_clusters, negative_clusters], [pos_grasps, neg_grasps]):
            for cluster in clusters:
                selected_grasp = grasps[cluster[0]][cluster[1]]
                hard_neg_candidates += utils.perturb_grasp(
                    selected_grasp,
                    self.collision_hard_neg_num_perturbations,
                    self.collision_hard_neg_min_translation,
                    self.collision_hard_neg_max_translation,
                    self.collision_hard_neg_min_rotation,
                    self.collision_hard_neg_max_rotation,
                )
        return hard_neg_candidates

    def read_grasp_file(self, path, return_all_grasps=False):
        file_name = path
        if self.caching and file_name in self.cache.keys():
            pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds, hard_negative_grasps = copy.deepcopy(
                self.cache[file_name])
            return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds, hard_negative_grasps

        pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds, hard_negative_grasps = self.read_object_grasp_data(
            path,
            return_all_grasps=return_all_grasps)

        if self.caching:
            self.cache[file_name] = (pos_grasps, neg_grasps, cad, point_clouds,
                                     camera_poses_for_prerendered_point_clouds, hard_negative_grasps)
            return copy.deepcopy(self.cache[file_name])

        return pos_grasps, neg_grasps, cad, point_clouds, camera_poses_for_prerendered_point_clouds, hard_negative_grasps

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
        mesh_file = os.path.join(self.opt.mesh_folder, json_dict['mesh/file'].split('/')[-1])
        object_model = Object(mesh_file)
        object_model.rescale(json_dict['mesh/scale'])

        object_model = object_model.mesh
        object_mean = np.mean(object_model.vertices, 0, keepdims=1)

        object_model.vertices -= object_mean

        grasps = np.asarray(json_dict['grasps/transformations'])

        try:
            hard_negative_grasps = np.asarray(json_dict['grasps/hard_negative'])
        except:
            hard_negative_grasps = np.empty(0)
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

            output_grasps = np.asarray(output_grasps,dtype=object)

            return output_grasps

        if not return_all_grasps:
            positive_grasps = cluster_grasps(
                positive_grasps)
            negative_grasps = cluster_grasps(
                negative_grasps)
        return positive_grasps, negative_grasps, object_model, point_clouds, camera_poses_for_prerendered_point_clouds, hard_negative_grasps
