# --------------------------------------------------------
# Octree-based Sparse Convolutional Neural Networks
# Copyright (c) 2022 Peng-Shuai Wang <wangps@hotmail.com>
# Licensed under The MIT License [see LICENSE for details]
# Written by Peng-Shuai Wang
# --------------------------------------------------------

import torch
import ocnn
from ocnn.octree import Octree, Points

class CustomTransform:
    r''' [MODIFIED SLIGHTLY FOR THIS PROJECT] A boilerplate class which transforms an input data for :obj:`ocnn`.
        The input data is first converted to :class:`Points`, then randomly transformed
        (if enabled), and converted to an :class:`Octree`.

        Args:
        depth (int): The octree depth.
        full_depth (int): The octree layers with a depth small than
            :attr:`full_depth` are forced to be full.
        augment (bool): If true, performs the data augmentation.
        angle (list): A list of 3 float values to generate random rotation angles.
        interval (list): A list of 3 float values to represent the interval of
            rotation angles.
        jittor (float): The maximum jitter values.
        orient_normal (str): Orient point normals along the specified axis, which is
            useful when normals are not oriented.
    '''

    def __init__(self, 
                 depth: int, 
                 full_depth: int, 
                 augment: bool, 
                 angle: list,
                 interval: list, 
                 jitter: float, 
                 **kwargs):
        
        super().__init__()

        # for octree building
        self.depth = depth
        self.full_depth = full_depth

        # for data augmentation
        self.augment = augment
        self.angle = angle
        self.interval = interval
        self.jitter = jitter

    def __call__(self, points: dict, idx: int):

        output = self.preprocess({"points": points}, idx)
        output = self.transform(output, idx)
        output['octree'] = self.points2octree(output['points'])
        return output

    def preprocess(self, sample: dict, idx: int):
        r''' Transforms :attr:`sample` to :class:`Points` and performs some specific
        transformations, like normalization.
        '''

        xyz = torch.from_numpy(sample.pop('points')).float()
        sample['points'] = Points(xyz, normals=None, features=None)

        # Need to normalize the point cloud into one unit sphere in [-0.8, 0.8]
        # See: https://github.com/octree-nn/ocnn-pytorch/blob/a0e2717427f79efaee82a680c0de7c445c7e3fb6/projects/datasets/modelnet40.py#L20
        bbmin, bbmax = sample['points'].bbox()
        sample['points'].normalize(bbmin, bbmax, scale=0.8)
        sample['points'].scale(torch.Tensor([0.8, 0.8, 0.8]))

        return sample

    def transform(self, sample: Points, idx: int):
        r''' Applies the general transformations provided by :obj:`ocnn`.
        '''

        # The augmentations including rotation, scaling, and jittering.
        points = sample['points']

        if self.augment:
            rng_angle, rng_jitter = self.rnd_parameters()
            points.rotate(rng_angle)
            points.translate(rng_jitter)

        # !!! NOTE: Clip the point cloud to [-1, 1] before building the octree
        inbox_mask = points.clip(min=-1, max=1)
        sample.update({'points': points, 'inbox_mask': inbox_mask})

        return sample


    def points2octree(self, points: Points):
        r''' Converts the input :attr:`points` to an octree.
        '''

        octree = Octree(self.depth, self.full_depth)
        octree.build_octree(points)
        return octree

    def rnd_parameters(self):
        r''' Generates random parameters for data augmentation.
        '''

        rnd_angle = [None] * 3
        for i in range(3):
            rot_num = self.angle[i] // self.interval[i]
            rnd = torch.randint(low=-rot_num, high=rot_num + 1, size=(1,))
            rnd_angle[i] = rnd * self.interval[i] * (3.14159265 / 180.0)

        rnd_angle = torch.cat(rnd_angle)

        rnd_jitter = torch.rand(3) * (2 * self.jitter) - self.jitter

        return rnd_angle, rnd_jitter


class CustomCollateBatch:
    r""" Merge a list of octrees and points into a batch.
  """

    def __init__(self, batch_size: int, merge_points: bool = False):
        self.merge_points = merge_points
        self.batch_size = batch_size

    def __call__(self, batch: list):
        assert type(batch) == list

        outputs = {}
        for key in batch[0].keys():
            outputs[key] = [b[key] for b in batch]

            # Merge a batch of octrees into one super octree
            if 'octree' in key:
                octree = ocnn.octree.merge_octrees(outputs[key])
                # NOTE: remember to construct the neighbor indices
                octree.construct_all_neigh()
                outputs[key] = octree

            # Merge a batch of points
            if 'points' in key and self.merge_points:
                outputs[key] = ocnn.octree.merge_points(outputs[key])

            # Convert the labels to a Tensor
            if 'target' in key:
                num_samples = len(outputs['target'])
                num_targets = outputs['target'][0].shape[0]
                target_reshape = torch.cat(outputs['target'])
                target_reshape = torch.reshape(target_reshape, (num_samples, num_targets))
                outputs['target'] = target_reshape

        return outputs


def load_octree_sample(points, idx, depth, full_depth, augment):
    """
    Loads a sample from a las file and converts it to octree format
    :param points: point cloud numpy array
    :param idx: sample idx
    :param depth: octree depth
    :param full_depth: octree full depth
    :param augment: whether to augment the data
    :return:
    """

    transform = CustomTransform(depth=depth,
                                full_depth=full_depth,
                                augment=augment,
                                angle=(0, 0, 5),  # x,y,z axes (currently a small rotation around z axis)
                                interval=(1, 1, 1),
                                jitter=0.125)

    sample = transform(points, idx=idx)

    return sample
