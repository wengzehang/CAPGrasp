import torch
import torch.nn as nn
from torch.nn import init
from torch.optim import lr_scheduler
import torch.nn.functional as F
# from constrained_6dof_graspnet.models import losses
from models import losses
import sys
sys.path.append("/media/zehang/LaCie/zehang/ubuntu/project/orienGrasp/Pointnet2_PyTorch")
from pointnet2_ops_lib.pointnet2_ops import pointnet2_modules as pointnet2


def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':

        def lambda_rule(epoch):
            lr_l = 1.0 - max(
                0, epoch + 1 + 1 - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer,
                                        step_size=opt.lr_decay_iters,
                                        gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer,
                                                   mode='min',
                                                   factor=0.2,
                                                   threshold=0.01,
                                                   patience=5)
    else:
        return NotImplementedError(
            'learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def init_weights(net, init_type, init_gain):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1
                                     or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError(
                    'initialization method [%s] is not implemented' %
                    init_type)
        elif classname.find('batchnorm.BatchNorm') != -1:
            init.normal_(m.weight.data, 1.0, init_gain)
            init.constant_(m.bias.data, 0.0)

    net.apply(init_func)


def init_net(net, init_type, init_gain, gpu_ids):
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        net.cuda(gpu_ids[0])
        net = net.cuda()
        net = torch.nn.DataParallel(net, gpu_ids)
    if init_type != 'none':
        init_weights(net, init_type, init_gain)
    return net


def define_classifier(opt, gpu_ids, arch, init_type, init_gain, device):
    # for old trained model, (yaw=4,8,16, pitch=1), we don't have the vanilla flag
    try:
        # if no vanilla flag in the option args, it is old model trained with constrained.
        vanilla = opt.vanilla
    except:
        opt.vanilla = False

    if arch == 'vae':
        net = GraspSamplerVAE(model_scale=opt.model_scale, pointnet_radius=opt.pointnet_radius,
                              pointnet_nclusters=opt.pointnet_nclusters,
                              orientation_constrained_sampling=opt.orientation_constrained,
                              latent_size=opt.latent_size, vanilla=opt.vanilla, device=device)
    elif arch == 'gan':
        raise NotImplementedError
    elif arch == 'evaluator':
        if not 'evaluator_type' in opt or opt.evaluator_type == "pointnet++":
            net = GraspEvaluator(
                opt.model_scale, opt.pointnet_radius,
                opt.pointnet_nclusters, device
            )
        elif opt.evaluator_type == "DGCNN":
            raise NotImplementedError
        else:
            raise ValueError("Evaluator type not found")

    else:
        raise NotImplementedError('model name [%s] is not recognized' % arch)
    return init_net(net, init_type, init_gain, gpu_ids)


def define_loss(opt):
    if opt.arch == 'vae':
        kl_loss = losses.kl_divergence
        reconstruction_loss = losses.control_point_l1_loss
        return kl_loss, reconstruction_loss
    elif opt.arch == 'gan':
        reconstruction_loss = losses.min_distance_loss
        return reconstruction_loss
    elif opt.arch == 'evaluator':
        loss = losses.classification_loss
        return loss
    else:
        raise NotImplementedError("Loss not found")


class GraspSampler(nn.Module):
    def __init__(self, latent_size, device, orientation_constrained_sampling=False, vanilla=False, equivariant=True):
        super(GraspSampler, self).__init__()
        self.latent_size = latent_size
        self.device = device
        self.orientation_constrained_sampling = orientation_constrained_sampling
        self.vanilla = vanilla
        self.equivariant = equivariant

    def create_decoder(self, model_scale, pointnet_radius, pointnet_nclusters,
                       num_input_features):

        self.decoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)
        self.q = nn.Linear(model_scale * 1024, 4)
        self.t = nn.Linear(model_scale * 1024, 3)

    def decode(self, xyz, z, features=None):
        xyz_features = self.concatenate_z_with_pc(xyz,
                                                  z)
        if self.orientation_constrained_sampling and features is not None:
            if self.vanilla != True:
                xyz_features = torch.cat((xyz_features, features), -1)
        elif self.orientation_constrained_sampling and features is None:
            query_point_encoding = self.setup_query_point_feature(xyz.shape[0], xyz.shape[1])
            if self.vanilla != True:
                xyz_features = torch.cat((xyz_features, query_point_encoding), -1)

        xyz_features = xyz_features.transpose(-1, 1).contiguous()
        for module in self.decoder[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        x = self.decoder[1](xyz_features.squeeze(-1))
        predicted_qt = torch.cat(
            (F.normalize(self.q(x), p=2, dim=-1), self.t(x)), -1)

        return predicted_qt

    def concatenate_z_with_pc(self, pc, z):
        z.unsqueeze_(1)
        z = z.expand(-1, pc.shape[1], -1)
        return torch.cat((pc, z), -1)

    def get_latent_size(self):
        return self.latent_size

    def setup_query_point_feature(self, batch_size, num_points):
        query_point_feature = torch.zeros((batch_size, num_points, 1)).to(self.device)
        query_point_feature[:, -1] = 1
        return query_point_feature


class GraspSamplerVAE(GraspSampler):
    """Network for learning a generative VAE grasp-sampler
    """

    def __init__(self,
                 model_scale,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 orientation_constrained_sampling=False,
                 latent_size=2,
                 vanilla=False,
                 equivariant=True,
                 device="cpu"):
        super(GraspSamplerVAE, self).__init__(latent_size, device, orientation_constrained_sampling, vanilla, equivariant)
        extra_input_feature = 0

        if self.vanilla == True:
            extra_input_feature = 0 # for vanilla version, add nothing
        else:
            if self.orientation_constrained_sampling:
                if self.equivariant:
                    extra_input_feature = 1
                else:
                    raise ValueError(
                        'Please set the flag equivariant to be True. We do not provide GoNet implementation in this repo.')

            if self.equivariant:
                extra_input_feature = 1  # only for the angle range

        # self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters, 19+extra_input_feature)
        self.create_encoder(model_scale, pointnet_radius, pointnet_nclusters, 10+extra_input_feature)

        self.create_decoder(model_scale, pointnet_radius, pointnet_nclusters,
                            latent_size + 3 + extra_input_feature)
        self.create_bottleneck(model_scale * 1024, latent_size)

    def create_encoder(
            self,
            model_scale,
            pointnet_radius,
            pointnet_nclusters,
            num_input_features
    ):
        # The number of input features for the encoder is 19+k: the x, y, z
        # position of the point-cloud, the flattened 4x4=16 grasp pose matrix,
        # and, if k is 1, a 1/0 binary encoding representing which point we want to generate
        # grasps around

        # Note: Change to x,y,z+7d representation + k -> 10 + k

        self.encoder = base_network(pointnet_radius, pointnet_nclusters,
                                    model_scale, num_input_features)

    def create_bottleneck(self, input_size, latent_size):
        mu = nn.Linear(input_size, latent_size)
        logvar = nn.Linear(input_size, latent_size)
        self.latent_space = nn.ModuleList([mu, logvar])

    def encode(self, pc_xyz, grasps, extra_input_features):
        grasp_features = grasps.unsqueeze(1).expand(-1, pc_xyz.shape[1], -1)
        features = torch.cat(
            (pc_xyz, grasp_features),
            -1)
        if self.orientation_constrained_sampling:
            if extra_input_features is None:
                query_point_feature = self.setup_query_point_feature(pc_xyz.shape[0], pc_xyz.shape[1])
                features = torch.cat((features, query_point_feature), -1)
            else:
                if self.vanilla != True:
                    features = torch.cat((features, extra_input_features), -1)
        features = features.transpose(-1, 1).contiguous()
        for module in self.encoder[0]:
            pc_xyz, features = module(pc_xyz, features)
        return self.encoder[1](features.squeeze(-1))

    def bottleneck(self, z):
        return self.latent_space[0](z), self.latent_space[1](z)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, pc, grasp=None, features=None, train=True):
        if train:
            return self.forward_train(pc, grasp, features)
        else:
            return self.forward_test(pc, grasp, features)

    def forward_train(self, pc, grasp, features):
        z = self.encode(pc, grasp, features)
        mu, logvar = self.bottleneck(z)
        z = self.reparameterize(mu, logvar)
        qt = self.decode(pc, z, features)
        return qt,  mu, logvar

    def forward_test(self, pc, grasp, features):
        z = self.encode(pc, grasp, features)
        mu, _ = self.bottleneck(z)
        qt = self.decode(pc, mu, features)
        return qt

    def sample_latent(self, batch_size):
        return torch.randn(batch_size, self.latent_size).to(self.device)

    def generate_grasps(self, pc, z=None, features=None):
        if z is None:
            z = self.sample_latent(pc.shape[0])
        qt = self.decode(pc, z, features)
        return qt, z.squeeze()

    def generate_dense_latents(self, resolution):
        latents = torch.meshgrid(*[
            torch.linspace(-2, 2, resolution) for i in range(self.latent_size)
        ])
        return torch.stack([latents[i].flatten() for i in range(len(latents))],
                           dim=-1).to(self.device)


class GraspEvaluator(nn.Module):
    def __init__(self,
                 model_scale=1,
                 pointnet_radius=0.02,
                 pointnet_nclusters=128,
                 device="cpu"):
        super(GraspEvaluator, self).__init__()
        self.create_evaluator(pointnet_radius, model_scale, pointnet_nclusters)
        self.device = device

    def create_evaluator(self, pointnet_radius, model_scale,
                         pointnet_nclusters):
        self.evaluator = base_network(pointnet_radius, pointnet_nclusters,
                                      model_scale, 4)
        self.predictions_logits = nn.Linear(1024 * model_scale, 1)

    def evaluate(self, xyz, xyz_features):
        for module in self.evaluator[0]:
            xyz, xyz_features = module(xyz, xyz_features)
        return self.evaluator[1](xyz_features.squeeze(-1))

    def forward(self, pc, gripper_pc, features=None, train=True):
        pc, pc_features = self.merge_pc_and_gripper_pc(pc, gripper_pc)
        x = self.evaluate(pc, pc_features.contiguous())
        return self.predictions_logits(x)

    def merge_pc_and_gripper_pc(self, pc, gripper_pc):
        """
        Merges the object point cloud and gripper point cloud and
        adds a binary auxiliary feature that indicates whether each point
        belongs to the object or to the gripper.
        """
        pc_shape = pc.shape
        gripper_shape = gripper_pc.shape
        assert (len(pc_shape) == 3)
        assert (len(gripper_shape) == 3)
        assert (pc_shape[0] == gripper_shape[0])

        batch_size = pc_shape[0]

        l0_xyz = torch.cat((pc, gripper_pc), 1)
        labels = [
            torch.ones(pc.shape[1], 1, dtype=torch.float32),
            torch.zeros(gripper_pc.shape[1], 1, dtype=torch.float32)
        ]
        labels = torch.cat(labels, 0)
        labels.unsqueeze_(0)
        labels = labels.repeat(batch_size, 1, 1)

        l0_points = torch.cat([l0_xyz, labels.to(self.device)],
                              -1).transpose(-1, 1)
        return l0_xyz, l0_points

    def improve_grasps_sampling_based(self,
                                      pcs,
                                      grasp_qt,
                                      num_refine_steps, delta_translation=0.005):

        raise NotImplementedError

    def improve_grasps_gradient_based(self, pcs, grasp_qt, num_refine_steps, max_tensor_size=130):
        raise NotImplementedError


def base_network(pointnet_radius, pointnet_nclusters, scale, in_features):
    sa1_module = pointnet2.PointnetSAModule(
        npoint=pointnet_nclusters,
        radius=pointnet_radius,
        nsample=64,
        mlp=[in_features, 64 * scale, 64 * scale, 128 * scale])
    sa2_module = pointnet2.PointnetSAModule(
        npoint=32,
        radius=0.04,
        nsample=128,
        mlp=[128 * scale, 128 * scale, 128 * scale, 256 * scale])

    sa3_module = pointnet2.PointnetSAModule(
        mlp=[256 * scale, 256 * scale, 256 * scale, 512 * scale])

    sa_modules = nn.ModuleList([sa1_module, sa2_module, sa3_module])
    fc_layer = nn.Sequential(nn.Linear(512 * scale, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True),
                             nn.Linear(1024 * scale, 1024 * scale),
                             nn.BatchNorm1d(1024 * scale), nn.ReLU(True))
    return nn.ModuleList([sa_modules, fc_layer])

