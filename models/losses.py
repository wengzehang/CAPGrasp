import torch

def D(x, alpha_range):
    if 0 < x <= alpha_range:
        # x_tensor = torch.tensor(x, dtype=torch.float).cuda()
        # alpha_range_tensor = torch.tensor(alpha_range, dtype=torch.float).cuda()
        # return (1/alpha_range) * (math.log(alpha_range) - torch.log(x))
        return 1.0
    else:
        return 0.0


def control_point_l1_loss(pred_control_points,
                          gt_control_points,
                          device="cpu"):
    """
      Computes the l1 loss between the predicted control points and the
      groundtruth control points on the gripper.
    """
    error = torch.sum(torch.abs(pred_control_points - gt_control_points), -1)
    error = torch.mean(error, -1)

    return torch.mean(error)


def classification_loss(pred_logit,
                        gt,
                        device="cpu"):
    """
      Computes the cross entropy loss. Returns cross entropy loss .
    """
    classification_loss = torch.nn.functional.binary_cross_entropy_with_logits(
        pred_logit, gt)
    return classification_loss


def min_distance_loss(pred_control_points,
                      gt_control_points,
                      device="cpu"):
    """
    Computes the minimum distance (L1 distance)between each gt control point 
    and any of the predicted control points.

    Args: 
      pred_control_points: tensor of (N_pred, M, 4) shape. N is the number of
        grasps. M is the number of points on the gripper.
      gt_control_points: (N_gt, M, 4)
    """
    pred_shape = pred_control_points.shape
    gt_shape = gt_control_points.shape

    if len(pred_shape) != 3:
        raise ValueError(
            "pred_control_point should have len of 3. {}".format(pred_shape))
    if len(gt_shape) != 3:
        raise ValueError(
            "gt_control_point should have len of 3. {}".format(gt_shape))
    if pred_shape != gt_shape:
        raise ValueError("shapes do no match {} != {}".format(
            pred_shape, gt_shape))

    error = pred_control_points.unsqueeze(1) - gt_control_points.unsqueeze(0)
    error = torch.sum(torch.abs(error),
                      -1)  # L1 distance of error (N_pred, N_gt, M)
    error = torch.mean(
        error, -1)  # average L1 for all the control points. (N_pred, N_gt)

    min_distance_error, closest_index = error.min(
        0)
    return torch.mean(min_distance_error)


def kl_divergence(mu, log_sigma, device="cpu"):
    """
      Computes the kl divergence for batch of mu and log_sigma.
    """
    return torch.mean(
        -.5 * torch.sum(1. + log_sigma - mu**2 - torch.exp(log_sigma), dim=-1))


