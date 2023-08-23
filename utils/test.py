from options.test_options import TestOptions
from models import create_model
import random
import numpy as np


def run_test(epoch=-1, name="", writer=None, dataset_test=None, maxtest=50):
    print('Running Test')
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    opt.name = name

    model = create_model(opt)
    # test
    point_clouds = []
    numtest = len(dataset_test)

    if maxtest >= numtest:
        maxtest = numtest

    vis_grasp_id = random.randint(0, maxtest)

    for i, data in enumerate(dataset_test):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
        if i == vis_grasp_id:
            # point_clouds.append(model.get_random_grasp_and_point_cloud())
            point_clouds.append(model.get_random_grasp_and_point_cloud())
            break
    writer.calculate_accuracy()
    writer.print_acc(epoch)
    writer.plot_acc(epoch)
    writer.plot_grasps(point_clouds, epoch)
    writer.reset_counter()


if __name__ == '__main__':
    run_test()
