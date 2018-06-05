import numpy as np
from caffe2.python import core, workspace, model_helper, optimizer, brew
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.proto import caffe2_pb2
import matplotlib.pyplot as plt
from cancer_dataset import CancerDataset
import operator
import os
import skimage.io
from PIL import Image

PREDICT_NET = "/usr/local/lib/python2.7/dist-packages/caffe2/python/models/squeezenet/predict_net.pb"
INIT_NET = "/usr/local/lib/python2.7/dist-packages/caffe2/python/models/squeezenet/init_net.pb"


def AddPredictNet(model, predict_net_path):
    predict_net_proto = caffe2_pb2.NetDef()
    with open(predict_net_path, "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)
    # Fix dimension incompatibility
    model.Squeeze("softmaxout", "softmax", dims=[2, 3])


def AddInitNet(model, init_net_path, out_dim=2, params_to_learn=None):
    init_net_proto = caffe2_pb2.NetDef()
    with open(init_net_path, "rb") as f:
        init_net_proto.ParseFromString(f.read())

    # Define params to learn in the model.
    for op in init_net_proto.op:
        param_name = op.output[0]
        if params_to_learn is None or op.output[0] in params_to_learn:
            tags = (ParameterTags.WEIGHT if param_name.endswith("_w")
                    else ParameterTags.BIAS)
            model.create_param(
                param_name=param_name,
                shape=op.arg[0],
                initializer=initializers.ExternalInitializer(),
                tags=tags,
            )

    # Remove conv10_w, conv10_b initializers at (50, 51)
    init_net_proto.op.pop(51)
    init_net_proto.op.pop(50)

    # Add new initializers for conv10_w, conv10_b
    model.param_init_net = core.Net(init_net_proto)
    model.param_init_net.XavierFill([], "conv10_w", shape=[out_dim, 512, 1, 1])
    model.param_init_net.ConstantFill([], "conv10_b", shape=[out_dim])


def AddTrainingOperators(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], "xent")
    loss = model.AveragedLoss(xent, "loss")
    brew.accuracy(model, [softmax, label], "accuracy")
    model.AddGradientOperators([loss])
    opt = optimizer.build_sgd(model, base_learning_rate=0.1)
    for param in model.GetOptimizationParamInfo():
        opt(model.net, model.param_init_net, param)


def ModelConstruction(output_dim=2):
    train_model = model_helper.ModelHelper("train_net")
    AddPredictNet(train_model, PREDICT_NET)
    AddInitNet(train_model, INIT_NET, out_dim=output_dim,
               params_to_learn=["conv10_w", "conv10_b"])  # Use None to learn everything.
    AddTrainingOperators(train_model, "softmax", "label")
    return train_model


def SetDeviceOption(model, device_option):
    # Clear op-specific device options and set global device option.
    for net in ("net", "param_init_net"):
        net_def = getattr(model, net).Proto()
        net_def.device_option.CopyFrom(device_option)
        for op in net_def.op:
            # Some operators are CPU-only.
            if op.output[0] not in ("optimizer_iteration", "iteration_mutex"):
                op.ClearField("device_option")
                op.ClearField("engine")
        setattr(model, net, core.Net(net_def))


def SetRunGPU(train_model):
    device_option = caffe2_pb2.DeviceOption()
    device_option.device_type = caffe2_pb2.CUDA
    device_option.cuda_gpu_id = 0
    SetDeviceOption(train_model, device_option)
    return device_option


def Finetuning(dataset, train_model, device_option, epochs):
    workspace.ResetWorkspace()

    # Initialization.
    # train_dataset = CancerDataset("train")
    for image, label, file in dataset.read(batch_size=1):
        workspace.FeedBlob("data", image, device_option=device_option)
        workspace.FeedBlob("label", label, device_option=device_option)
        break
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    # Main loop.
    batch_size = 10
    print_freq = 10
    losses = []
    for epoch in range(epochs):
        for index, (image, label, file) in enumerate(dataset.read(batch_size)):
            # im_plt = image[0]
            # im_plt = im_plt.swapaxes(0, 1).swapaxes(1, 2)  # CHW to HWC dimension
            # im_plt = im_plt + 128
            # im_plt = np.uint8(im_plt)
            # plt.figure()
            # skimage.io.imshow(im_plt)
            # plt.show()

            # plt.figure()
            # skimage.io.imshow(dataset.X_train[0])
            # plt.show()

            workspace.FeedBlob("data", image, device_option=device_option)
            workspace.FeedBlob("label", label, device_option=device_option)
            workspace.RunNet(train_model.net)
            accuracy = float(workspace.FetchBlob("accuracy"))
            #softmax = workspace.FetchBlob("softmax")
            loss = workspace.FetchBlob("loss").mean()
            losses.append(loss)
            if index % print_freq == 0:
                print("[{}][{}/{}] loss={}, accuracy={}".format(
                    epoch, index, int(len(dataset.X_train) / batch_size),
                    loss, accuracy))

    ### test (to do)
    # arg_scope = {"order": "NCHW"}
    # test_model = model_helper.ModelHelper(
    #    name="cancer_test", arg_scope=arg_scope, init_params=False)
    # workspace.RunNetOnce(test_model.param_init_net)
    # workspace.CreateNet(test_model.net, overwrite=True)
    ### end test

    return losses


def PlotLearintProgress(losses):
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.grid("on")


def DeployModel(device_option):
    deploy_model = model_helper.ModelHelper("deploy_net")
    AddPredictNet(deploy_model, PREDICT_NET)
    SetDeviceOption(deploy_model, device_option)
    return deploy_model


def TestModel(dataset, deploy_model, device_option):
    test_accuracy = 0
    hits = 0
    ntests = len(dataset.y_test)

    for index, (image, label, file) in enumerate(dataset.read(batch_size=1, shuffle=False, train=False)):
        # for test_index in range(ntests):
        # image, label = dataset.getitem(test_index, False)
        # image = image[np.newaxis, :]
        workspace.FeedBlob("data", image, device_option=device_option)
        workspace.FeedBlob("label", label, device_option=device_option)
        workspace.RunNetOnce(deploy_model.net)
        result = workspace.FetchBlob("softmax")[0]
        curr_pred, curr_conf = max(enumerate(result), key=operator.itemgetter(1))

        if curr_pred == label[0]:
            hits = hits + 1

        print("Image={} Label={} Prediction={} Confidence={}".
              format(os.path.basename(file[0]), label[0], curr_pred, curr_conf))

    calc_accuracy = float(hits) / ntests
    print("Hits: {}/{}".format(hits, ntests))
    print("Test accuracy: {}".format(calc_accuracy))

    # plt.figure()
    # skimage.io.imshow(dataset.getitem(0)[0])
    # plt.show()

    # plt.figure()
    # skimage.io.imshow(dataset.X_test[0])
    # plt.show()


def main():
    print("Start Process")
    print("Finetuning ")

    nr_classes = 2
    dataset_mode = 1  # 1 => 2 Class    2 => 8 Class  see cancer_dataset.py
    epochs = 10

    train_model = ModelConstruction(output_dim=nr_classes)
    device_option = SetRunGPU(train_model)
    dataset = CancerDataset(dataset_mode=dataset_mode)
    losses = Finetuning(dataset, train_model, device_option, epochs)
    PlotLearintProgress(losses)
    deploy_model = DeployModel(device_option)
    TestModel(dataset, deploy_model, device_option)


if __name__ == '__main__':
    main()
