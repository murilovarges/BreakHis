import numpy as np
from caffe2.python import core, workspace, model_helper, optimizer, brew
from caffe2.python.modeling import initializers
from caffe2.python.modeling.parameter_info import ParameterTags
from caffe2.proto import caffe2_pb2
from cancer_dataset import CancerDataset
import operator
import os
import argparse
import logging
import csv
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from helpers import *

# Logger
logging.basicConfig(filename='finetuning.log')
log = logging.getLogger("finetuning")
log.setLevel(logging.INFO)


def AddPredictNet(model, predict_net_path):
    predict_net_proto = caffe2_pb2.NetDef()
    with open(predict_net_path, "rb") as f:
        predict_net_proto.ParseFromString(f.read())
    model.net = core.Net(predict_net_proto)
    # Fix dimension incompatibility
    model.Squeeze("softmaxout", "softmax", dims=[2, 3])


def AddInitNet(model, args, params_to_learn=None):
    init_net_proto = caffe2_pb2.NetDef()
    with open(args.init_net, "rb") as f:
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
    model.param_init_net.XavierFill([], "conv10_w", shape=[args.nr_classes, 512, 1, 1])
    model.param_init_net.ConstantFill([], "conv10_b", shape=[args.nr_classes])


def AddTrainingOperators(model, softmax, label):
    xent = model.LabelCrossEntropy([softmax, label], "xent")
    loss = model.AveragedLoss(xent, "loss")
    brew.accuracy(model, [softmax, label], "accuracy")
    model.AddGradientOperators([loss])
    opt = optimizer.build_sgd(model, base_learning_rate=0.1)
    for param in model.GetOptimizationParamInfo():
        opt(model.net, model.param_init_net, param)


def ModelConstruction(args):
    train_model = model_helper.ModelHelper("train_net")
    AddPredictNet(train_model, args.predict_net)
    AddInitNet(train_model, args,
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


def Finetuning(dataset, train_model, device_option, epochs, batch_size):
    workspace.ResetWorkspace()
    # Initialization
    for image, label, file in dataset.read(batch_size=1):
        workspace.FeedBlob("data", image, device_option=device_option)
        workspace.FeedBlob("label", label, device_option=device_option)
        break
    workspace.RunNetOnce(train_model.param_init_net)
    workspace.CreateNet(train_model.net, overwrite=True)

    # Main loop
    print_freq = 10
    losses = []
    for epoch in range(epochs):
        for index, (image, label, file) in enumerate(dataset.read(batch_size)):
            workspace.FeedBlob("data", image, device_option=device_option)
            workspace.FeedBlob("label", label, device_option=device_option)
            workspace.RunNet(train_model.net)
            accuracy = float(workspace.FetchBlob("accuracy"))
            loss = workspace.FetchBlob("loss").mean()
            losses.append(loss)
            if index % print_freq == 0:
                print("[{}][{}/{}] loss={}, accuracy={}".format(
                    epoch, index, int(len(dataset.X_train) / batch_size),
                    loss, accuracy))

    return losses


def PlotLearintProgress(losses):
    plt.plot(losses)
    plt.xlabel("iterations")
    plt.ylabel("loss")
    plt.grid("on")


def DeployModel(device_option, predict_net):
    deploy_model = model_helper.ModelHelper("deploy_net")
    AddPredictNet(deploy_model, predict_net)
    SetDeviceOption(deploy_model, device_option)
    return deploy_model


def TestModel(dataset, deploy_model, device_option, args):
    hits = 0
    ntests = len(dataset.y_test)
    predictions = []
    p_image = []
    l_image = []

    for index, (image, label, file) in enumerate(dataset.read(batch_size=1, shuffle=False, train=False)):
        workspace.FeedBlob("data", image, device_option=device_option)
        workspace.FeedBlob("label", label, device_option=device_option)
        workspace.RunNetOnce(deploy_model.net)
        result = workspace.FetchBlob("softmax")[0]
        curr_pred, curr_conf = max(enumerate(result), key=operator.itemgetter(1))

        if curr_pred == label[0]:
            hits = hits + 1

        p_image.extend([curr_pred])
        l_image.extend([label[0]])

        predictions.append({
            'image': file[0],
            'label': label[0],
            'prediction': curr_pred,
            'confidence': curr_conf
        })

        print("Image={} Label={} Prediction={} Confidence={}".
              format(os.path.basename(file[0]), label[0], curr_pred, curr_conf))

    # Saving confunsion matrix file
    cnf_matrix = confusion_matrix(l_image, p_image)
    cnf_file = os.path.join(args.output_dir, args.experiment_name, 'conf_matrix.txt')
    np.savetxt(cnf_file, cnf_matrix, delimiter=",", fmt='%1.3f')

    #
    hp = Helpers()
    class_names = np.array(['Benign', 'Malignant'])
    # Plot non-normalized confusion matrix clip
    plt.figure()
    f = os.path.join(args.output_dir, args.experiment_name, 'ConfusionNonNormalized.png')
    #files.append(f)
    hp.plot_confusion_matrix(cnf_matrix, classes=class_names,
                                                 title='Confusion matrix, without normalization', show=False,
                                                 fileName=f)

    # Plot normalized confusion matrix clip
    plt.figure()
    f = os.path.join(args.output_dir, args.experiment_name, 'ConfusionNormalized.png')
    hp.plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
                                               title='Normalized confusion matrix', show=False, fileName=f)
    # save predictions for every image
    predictions_file = os.path.join(args.output_dir, args.experiment_name, 'predictions.txt')
    keys = predictions[0].keys()
    with open(predictions_file, 'w') as f:  # Just use 'w' mode in 3.x
        w = csv.DictWriter(f, keys)
        w.writeheader()
        for data in predictions:
            w.writerow(data)

    # Computing Clip Accuracy
    ac1 = accuracy_score(l_image, p_image)
    ac2 = accuracy_score(l_image, p_image, normalize=False)

    calc_accuracy = float(hits) / ntests
    msg = "Hits: {}/{}\n".format(hits, ntests)
    msg += "Test accuracy: {}\n".format(calc_accuracy)
    msg += "Ac1: {}\n".format(ac1)
    msg += "Ac2: {}".format(ac2)
    print(msg)

    accuracy_file = os.path.join(args.output_dir, args.experiment_name, 'accuracy.txt')
    SaveFileInformation(accuracy_file, msg)


def SaveFileInformation( file_name, msg):
    base_out_dir = os.path.dirname(os.path.abspath(file_name))
    if not os.path.exists(base_out_dir):
        os.makedirs(base_out_dir)

    file = open(file_name, 'w')
    file.write(str(msg))
    file.close()


def main():
    parser = argparse.ArgumentParser(
        description="BreakHis: breast cancer classification"
    )
    parser.add_argument("--predict_net", type=str, default='models/squeezenet/predict_net.pb',
                        help="Predict net file (net with operators")
    parser.add_argument("--init_net", type=str, default='models/squeezenet/init_net.pb',
                        help="Predict net file (net with weights")
    parser.add_argument("--experiment_name", type=str, default='BreakHis200X',
                        help="Name of the experiment")
    parser.add_argument("--dataset_name", type=str, default='img200',
                        help="Name of the dataset")
    parser.add_argument("--output_dir", type=str, default='/home/murilo/PycharmProjects/BreakHis/results',
                        help="Directory to store outputs")
    parser.add_argument("--nr_classes", type=int, default=2,
                        help="Number of classes")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Num epochs.")
    parser.add_argument("--batch_size", type=int, default=10,
                        help="Batch size, total over all GPUs")
    parser.add_argument("--test_trials", type=int, default=5,
                        help="Number of train/test trial")
    parser.add_argument("--dataset_shuffle_seeds", type=str, default='159,225,350,487,789',
                        help="Seeds for shuffle dataset")
    args = parser.parse_args()

    log.info(args)
    print(args)

    base_experiment_name = args.experiment_name
    data_set_shuffle_seeds = args.dataset_shuffle_seeds.split(',')
    for trial in range(args.test_trials):
        args.experiment_name = base_experiment_name + '_' + str(trial+1)

        # Save parameters
        parameters_file = os.path.join(args.output_dir, args.experiment_name, 'parameters.txt')
        SaveFileInformation(parameters_file, args)


        train_model = ModelConstruction(args)
        device_option = SetRunGPU(train_model)

        data_set_seed = int(data_set_shuffle_seeds[trial])
        dataset = CancerDataset(dataset_name=args.dataset_name, nr_classes=args.nr_classes, randon_state=data_set_seed)
        train_file = os.path.join(args.output_dir, args.experiment_name, 'train.txt')
        SaveFileInformation(train_file, dataset.X_train)
        test_file = os.path.join(args.output_dir, args.experiment_name, 'test.txt')
        SaveFileInformation(test_file, dataset.X_test)


        losses = Finetuning(dataset, train_model, device_option, args.num_epochs, args.batch_size)
        #PlotLearintProgress(losses)
        deploy_model = DeployModel(device_option, args.predict_net)
        TestModel(dataset, deploy_model, device_option, args)


if __name__ == '__main__':
    workspace.GlobalInit(['caffe2', '--caffe2_log_level=2'])
    main()
