## @package lmdb_create_example
# Module caffe2.python.examples.lmdb_create_example
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import numpy as np

import lmdb
from caffe2.proto import caffe2_pb2
from caffe2.python import workspace, model_helper
import os, glob, random
import skimage.io
import skimage.transform

'''
Simple example to create an lmdb database of random image data and labels.
This can be used a skeleton to write your own data import.

It also runs a dummy-model with Caffe2 that reads the data and
validates the checksum is same.
'''


def rescale(img, input_height, input_width):
    aspect = img.shape[1] / float(img.shape[0])
    if aspect > 1:
        return skimage.transform.resize(img, (input_width, int(aspect * input_height)))
    elif aspect < 1:
        return skimage.transform.resize(img, (int(input_width / aspect), input_height))
    else:
        return skimage.transform.resize(img, (input_width, input_height))


def crop_center(img, cropx, cropy):
    y, x, c = img.shape
    startx = x // 2 - (cropx // 2)
    starty = y // 2 - (cropy // 2)
    return img[starty:starty + cropy, startx:startx + cropx]


def prepare_image(img_path):
    img = skimage.io.imread(img_path)
    img = skimage.img_as_float(img)
    img = rescale(img, 227, 227)
    img = crop_center(img, 227, 227)
    img = img.swapaxes(1, 2).swapaxes(0, 1)  # HWC to CHW dimension
    img = img[(2, 1, 0), :, :]  # RGB to BGR color order
    img = img * 255 - 128  # Subtract mean = 128
    return img.astype(np.float32)


def create_db(input_dir, output_file):
    print(">>> Write database...")
    LMDB_MAP_SIZE = 1 << 40   # MODIFY

    categories = {"B": 0, "M": 1}
    class_position = 1

    env = lmdb.open(output_file, map_size=LMDB_MAP_SIZE)
    X_train = list(sorted(glob.glob(os.path.join(input_dir,  "*.png"))))
    y_train = [categories.get(os.path.basename(path).strip().replace("-", "_").split("_")[class_position], -1)
        for path in X_train]
    order = list(range(len(X_train)))
    checksum = 0
    with env.begin(write=True) as txn:
        for i in order:
            #print(i)
            image = prepare_image(X_train[i])
            label = y_train[i]
            print(X_train[i])

            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(image.shape)
            img_tensor.data_type = 1

            flatten_img = image.reshape(np.prod(image.shape))
            img_tensor.float_data.extend(flatten_img)

            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2
            label_tensor.int32_data.append(label)
            txn.put(
                '{}'.format(i).encode('ascii'),
                tensor_protos.SerializeToString()
            )

            checksum += np.sum(image) * label
            if (i % 16 == 0):
                print("Inserted {} rows".format(i))


        '''
        for j in range(0, 128):
            # MODIFY: add your own data reader / creator
            label = j % 10
            width = 64
            height = 32

            img_data = np.random.rand(3, width, height)
            # ...

            # Create TensorProtos
            tensor_protos = caffe2_pb2.TensorProtos()
            img_tensor = tensor_protos.protos.add()
            img_tensor.dims.extend(img_data.shape)
            img_tensor.data_type = 1

            flatten_img = img_data.reshape(np.prod(img_data.shape))
            img_tensor.float_data.extend(flatten_img)

            label_tensor = tensor_protos.protos.add()
            label_tensor.data_type = 2
            label_tensor.int32_data.append(label)
            txn.put(
                '{}'.format(j).encode('ascii'),
                tensor_protos.SerializeToString()
            )

            checksum += np.sum(img_data) * label
            if (j % 16 == 0):
                print("Inserted {} rows".format(j))
        '''
    print("Checksum/write: {}".format(int(checksum)))
    return checksum


def read_db_with_caffe2(db_file, expected_checksum):
    print(">>> Read database...")
    model = model_helper.ModelHelper(name="lmdbtest")
    batch_size = 32
    data, label = model.TensorProtosDBInput(
        [], ["data", "label"], batch_size=batch_size,
        db=db_file, db_type="lmdb")

    checksum = 0

    workspace.RunNetOnce(model.param_init_net)
    workspace.CreateNet(model.net)

    for _ in range(0, 4):
        workspace.RunNet(model.net.Proto().name)

        img_datas = workspace.FetchBlob("data")
        labels = workspace.FetchBlob("label")
        for j in range(batch_size):
            checksum += np.sum(img_datas[j, :]) * labels[j]

    print("Checksum/read: {}".format(int(checksum)))
    assert np.abs(expected_checksum - checksum < 0.1), \
        "Read/write checksums dont match"


def main():
    parser = argparse.ArgumentParser(
        description="Example LMDB creation"
    )
    parser.add_argument("--input_dir", type=str, default='/home/murilo/dataset/BreaKHis_v1/folds/fold1/test/40X',
                        help="Path to write the database to")

    parser.add_argument("--output_file", type=str, default='/home/murilo/data/tutorial_data/BreakHis/breakhis40x-test-nchw-lmdb',
                        help="Path to write the database to")

    args = parser.parse_args()
    checksum = create_db(args.input_dir, args.output_file)

    # For testing reading:
    #read_db_with_caffe2(args.output_file, checksum)


if __name__ == '__main__':
    main()