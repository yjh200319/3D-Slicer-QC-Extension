#!/usr/bin/env python
import argparse
import os
import logging
import onnxruntime
import multiprocessing
from utils import get_subj_data, collate_inferences

import sys
print("Python executable:", sys.executable)
print("Python version:", sys.version)

WEIGHTS = os.path.join(os.path.dirname(__file__), 'weights/model.FINAL.onnx')
print("Weights file: ", WEIGHTS)
logger = logging.getLogger()


def parse_arguments():
    parser = argparse.ArgumentParser(description="Image processing script")

    # input_group = parser.add_mutually_exclusive_group(required=True)

    # input_group.add_argument("-i", "--image_paths",
    #                          metavar="image_paths",
    #                          type=str,
    #                          nargs="+",
    #                          help="List of image paths to process")

    # parser.add_argument("-i", "--image_paths",
    #                     dest="image_paths",
    #                     type=str,
    #                     default=r"D:\NIMG_data\Caffine\sub-015\t1w.nii.gz",
    #                     help="Optional path")

    # input_group.add_argument("-f", "--inputs_file",
    #                          metavar="inputs_file",
    #                          type=str,
    #                          help="Path to a file containing a list of image paths")

    parser.add_argument("-c", "--csv_filepath",
                        dest="csv_filepath",
                        type=str,
                        default=None,
                        help="Optional path to a CSV file")

    parser.add_argument("-m", "--mc_runs",
                        dest="mc_runs",
                        type=int,
                        default=10,
                        help="Number of Monte Carlo runs")

    parser.add_argument("-q", "--queue_max_items",
                        dest="queue_max_items",
                        type=int,
                        default=10,
                        help="Number of items in the queue")

    parser.add_argument("-s", "--seed",
                        dest="seed",
                        type=int,
                        default=1010,
                        help="Seed for random generation")

    parser.add_argument("-v", "--verbose",
                        dest="verbose",
                        action="store_true",
                        help="Verbose output if set")

    return parser.parse_args()


def load_image_paths(image_queue, image_paths):
    for image_path in image_paths:
        X = get_subj_data(image_path)
        image_queue.put((image_path, X))
    image_queue.put(None)  # Signal the end of the queue


def main(image_paths, csv_filepath=None, mc_runs=10, max_items=10, verbose=False):
    from sys import stdout
    if verbose:
        logger.info('Loading model : {}'.format(WEIGHTS))

    image_queue = multiprocessing.Queue(max_items)
    producer = multiprocessing.Process(target=load_image_paths, args=(image_queue, image_paths))
    producer.start()

    # https://github.com/onnx/onnx/issues/3753
    sess = onnxruntime.InferenceSession(
        WEIGHTS, disabled_optimizers=["EliminateDropout"],
        # providers=["CUDAExecutionProvider"]
    )
    input_name = sess.get_inputs()[0].name
    csv_output = ["image_path,inferred_class,probability"]

    while True:
        item = image_queue.get()
        if item is None:
            break
        image_path, X = item

        predictions = []
        for iter, mc in enumerate(range(0, mc_runs)):
            logger.info('Working on mc run {}'.format(mc))

            y = sess.run(None, {input_name: X})[0][0]
            predictions.append(y)

            if verbose:
                inferred_class, p, c, s = collate_inferences(predictions)
                stdout.write("iter {}, predicted class: {} with {:.1f}% probability ({} out of {})\n".format(
                    iter, inferred_class, p, c, s)
                )
                stdout.flush()

        inferred_class, p, _, _ = collate_inferences(predictions)
        csv_output.append("{},{},{}".format(image_path, inferred_class, p))
        # print(csv_output[-1])
        return csv_output[-1]

    if csv_filepath is not None:
        with open(csv_filepath, "w") as fd:
            fd.write(os.linesep.join(csv_output) + os.linesep)
        print("wrote results to {}".format(csv_filepath))

    producer.join()


if __name__ == "__main__":
    import sys
    import os
    print(">>> Python executable:", sys.executable)
    print(">>> Python version:", sys.version)
    print(">>> os module path:", os.__file__)
    print(">>> sys.path:", sys.path)

    # logging.basicConfig(level=logging.ERROR)
    args = parse_arguments()
    onnxruntime.set_seed(args.seed)
    # print(args.inputs_file, '111')

    # if args.inputs_file:
    #     with open(args.inputs_file, 'r') as f:
    #         image_paths = [line.strip() for line in f.readlines()]
    # else:
    #     image_paths = args.image_paths
    #     print(args.image_paths, '===')
    # print(image_paths, 'xxxx')
    # image_paths = ['D:\\NIMG_data\\Caffine\\sub-015\\t1w.nii.gz']

    # 从 stdin 读取输入
    input_str = sys.stdin.read()

    # 处理数据（示例：转换为大写）
    # processed_str = input_str.upper()
    processed_str = input_str
    image_paths = [processed_str]
    # 结果输出到 stdout
    result = main(image_paths, args.csv_filepath, args.mc_runs, args.queue_max_items, args.verbose)

    print(result)
