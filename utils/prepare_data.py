"""Script to prepare the pretraining and training datasets for training an OCR post-correction model.
For pretraining data, the script combines all uncorrected files into a single text file.
For training data, the script splits the manually corrected files into training, development, and testing sets.
The fraction of training data is controlled with the --training_frac option.
The remaining data is equally split between development and testing sets.
Copyright (c) 2021, Shruti Rijhwani
All rights reserved.
This source code is licensed under the BSD-style license found in the
LICENSE file in the root directory of this source tree. 
"""


import argparse
import os
import glob
import random
import logging

logging.basicConfig(format="%(message)s")


def prepare_pretraining_data(src1, src2, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    src1_all_lines = []
    src2_all_lines = []

    src1_paths = sorted(glob.glob(src1 + "/*"))
    if src2:
        src2_paths = sorted(glob.glob(src2 + "/*"))
    else:
        src2_paths = src1_paths

    assert len(src1_paths) == len(src2_paths)

    for src1_file, src2_file in zip(src1_paths, src2_paths):
        assert src1_file.split("/")[-1] == src2_file.split("/")[-1]

        src1_lines = open(src1_file, encoding="utf8").readlines()
        src2_lines = open(src2_file, encoding="utf8").readlines()

        if len(src1_lines) != len(src2_lines):
            logging.warning(
                "WARNING: Unequal lines in: {} {}".format(src1_file, src2_file)
            )
            continue

        for src1_line, src2_line in zip(src1_lines, src2_lines):
            if (not src1_line.strip()) or (not src2_line.strip()):
                logging.info(
                    "WARNING: Skipping blank lines in: {} {}".format(
                        src1_file, src2_file
                    )
                )
                continue
            src1_all_lines.append(src1_line)
            src2_all_lines.append(src2_line)

    open("{}/pretrain_src1.txt".format(output_folder), "w", encoding="utf8").write(
        "".join(src1_all_lines)
    )
    if src2:
        open("{}/pretrain_src2.txt".format(output_folder), "w", encoding="utf8").write(
            "".join(src2_all_lines)
        )


def write_training_data(filenames, output_name, check):
    src1_all_lines = []
    src2_all_lines = []
    tgt_all_lines = []

    for src1_file, src2_file, tgt_file in filenames:
        assert (
            src1_file.split("/")[-1]
            == src2_file.split("/")[-1]
            == tgt_file.split("/")[-1]
        )

        src1_lines = open(src1_file, encoding="utf8").readlines()
        src2_lines = open(src2_file, encoding="utf8").readlines()
        tgt_lines = open(tgt_file, encoding="utf8").readlines()

        if len(src1_lines) != len(src2_lines) or len(src1_lines) != len(tgt_lines):
            logging.warning(
                "WARNING: Unequal lines in: {} {} {}".format(
                    src1_file, src2_file, tgt_file
                )
            )
            continue

        for src1_line, src2_line, tgt_line in zip(src1_lines, src2_lines, tgt_lines):
            if (
                (not src1_line.strip())
                or (not src2_line.strip())
                or (not tgt_line.strip())
            ):
                logging.info(
                    "WARNING: Skipping blank lines in: {} {} {}".format(
                        src1_file, src2_file, tgt_file
                    )
                )
                continue
            src1_all_lines.append(src1_line)
            src2_all_lines.append(src2_line)
            tgt_all_lines.append(tgt_line)

    open("{}src1.txt".format(output_name), "w", encoding="utf8").write(
        "".join(src1_all_lines)
    )
    if check:
        open("{}src2.txt".format(output_name), "w", encoding="utf8").write(
            "".join(src2_all_lines)
        )
    open("{}tgt.txt".format(output_name), "w", encoding="utf8").write(
        "".join(tgt_all_lines)
    )


def prepare_training_data(src1, src2, tgt, output_folder, training_frac):
    assert training_frac < 1.0

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    src1_paths = sorted(glob.glob(src1 + "/*"))
    if src2:
        check = True
        src2_paths = sorted(glob.glob(src2 + "/*"))
    else:
        check = False
        src2_paths = src1_paths
    tgt_paths = sorted(glob.glob(tgt + "/*"))

    assert len(src1_paths) == len(src2_paths) == len(tgt_paths)

    all_files = list(zip(src1_paths, src2_paths, tgt_paths))
    random.shuffle(all_files)
    
    #    Shuffle into 10 buckets
    files = []
    for i in range(10):
        files.append([])
    
    for i, f in enumerate(all_files):
        files[i%10].append(f)
   
    for i in range(10):
#         for j in range(i):
#             id  = i*10 + j
            id = i
            dev_idx = (i+1)%10
            test_idx = i
            train_files = list(set(all_files)-set(files[dev_idx]+files[test_idx]))
            write_training_data(train_files, "{}/train{}_".format(output_folder, id), check)
            write_training_data(
                files[dev_idx], "{}/dev{}_".format(output_folder, id), check
            )
            write_training_data(files[test_idx], "{}/test{}_".format(output_folder, id), check)
            
#     train_idx = round(training_frac * len(all_files))
#     dev_idx = train_idx + round((1.0 - training_frac) * len(all_files) / 2)

#     if dev_idx <= train_idx or dev_idx == len(all_files):
#         logging.error(
#             "ERROR: Fractions for data split are not usable with the dataset size. Adjust the parameter and try again. "
#         )
#         return

#     write_training_data(all_files[:train_idx], "{}/train{}_".format(output_folder, id), check)
#     write_training_data(
#         all_files[train_idx:dev_idx], "{}/dev{}_".format(output_folder, id), check
#     )
#     write_training_data(all_files[dev_idx:], "{}/test{}_".format(output_folder, id), check)




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--unannotated_src1")
    parser.add_argument("--unannotated_src2")
    parser.add_argument("--annotated_src1")
    parser.add_argument("--annotated_src2")
    parser.add_argument("--annotated_tgt")
    parser.add_argument("--output_folder")
    parser.add_argument("--training_frac", type=float, default=0.8)
    args = parser.parse_args()

    prepare_pretraining_data(
        src1=args.unannotated_src1,
        src2=args.unannotated_src2,
        output_folder="{}/pretraining/".format(args.output_folder),
    )
    prepare_training_data(
        src1=args.annotated_src1,
        src2=args.annotated_src2,
        tgt=args.annotated_tgt,
        output_folder="{}/training/".format(args.output_folder),
        training_frac=args.training_frac,
    )
