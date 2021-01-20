"""
MIT License

Copyright (c) 2020-2021 Ecole Polytechnique.

@Author: Khaled Zaouk <khaled.zaouk@polytechnique.edu>

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np
from sparkmodeling.common.lodatastruct import LODataStruct
from config import *
from helpers_make_lods import make_LODatastruct_mul, make_LODatastruct_X
from tests_consistency import test_eval_splits, test_eval_jobs,\
    test_training_jobs, test_training_splits
from tests_split_configs import check_shared_configs_consistency, \
    assert_no_shared_in_test

import os


def main():

    if not os.path.exists(OUTPUT_FOLDER):
        os.makedirs(OUTPUT_FOLDER)

    config_dict = get_config_dict()
    flat_test_workloads = [TEST_WORKLOADS[t] for t in TEST_WORKLOADS]
    flat_test_workloads = [e for l in flat_test_workloads for e in l]

    lods = make_LODatastruct_mul(
        flat_test_workloads, DATA_FOLDER, CONFIG_PATH,
        with_intensive=WITH_INTENSIVE, si=SEPARATE_INTERSECTIONS_MUL,
        config_dict=config_dict,
        shared_within_templates=SHARED_WITHIN_TEMPLATES)
    sd = lods.get_split_definitions()

    for temp in TEMPLATES:
        print("[making LODS_{}]".format(temp))
        make_LODatastruct_X(
            TEMPLATES[temp],
            TEST_WORKLOADS[temp],
            DATA_FOLDER, CONFIG_PATH, with_intensive=WITH_INTENSIVE,
            split_definitions=sd, X=temp,
            separate_intersections=SEPARATE_INTERSECTIONS_X,
            config_dict=config_dict,
            shared_within_templates=SHARED_WITHIN_TEMPLATES)

    autobuild = DESTROY_ON_SERIALIZE
    print("Loading LODS...")
    lods = {}
    lods_mul = LODataStruct.load_from_file(os.path.join(
        OUTPUT_FOLDER, "lods_mul.bin"),
        autobuild=DESTROY_ON_SERIALIZE)
    lods["mul"] = lods_mul
    print("LODS_mul loaded...")
    for t in TEMPLATES:
        print("Loading LODS_{}".format(t))
        lods[t] = LODataStruct.load_from_file(
            os.path.join(OUTPUT_FOLDER, "lods_{}.bin".format(t)),
            autobuild=DESTROY_ON_SERIALIZE)

    for lod_name in lods:
        lods[lod_name].minmaxscale("X")
        lods[lod_name].minmaxscale("Y")

    print("LODS loaded, starting tests...")
    print("**** CONSISTENCY TESTS (LODS_mul & LODS_X) *****")
    test_eval_jobs(lods, 'test')
    test_eval_splits(lods, 'test')

    test_eval_jobs(lods, 'traincomplement')
    test_eval_splits(lods, 'traincomplement')

    test_training_jobs(lods, "trainval")
    test_training_splits(lods, "trainval")

    print("****           *****                        *****")

    if SEPARATE_INTERSECTIONS_MUL:
        print("***** SHARED/UNSHARED TESTS ON LODS_mul *****")
        check_shared_configs_consistency(lods["mul"])
        assert_no_shared_in_test(lods["mul"])
        assert_no_shared_in_test(lods["mul"], 'shared_traincomplement')


if __name__ == "__main__":
    main()
