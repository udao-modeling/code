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

import os
from sparkmodeling.common.lodatastruct import LODataStruct

IN_FOLDER = "output-"
OUT_FOLDER = "output/"


def main():
    if not os.path.exists(OUT_FOLDER):
        os.makedirs(OUT_FOLDER)

    for fname in os.listdir(IN_FOLDER):
        if ".bin" in fname:
            fpath = os.path.join(IN_FOLDER, fname)
            lods = LODataStruct.load_from_file(
                os.path.join(fpath), autobuild=False)
            lods.serialize(os.path.join(OUT_FOLDER, fname), destroy=True)


if __name__ == "__main__":
    main()
