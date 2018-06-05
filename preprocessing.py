import glob
import os
from shutil import copyfile

base_dir = "/home/murilo/dataset/BreaKHis_v1/**/*.png"
out_dir = "/home/murilo/dataset/BreaKHis_v1/img400/"

def main():
    dirList = glob.glob(base_dir, recursive=True)
    for f in dirList:
        if "400X" in f:
            file_name = os.path.basename(f)
            print(f)
            copyfile(f, out_dir+file_name)

if __name__ == '__main__':
    main()
