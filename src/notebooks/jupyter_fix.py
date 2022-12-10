# undefined behaviour of jupyter server cannot start from project root
# temporary fix by manual change of dir when detecting starting wrong one
# with fixed jupyter notebook all path files can be also fixed to use relative one
# @TODO investigate further

import os


def fix_jupyter_path():
    if os.getcwd()[-14:] == '/src/notebooks':
        os.chdir(os.getcwd()[:-14])
