import re
import argparse
import os.path
from argparse import RawTextHelpFormatter

if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=RawTextHelpFormatter)
    parser.add_argument('--model_file', type=str, default="dnn_model", help="<name of your model file with layer specs>")
    parser.add_argument('--dataflow', type=str, default="ykp_os", help='dataflow choices: ykp_os, kcp_ws, xp_ws, rs')
    parser.add_argument('--outfile', type=str, default="out.m", help='output file name')
    opt = parser.parse_args()
    print('Begin processing')
    dsconv = 0
    base_path = '../../data/'
    if os.path.exists(base_path + 'model/' + opt.model_file):
        with open('./dataflow/' + opt.dataflow + ".m" ,"r") as fd:
            with open('./dataflow/'+ 'dpt.m' , "r") as fdpt:
                with open(base_path + 'mapping/' + opt.outfile, "w") as fo:
                    with open(base_path + 'model/' + opt.model_file, "r") as fm:
                        for line in fm:
                            if(re.search("DSCONV",line)):
                                dsconv = 1
                            if(re.search("Dimensions",line)):
                                fo.write(line)
                                if(dsconv):
                                    fdpt.seek(0)
                                    fo.write(fdpt.read())
                                else:
                                    fd.seek(0)
                                    fo.write(fd.read())
                                dsconv=0
                            else:
                                fo.write(line)

        print("Mapping file created")
    else:
        print("Model file not found, please provide one")
