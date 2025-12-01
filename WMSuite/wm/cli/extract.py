import argparse
from  wm.core.helpers import *
from wm.core.decode import *


def Options():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument('--data_path', type=str, help='dir used to embed watermarks')
    parser.add_argument('--method', type=str, default='rivaGan', choices=['dwtdct','stegastamp', 'hidden', 'rivaGan','stable_signature','vine'], help='watermark algorithms')
    parser.add_argument('--device', type=str, default='cuda:0', help='device to use')
    parser.add_argument('--beta', type=float, default=0.000001, help='beta for cal_tolerant')
    parser.add_argument('--log_dir',type=str, help='dir to save the decoding results file')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    set_seeds(2024)
    opt = Options()
    _, decoder = load_models(opt.method, opt.device)
    bitwise_accuracy_avg,FPR = decode_from_folder(opt,decoder)
    print(round(bitwise_accuracy_avg, 4),round(FPR, 4))
