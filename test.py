import os
import re
import json
import hydra
#import click
import torch
#from utils.torch_utils import distributed as dist
import utils.setup as setup
import urllib.request


def _main(args):

    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #assert torch.cuda.is_available()
    #device="cuda"

    global __file__
    __file__ = hydra.utils.to_absolute_path(__file__)
    dirname = os.path.dirname(__file__)
    args.model_dir = os.path.join(dirname, str(args.model_dir))
    if not os.path.exists(args.model_dir):
            os.makedirs(args.model_dir)

    args.exp.model_dir=args.model_dir

    torch.multiprocessing.set_start_method('spawn')

    diff_params=setup.setup_diff_parameters(args)
    network=setup.setup_network(args, device)

    test_set=setup.setup_dataset_test(args)

    tester=setup.setup_tester(args, network=network, diff_params=diff_params, test_set=test_set, device=device) #this will be used for making demos during training
    # Print options.
    print()
    print('Training options:')
    print()
    print(f'Output directory:        {args.model_dir}')
    print(f'Network architecture:    {args.network.callable}')
    print(f'Diffusion parameterization:  {args.diff_params.callable}')
    print(f'Tester:                  {args.tester.callable}')
    print(f'Experiment:                  {args.exp.exp_name}')
    print()

    # Train.
    print("loading checkpoint path:", args.tester.checkpoint)
    if args.tester.checkpoint != 'None':
        ckpt_path=os.path.join(dirname, args.tester.checkpoint)
        if not(os.path.exists(ckpt_path)):
            print("download ckpt")
            path, basename= os.path.split(args.tester.checkpoint)
            if not(os.path.exists(os.path.join(dirname, path))):
                    os.makedirs(os.path.join(dirname,path))
            HF_path="https://huggingface.co/Eloimoliner/babe/resolve/main/"+os.path.basename(args.tester.checkpoint)
            urllib.request.urlretrieve(HF_path, filename=ckpt_path)
        #relative path
        ckpt_path=os.path.join(dirname, args.tester.checkpoint)
        tester.load_checkpoint(ckpt_path) 
        #jexcept:
        #j    #absolute path
        #j   tester.load_checkpoint(os.path.join(args.model_dir,args.tester.checkpoint)) 
    else:
        print("trying to load latest checkpoint")
        tester.load_latest_checkpoint()

    tester.dodajob()

@hydra.main(config_path="conf", config_name="conf")
def main(args):
    _main(args)

if __name__ == "__main__":
    main()

#----------------------------------------------------------------------------
