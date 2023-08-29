from templates import *
from templates_latent import *

if __name__ == '__main__':
    # 256 requires 8x v100s, in our case, on two nodes.
    # do not run this directly, use `sbatch run_ffhq256.sh` to spawn the srun properly.
    gpus = [2,1,0]
    conf = liver_autoenc()
    train(conf, gpus=gpus)