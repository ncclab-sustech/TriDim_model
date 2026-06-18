## Note

This directory mainly contains the core files used for our pretraining reproduction and modified experiments.

Please note:

1. The uploaded code in this folder is mainly organized around the **pretraining** pipeline.
2. The files most directly related to our modifications and experiments include:

   * `basis_pretrain_wrapper_v10.py`
   * `eeg_mixer_v11_1_spatial_multilevel.py`
   * `pretrain_main.py`
   * `pretrain_trainer.py`
   * `train_pretrain_v11_v5.py`
   * `train_basis_pretrain_h5.sh`
   * `train_cbramod_pretrain_h5.sh`
3. Most of the remaining **official source code** has not been substantially modified. For the complete original project structure and reference implementation, please see:

`/vePFS-0x0d/home/ws2319/Tridim/CBraMod`

In addition, this repository currently contains both the “original reproduction code” and our “modified versions,” so please make sure to use the correct training entry script and corresponding bash script for each experiment.
