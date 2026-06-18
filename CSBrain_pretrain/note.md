**Note**

This directory contains two parallel pretraining pipelines:

1. **Original CSBrain reproduction**

   * `pretrain_csbrain_h5.py`
   * `pretrain_trainer_h5.py`
   * `train_csbrain_pretrain_h5.sh`

   These files are used to reproduce the original CSBrain pretraining pipeline.

2. **CSBrain with our TriAxial block**

   * `CSBrain_TriAxial.py`
   * `pretrain_csbrain_triaxial_h5.py`
   * `train_csbrain_triaxial_pretrain_h5.sh`

   These files keep the original CSBrain pretraining framework and training protocol, but replace the core encoder block with our TriAxial architecture.

Other shared files, such as `pretraining_dataset_h5.py`, are used by both pipelines.
