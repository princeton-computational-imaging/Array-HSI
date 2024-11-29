# Learned Multi-aperture Color-coded Optics for Snapshot Hyperspectral Imaging
### [Project Page](https://light.princeton.edu/publication/array-hsi/) | [Paper](https://dl.acm.org/doi/10.1145/3687976)

[Zheng Shi*](https://zheng-shi.github.io/), [Xiong Dun*](), [Haoyu Wei](https://whywww.github.io/), [Shiyu Dong](), [Zhanshan Wang](),[Xinbin Cheng](), [Felix Heide](https://www.cs.princeton.edu/~fheide/),[Yifan (Evan) Peng](https://www.eee.hku.hk/~evanpeng/)

If you find our work useful in your research, please cite:
```
@article{ArrayHSI2024,
author = {Shi, Zheng and Dun, Xiong and Wei, Haoyu and Dong, Shiyu and Wang, Zhanshan and Cheng, Xinbin and Heide, Felix and Peng, Yifan (Evan)},
title = {Learned Multi-aperture Color-coded Optics for Snapshot Hyperspectral Imaging},
year = {2024},
issue_date = {December 2024},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
volume = {43},
number = {6},
issn = {0730-0301},
url = {https://doi.org/10.1145/3687976},
doi = {10.1145/3687976},
journal = {ACM Trans. Graph.},
month = {dec},
articleno = {208},
}
```
## Requirements
This code is developed using Pytorch on Linux machine. Full frozen environment can be found in 'environment.yml', note some of these libraries are not necessary to run this code. 

## Training and Evaluation
Please refer to the train.py and eval.py for details on training and evaluating the model. Example usage can also be found in the accompanying bash scripts. Additionally, please refer to 'param/' for optics and sensor specs, and 'utils/edof_reader.py' for data processing. 

## Pre-trained Models and Optimized DOE Designs
Optimzed DOE Designs and pre-trained models are available under 'ckpt/' folder. Please refer to the supplemental documents for fabrication details.

## License
Our code is licensed under BSL-1. By downloading the software, you agree to the terms of this License. 

## Questions
If there is anything unclear, please feel free to reach out to me using the latest email on my [personal website](https://zheng-shi.github.io/).