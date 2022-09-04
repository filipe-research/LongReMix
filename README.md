# LongReMix: Robust Learning with High Confidence Samples in a Noisy Label Environment
This repository is the official implementation of [LongReMix: Robust Learning with High Confidence Samples in a Noisy Label Environment](https://www.sciencedirect.com/science/article/abs/pii/S0031320322004939) (Pattern Recognition 2022).

<b>Authors</b>: Filipe R. Cordeiro; Vasileios Belagiannis, Ian Reid and Gustavo Carneiro


<!-- <b>Illustration</b>\
<img src="img/propmix_scheme.png"> -->

## Requirements
- This codebase is written for `python3`.
- To install necessary python packages, run `pip install -r requirements.txt`.

## Training and Evaluating
The pipeline for training with LongReMix, for CIFAR dataset, is the following:

1. Run the first stage the find the high confidence samples set. For example, to run an experiment for cifar-10, with 50% symmetric noise, run:

`python3 Train_cifar_longremix_stage1.py --lambda_u 25 --r 0.5 --noise_mode sym`

2. Run the second stage:

`python3 Train_cifar_longremix_stage2.py --lambda_u 25 --r 0.5 --noise_mode sym`

The parameteres should be adapted according to the dataset and noise rate, as reported in the paper.

For running with red noise, you can use the files `Train_cifar_longremix_stage1_red.py` and `Train_cifar_longremix_stage2_red.py`.




## License and Contributing
- This README is formatted based on [paperswithcode](https://github.com/paperswithcode/releasing-research-code).
- This project is licensed under the terms of the MIT license.
- Feel free to post issues via Github. 

<b>Cite LongReMix</b>\
If you find the code useful in your research, please consider citing our paper:

```
@article{cordeiroLongReMix22,
  title={LongReMix: Robust Learning with High Confidence Samples in a Noisy Label Environment},
  author={Cordeiro, F. R. and Belagiannis, Vasileios and Reid, Ian and Carneiro, Gustavo},
  journal={Pattern Recognition},
  volume={?},
  year={2022}
}
```
## Contact
Please contact filipe.rolim@ufrpe.br if you have any question on the codes.
