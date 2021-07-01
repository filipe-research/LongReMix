First run Train_cifar_longremix_stage1.py. When it finish, run Train_cifar_longremix_stage2.py.

For example, to run an experiment for cifar-10, with 50% symmetric noise, run:
python3 Train_cifar_longremix_stage1.py --lambda_u 25 --r 0.5 --noise_mode sym

When it finish, run:
python3 Train_cifar_longremix_stage2.py --lambda_u 25 --r 0.5 --noise_mode sym


