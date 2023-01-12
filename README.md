# SRG
stochastic reweighted gradient


## Compile cpp 
- ``cd cpp``
- ``cmake .``
- ``make``

## argument for train.py
- ``--optimizer``: choose the optimizer. (options: GD, SGD, SVRG, SRG)
- ``--dataset``: choose the dataset. (options: Mushrooms, Phishing, W8A, IJCNN1, SYN)
- ``--n_iter``: set the number of iterations.
- ``--lr``: set the learning rate.
- ``--batch_size``: set the batch size.
- ``--store_status_interval``: set the interval of storing data.
- ``--save``: choose whether to store training status.
- ``--save_iter_weight``: chooose whether to save model weight.
- ``--lr_decay``: choose whether to use learning rate decay.
