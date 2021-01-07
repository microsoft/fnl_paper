# Factorized Neural Layers

This repo contains code to reproduce experiments in the paper "Initialization and Regularization of Factorized Neural Layers" (citation below). It is split into codebases for different models and settings we evaluate; please see the corresponding directories for information about the relevant papers.

## Getting Started

The codebase has been tested with Python 3.6 and CUDA 10. Executing <tt>sh setup.sh</tt> will install requirements and generate experimental scripts in the subfolders <tt>\*/generated-scripts</tt> that can be run to compute all ResNet experiments (including normalization plots and distillation results), model compression comparisons, tensor comparisons, and Transformer translation experiments. For reproducing specific experiments please see the appropriately named script; we have also provided example commands for these four settings below.

Parts of the code require the TinyImageNet dataset, which can be downloaded from [here](http://cs231n.stanford.edu/tiny-imagenet-200.zip), and the IWSLT'14 German-English translation dataset, which can be constructed by following the instructions in <tt>Transformer-PyTorch/data</tt>. The scripts also require the home directory of this repo to be in the <tt>PYTHONPATH</tt>.

## ResNet Experiments <tt>[pytorch_resnet_cifar10]</tt>

To train a factorized ResNet20 on CIFAR-10 with spectral initialization and Frobenius decay run
```
python trainer.py --arch resnet20 --rank-scale 0.111 --save-dir results/resnet20-factorized --spectral --wd2fd
```

To run the normalization experiment run
```
python trainer.py --arch resnet20 --rank-scale 0.111 --seed 0 --save-dir results/resnet20-frob --dump-frobnorms --wd2fd
python trainer.py --arch resnet20 --rank-scale 0.111 --seed 0 --save-dir results/resnet20-norm --no-frob --normalize frob/frobnorms.tensor
```

To perform a deep distillation with ResNet32 on CIFAR-100 run
```
python trainer.py --data cifar100 --arch resnet32 --rank-scale 1.0 --square --save-dir results/resnet32-deep --wd2fd
```

## Model Compression Comparisons <tt>[EigenDamage-Pytorch]</tt>

To train a factorized ResNet32 on CIFAR-10 with target compression rate 0.1 using spectral initialization and Frobenius decay run
```
python main_pretrain.py --network resnet --weight_decay 1E-4 --depth 32 --target-ratio 0.1 --log_dir results/resnet32 --spectral --wd2fd
```

To train a factorized VGG19 on TinyImageNet with target compression rate 0.02 using spectral initialization and Frobenius decay run
```
python main_pretrain.py --dataset tiny_imagenet --network vgg --weight_decay 2E-4 --depth 19 --target-ratio 0.02 --log_dir results/vgg19 --spectral --wd2fd
```

## Tensor Comparisons <tt>[deficient-efficient]</tt>

To train a factorized WideResNet28-10 on CIFAR-10 with target compression rate 0.06667 using spectral initialization and Frobenius decay run
```
python main.py cifar10 teacher --wrn_depth 28 --wrn_width 10 --epochs 200 --conv Conv -t results/conv --target-ratio 0.06667 --spectral --wd2fd
```

To train a Tensor-Train-factorized VGG19 WideResNet28-10 on CIFAR-10 with target compression rate 0.01667 using spectral initialization and Frobenius decay run
```
python main.py cifar10 teacher --wrn_depth 28 --wrn_width 10 --epochs 200 --conv TensorTrain_0.234 -t results/tt --spectral --wd2fd
```

## Transformer Translation Experiments <tt>[Transformer-PyTorch]</tt>

To train and evaluate the resulting BLEU scroe of a factorized small Transformer with spectral initialization and Frobenius decay on all linear layers, the Query-Key quadratic form in MHA, and the Output-Value quadratic form in MHA, run
```
python train.py data-bin/iwslt14.tokenized.de-en --arch transformer_small --clip-norm 0.1 --dropout 0.2 --max-tokens 4000 --criterion label_smoothed_cross_entropy --label-smoothing 0.1 --lr-scheduler inverse_sqrt --lr 0.25 --optimizer nag --warmup-init-lr 0.25 --warmup-updates 4000 --max-update 100000 --no-epoch-checkpoints --save-dir results
--rank-scale 0.5 --spectral --spectral-quekey --spectral-outval --wd2fd --wd2fd-quekey --wd2fd-outval --distributed-world-size 1
python generate.py data-bin/iwslt14.tokenized.de-en --batch-size 128 --beam 5 --remove-bpe --quiet --path results/checkpoint_best.pt --dump results/bleu.log --rank-scale 0.5
```

## Citation
  
    @misc{khodak2021factorized,
      title={Initalization and Regularization of Factorized Neural Layers},
      author={Mikhail Khodak and Neil A. Tenenholtz and Lester Mackey and Nicol\`o Fusi},
      year={2021}
    }

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies.
