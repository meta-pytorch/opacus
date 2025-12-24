<p align="center"><img src="https://github.com/pytorch/opacus/blob/main/website/static/img/opacus_logo.svg" alt="Opacus" width="500"/></p>

<hr/>

[![PyPI Downloads](https://static.pepy.tech/badge/opacus)](https://pepy.tech/projects/opacus)
[![GitHub Actions](https://github.com/pytorch/opacus/actions/workflows/ci_cpu.yml/badge.svg)](https://github.com/pytorch/opacus/actions/workflows/ci_cpu.yml)
[![Coverage Status](https://coveralls.io/repos/github/pytorch/opacus/badge.svg?branch=main)](https://coveralls.io/github/pytorch/opacus?branch=main)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![License](https://img.shields.io/badge/license-apache2-green.svg)](LICENSE)

[Opacus](https://opacus.ai) is a library that enables training PyTorch models
with differential privacy. It supports training with minimal code changes
required on the client, has little impact on training performance, and allows
the client to online track the privacy budget expended at any given moment.


## Target audience

This code release is aimed at two target audiences:

1. ML practitioners will find this to be a gentle introduction to training a
   model with differential privacy as it requires minimal code changes.
2. Differential Privacy researchers will find this easy to experiment and tinker
   with, allowing them to focus on what matters.


## Latest updates

2024-12-18: We updated this [tutorial](https://github.com/pytorch/opacus/blob/main/tutorials/building_text_classifier.ipynb) to show how [LoRA](https://arxiv.org/abs/2106.09685) and [peft](https://huggingface.co/docs/peft/en/index) library could be used in conjuncture with DP-SGD.

2024-08-20: We introduced [Fast Gradient Clipping](https://arxiv.org/abs/2009.03106) and Ghost Clipping(https://arxiv.org/abs/2110.05679) to Opacus, significantly reducing the memory requirements of DP-SGD. Please refer to our [blogpost](https://pytorch.org/blog/clipping-in-opacus/) for more information.

## Installation

The latest release of Opacus can be installed via `pip`:

```bash
pip install opacus
```

OR, alternatively, via `conda`:

```bash
conda install -c conda-forge opacus
```

You can also install directly from the source for the latest features (along
with its quirks and potentially occasional bugs):

```bash
git clone https://github.com/pytorch/opacus.git
cd opacus
pip install -e .
```

## Getting started

To train your model with differential privacy, all you need to do is to
instantiate a `PrivacyEngine` and pass your model, data_loader, and optimizer to
the engine's `make_private()` method to obtain their private counterparts.

```python
# define your components as usual
model = Net()
optimizer = SGD(model.parameters(), lr=0.05)
data_loader = torch.utils.data.DataLoader(dataset, batch_size=1024)

# enter PrivacyEngine
privacy_engine = PrivacyEngine()
model, optimizer, data_loader = privacy_engine.make_private(
    module=model,
    optimizer=optimizer,
    data_loader=data_loader,
    noise_multiplier=1.1,
    max_grad_norm=1.0,
)
# Now it's business as usual
```

The
[MNIST example](https://github.com/pytorch/opacus/tree/main/examples/mnist.py)
shows an end-to-end run using Opacus. The
[examples](https://github.com/pytorch/opacus/tree/main/examples/) folder
contains more such examples.

## Learn more

### Interactive tutorials

We've built a series of IPython-based tutorials as a gentle introduction to
training models with privacy and using various Opacus features.

- [Building text classifier with Differential Privacy on BERT](https://github.com/pytorch/opacus/blob/main/tutorials/building_text_classifier.ipynb)
- [Building an Image Classifier with Differential Privacy](https://github.com/pytorch/opacus/blob/main/tutorials/building_image_classifier.ipynb)
- [Training a differentially private LSTM model for name classification](https://github.com/pytorch/opacus/blob/main/tutorials/building_lstm_name_classifier.ipynb)
- [Opacus Guide: Introduction to advanced features](https://github.com/pytorch/opacus/blob/main/tutorials/intro_to_advanced_features.ipynb)
- [Opacus Guide: Grad samplers](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_grad_sampler.ipynb)
- [Opacus Guide: Module Validator and Fixer](https://github.com/pytorch/opacus/blob/main/tutorials/guide_to_module_validator.ipynb)
- [Opacus Guide: Training with Non-Wrapping Mode](https://github.com/pytorch/opacus/blob/main/tutorials/non_wrapping_mode.ipynb)

## Technical report and citation

The technical report introducing Opacus, presenting its design principles,
mathematical foundations, and benchmarks can be found
[here](https://arxiv.org/abs/2109.12298).

Consider citing the report if you use Opacus in your papers, as follows:

```
@article{opacus,
  title={Opacus: {U}ser-Friendly Differential Privacy Library in {PyTorch}},
  author={Ashkan Yousefpour and Igor Shilov and Alexandre Sablayrolles and Davide Testuggine and Karthik Prasad and Mani Malek and John Nguyen and Sayan Ghosh and Akash Bharadwaj and Jessica Zhao and Graham Cormode and Ilya Mironov},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

### Blogposts and talks

If you want to learn more about DP-SGD and related topics, check out our series
of blogposts and talks:

- [Enabling Fast Gradient Clipping and Ghost Clipping in Opacus](https://pytorch.org/blog/clipping-in-opacus/)
- [Differential Privacy Series Part 1 | DP-SGD Algorithm Explained](https://medium.com/pytorch/differential-privacy-series-part-1-dp-sgd-algorithm-explained-12512c3959a3)
- [Differential Privacy Series Part 2 | Efficient Per-Sample Gradient Computation in Opacus](https://medium.com/pytorch/differential-privacy-series-part-2-efficient-per-sample-gradient-computation-in-opacus-5bf4031d9e22)
- [PriCon 2020 Tutorial: Differentially Private Model Training with Opacus](https://www.youtube.com/watch?v=MWPwofiQMdE&list=PLUNOsx6Az_ZGKQd_p4StdZRFQkCBwnaY6&index=52)
- [Differential Privacy on PyTorch | PyTorch Developer Day 2020](https://www.youtube.com/watch?v=l6fbl2CBnq0)
- [Opacus v1.0 Highlights | PyTorch Developer Day 2021](https://www.youtube.com/watch?v=U1mszp8lzUI)

## FAQ

Check out the [FAQ](https://opacus.ai/docs/faq) page for answers to some of the
most frequently asked questions about differential privacy and Opacus.

## Contributing

See the
[CONTRIBUTING](https://github.com/pytorch/opacus/tree/main/CONTRIBUTING.md) file
for how to help out. Do also check out the README files inside the repo to learn
how the code is organized.

## License

This code is released under Apache 2.0, as found in the
[LICENSE](https://github.com/pytorch/opacus/tree/main/LICENSE) file.
