# Benchmarks

A set of micro-benchmarks that measure elapsed runtime and allocated CUDA memory for both [basic modules](https://github.com/pytorch/opacus/tree/main/opacus/grad_sample) and [more complex layers](https://github.com/pytorch/opacus/tree/main/opacus/layers), as well as their respective [torch.nn](https://pytorch.org/docs/stable/nn.html) counterparts.

Requires `torch >= 1.10`

For `--grad_sample_modes functorch` requires `torch==1.12`

For `--grad_sample_modes ew` requires `torch>=1.13`

## Contents

- [run_benchmarks.py](run_benchmarks.py) runs a list of benchmarks based on the given config file and writes out the results as pickle files to `results/raw` (by default)

- [benchmark_layer.py](benchmark_layer.py) benchmarks a single layer at the given batch size for both runtime and memory

- [config.json](config.json) contains an example JSON config for running benchmarks

- [layers.py](layers.py) implements each layer with its corresponding inputs/outputs, moving each component to device, and collecting respective memory statistics

- [utils.py](utils.py) implements helper functions, e.g. for saving results as pickle files


## Benchmarks

For each layer and batch size in [config.json](config.json), [run_benchmarks.py](run_benchmarks.py) will do the following:
```
Do this num_runs times:
    Init layer, one batch of random input, one batch of random "labels"
    Move each component to GPU and collect its allocated CUDA memory (if applicable)

    Start timer

    Do this num_repeats times:
        preds = layer(input)
        loss = criterion(preds, labels)
        loss.backward()

    Stop timer

    Return elapsed time / num_repeats and memory statistics
```

## Layers

All layers follow the corresponding `torch.nn` module's interface with the same default values (if not specified in [config.json](config.json)).

A note on `input_shape` in [config.json](config.json): parameters that are shared between the model and the input are listed separately. Therefore, the actual input shape will vary:

- Linear: `(batch_size, *input_shape, in_features)`

- Convolutional: `(batch_size, in_channels, *input_shape)`

- LayerNorm:
    - Input: `(batch_size, *input_shape)`
    - Normalized shape: `(input_shape[-D:])`

- InstanceNorm: `(batch_size, num_features, *input_shape)`

- GroupNorm: `(batch_size, num_channels, *input_shape)`

- Embedding: `(batch_size, *input_shape)`

- MultiheadAttention:
    - If not batch_first: `(targ_seq_len, batch_size, embed_dim)`
    - Else: `(batch_size, targ_seq_len, embed_dim)`

- RNN, GRU, LSTM:
    - If not batch_first: `(seq_len, batch_size, input_size)`
    - Else: `(batch_size, seq_len, input_size)`


## Usage

`run_benchmarks.py` without additional arguments will replicate the results in the [technical report introducing Opacus](https://arxiv.org/abs/2109.12298), which you can cite as follows:
```
@article{opacus,
  title={Opacus: {U}ser-Friendly Differential Privacy Library in {PyTorch}},
  author={Ashkan Yousefpour and Igor Shilov and Alexandre Sablayrolles and Davide Testuggine and Karthik Prasad and Mani Malek and John Nguyen and Sayan Ghosh and Akash Bharadwaj and Jessica Zhao and Graham Cormode and Ilya Mironov},
  journal={arXiv preprint arXiv:2109.12298},
  year={2021}
}
```

If saving results, ensure that the `results/raw` (or otherwise specified) root directory exists.

```
usage: run_benchmarks.py [-h]
                         [--layers {linear,conv,layernorm,instancenorm,groupnorm,embedding,mha,dpmha,rnn,dprnn,gru,dpgru,lstm,dplstm} [{linear,conv,layernorm,instancenorm,groupnorm,embedding,mha,dpmha,rnn,dprnn,gru,dpgru,lstm,dplstm} ...]]
                         [--batch_sizes BATCH_SIZES [BATCH_SIZES ...]]
                         [--num_runs NUM_RUNS] [--num_repeats NUM_REPEATS]
                         [--forward_only] [--random_seed RANDOM_SEED]
                         [-c CONFIG_FILE] [--cont] [--root ROOT]
                         [--suffix SUFFIX]
                         [--grad_sample_modes {baseline,hooks,ew,functorch} [{baseline,hooks,ew,functorch} ...]]
                         [--no_save] [-v]

optional arguments:
  -h, --help            show this help message and exit
  --layers {linear,conv,layernorm,instancenorm,groupnorm,embedding,mha,dpmha,rnn,dprnn,gru,dpgru,lstm,dplstm} [{linear,conv,layernorm,instancenorm,groupnorm,embedding,mha,dpmha,rnn,dprnn,gru,dpgru,lstm,dplstm} ...]
  --batch_sizes BATCH_SIZES [BATCH_SIZES ...]
  --num_runs NUM_RUNS   number of benchmarking runs for each layer and batch
                        size
  --num_repeats NUM_REPEATS
                        number of forward/backward passes per run
  --forward_only        only run forward passes
  --random_seed RANDOM_SEED
                        random seed for the first run for each layer and batch
                        size, subsequent runs increase the random seed by 1
  -c CONFIG_FILE, --config_file CONFIG_FILE
                        path to config file with settings for each layer
  --cont                only run missing experiments
  --root ROOT           path to directory where benchmark results should be
                        saved
  --suffix SUFFIX       suffix to append to each result file's name
  --grad_sample_modes {baseline,hooks,ew,functorch} [{baseline,hooks,ew,functorch} ...]
                        Mode to compute per-sample gradients: Classic (hooks),
                        Functorch(functorch), ExpandedWeights(ew), Non-
                        private(baseline)
  --no_save
  -v, --verbose
```

`generate_report.py` will take as an input the path where `run_benchmarks.py` has written the results and it will generate a report.
```
usage: generate_report.py [-h] [--path-to-results PATH_TO_RESULTS]
                          [--save-path SAVE_PATH] [--format {csv,pkl}]

optional arguments:
  -h, --help            show this help message and exit
  --path-to-results PATH_TO_RESULTS
                        the path that `run_benchmarks.py` has saved results
                        to.
  --save-path SAVE_PATH
                        path to save the output.
  --format {csv,pkl}    output format
```
## Tests

```python -m pytest tests/```

Tests for CUDA memory benchmarking will be skipped on CPU.
