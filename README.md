# BigVGAN

My implementation of BigVGAN-base([paper](https://arxiv.org/abs/2206.04658)) for JSUT([link](https://sites.google.com/site/shinnosuketakamichi/publication/jsut)) powerd by lightning.


The differences between HiFi-GAN and this are
- Activation is replaced by AntiAliasActivation, which is composed of 2xUpsample, Snake, 2xDownsample, instead of LeakyReLU.
- Remove pre-activation of each ConvTranspose1d w.r.t. paper.

# Usage
Running run.sh will automatically download the data and begin training.  
So just execute the following commands to begin training.

```sh
cd scripts
./run.sh
```

synthesize.sh uses last.ckpt by default, so if you want to use a specific weight, change it.

```sh
cd scripts
./synthesis.sh
```

# Requirements

```sh
pip install torch torchaudio lightning pandas
```

# Result
WIP
