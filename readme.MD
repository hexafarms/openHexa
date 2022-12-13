![logo](demo/logo.png)

# Hexafarms' computer vision Open source

## Introduction

hexafarms' software-as-a-service (SaaS) lets controlled-environment agriculture save costs upto 40%.
Computer vision tool-kits are being published in this repo.
From plant images, valueable features are extract using this repo.
Our AI model is trained on top all rich features.

## Setup

```bash
conda create --name openHexa python=3.10
conda activate openHexa
git clone https://github.com/hexafarms/openHexa.git openHexa
cd openHexa
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.
```


## Tutorial

Follow detailed tutorial [HERE](demo/docs/tutorial.MD)

## License

This project is released under the [Apache 2.0 license](LICENSE).