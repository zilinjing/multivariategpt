# multivariategpt

**multivariategpt** is a decoder-only transformer for mixed categorical and numeric data that allows for contiuous representation of numeric quantities without discretization. Categorical data are treated exactly like tokens in a language model. Numeric data use a modfied embedding scheme and loss function to allow for joint class and value prediction.

## Overview

This package is the code for the method described in [multivariateGPT: a decoder-only transformer for multivariate categorical and numeric data](https://arxiv.org/abs/2505.21680) (Loza et al., 2025).

Real-world processes often generate data that are a mix of categorical and numeric values recorded at irregular and informative intervals. **multivariateGPT** provides a single architecture for modeling sequences of mixed categorical (including tokenized text) and numeric data through:

- **Hybrid tokenization** tokens are two parallel streams of class id and value (nan if categorical)
- **Embedding scheme** that handles both categorical and continuous values. categorical tokens receive an identical embedding as in language models, while continuous values are embedded by modifying a class-specific embedding with the numeric value.
- **Loss function** extending next token prediction to likelihood estimation of the joint distribution of next token class and value.

## Demos
The demo notebook `demo/oscillator_demo.ipynb` illustrates how to use the package to model simple oscillator data.
The demo notebook `demo/tokenization.ipynb` illustrates the format of tokens for categorical and continuous data.

## Installation

```bash
pip install multivariategpt
```

## Acknowledgements
This code is based on the nanoGPT codebase by Andrej Karpathy found here: http://github.com/karpathy/nanoGPT

## Citation

If you use this code, please cite:

```bibtex
@article{loza2025multivariategpt,
  title={multivariateGPT: a decoder-only transformer for multivariate categorical and numeric data},
  author={Loza, Andrew J. and Kim, Jun Yup and Song, Shangzheng and Liu, Yihang and Sung, Joseph J. Y. and Taylor, R Andrew and Shung, Dennis L.},
  journal={arXiv preprint arXiv:2505.21680},
  year={2025}
}
```


## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
