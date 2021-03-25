# Mode Collapse Example

<table>
  <tr>
    <td> <img src="/plots/hinge.gif"  alt="1" width = 200px height = 150px ></td>
    <td><img src="/plots/hinge.gif" alt="2" width = 200px height = 150px></td>
  </tr> 
  <tr>
    <td> <img src="/plots/hinge.gif"  alt="3" width = 200px height = 150px ></td>
    <td><img src="/plots/hinge.gif" alt="4" width = 200px height = 150px></td>
  </td>
  </tr>
</table>

## Installation

## Run Examples

|Argument | Default value | Info |
| --- | :---: | --- |
|`--device` | 'cuda' | Set device to be utilized (cuda or cpu) |
|`--epochs` | 500 | Training epochs to be performed |
|`--plot_frequency` | 10 | Frequency of epochs to produce plots |
|`--lr` | 0.0001 | Learning rate to be applied |
|`--latent_size` | 32 | Size of latent vector to be utilized |
|`--samples` | 10000 | Number of samples from the real distribution |
|`--batch_size` | 500 | Batch size to be utilized |
|`--loss` | 'standard' | GAN loss function to be used (standard, non-saturating, hinge, wasserstein, wasserstein-gp or least-squares) |
|`--spectral_norm` | False | If set spectral norm is utilized |
|`--topk` | False | If set top-k training is utilized after 0.5 of the epochs to be performed |

## References

````bibtex
@inproceedings{Goodfellow2014,
    title = {Generative Adversarial Nets},
    author = {Goodfellow, Ian and Pouget-Abadie, Jean and Mirza, Mehdi and Xu, Bing and Warde-Farley, David and Ozair, Sherjil and Courville, Aaron and Bengio, Yoshua},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {27},
    year = {2014}
}
````

````bibtex
@inproceedings{Arjovsky2017,
    title={Wasserstein generative adversarial networks},
    author={Arjovsky, Martin and Chintala, Soumith and Bottou, L{\'e}on},
    booktitle={International conference on machine learning},
    pages={214--223},
    year={2017},
    organization={PMLR}
}
````

````bibtex
@inproceedings{Gulrajani2017,
    title={Improved training of wasserstein GANs},
    author={Gulrajani, Ishaan and Ahmed, Faruk and Arjovsky, Martin and Dumoulin, Vincent and Courville, Aaron},
    booktitle={Proceedings of the 31st International Conference on Neural Information Processing Systems},
    pages={5769--5779},
    year={2017}
}
````

````bibtex
@article{Lim2017,
    title={Geometric gan},
    author={Lim, Jae Hyun and Ye, Jong Chul},
    journal={arXiv preprint arXiv:1705.02894},
    year={2017}
}
````

````bibtex
@inproceedings{Mao2017,
    title={Least squares generative adversarial networks},
    author={Mao, Xudong and Li, Qing and Xie, Haoran and Lau, Raymond YK and Wang, Zhen and Paul Smolley, Stephen},
    booktitle={Proceedings of the IEEE international conference on computer vision},
    pages={2794--2802},
    year={2017}
}
````

````bibtex
@inproceedings{Sinha2020,
    title = {Top-k Training of GANs: Improving GAN Performance by Throwing Away Bad Samples},
    author = {Sinha, Samarth and Zhao, Zhengli and ALIAS PARTH GOYAL, Anirudh Goyal and Raffel, Colin A and Odena, Augustus},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {33},
    year = {2020}
}
````
