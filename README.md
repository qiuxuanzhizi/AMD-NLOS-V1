# AMD-NLOS

##### Adaptive Attention based on Mixture Distribution for Zero-shot Non-line-of-sight Imaging. [IEEE](https://ieeexplore.ieee.org/document/10955200)


## Datasets

#### Synthetic Data

##### $\textit{bowling}$: $64\times64\times256$, we utilize the data provided by [CurvNLOS](https://github.com/Duanlab123/CurvNLOS).

##### $bunny$: $64\times64\times300$, we utilize the prepocessed data provided by [ConvNLOS](https://github.com/ByeongjooAhn/conv_nlos).

#### Real Data

##### We utilize the real-world data provided by [FK](https://github.com/computational-imaging/nlos-fk). Or you can download the preprocessed data [Here](https://drive.google.com/file/d/1MW4cnsEbHoAAicW-8j4I4STrpE7-bF2O/view).



## Experiments

##### If you want to test our method on synthetic data, you can run the [demo_bowling.m](https://github.com/qiuxuanzhizi/AMD-NLOS/blob/main/demo_bowling.m) and [demo_bunny.m](https://github.com/qiuxuanzhizi/AMD-NLOS/blob/main/demo_bunny.m);

##### If you want to test our method on real-world data, you can run the [demo_real.m](https://github.com/qiuxuanzhizi/AMD-NLOS/blob/main/demo_real.m).

##### The hyper-parameters of our method are as following:

| Scene        | bowling          | bunny            | Real dataset     |
| ------------ | ---------------- | ---------------- | ---------------- |
| $\lambda$    | 4                | 1                | 1                |
| $\eta$       | 10000            | 10000            | 1000             |
| $\sigma^{1}$ | 0.005            | 0.005            | 0.001            |
| $\sigma^{2}$ | 10               | 10               | 10               |
| $K$          | 40               | 10               | 10               |
| $J$          | 2                | 2                | 2                |
| $c_1$        | $1\times10^{-5}$ | $1\times10^{-5}$ | $1\times10^{-5}$ |
| $c_2$        | $5\times10^{-4}$ | $1\times10^{-3}$ | $5\times10^{-4}$ |

##### where $\lambda$ and $\eta$ are the regularization parameters, $\sigma^1$ and $\sigma^2$ are the inital standard variation of two Gaussian distributions, $K$ is the number of outer iterations and $c_1$ and $c_2$ denote the stopping criteria of the first and the second subproblem, respectively.

##### If you want to utilize the GPUs to accelerate our algorithm, you can use the command "gpuArray" to convert the large-scale array into GPUs.



## Contact

##### For any questions about the code, please contact qhzhang@mail.bnu.edu.cn



## Cite

##### If you find it useful, please cite our paper.

@article{zhang2025adaptive,
  title={Adaptive Attention based on Mixture Distribution for Zero-shot Non-line-of-sight Imaging},
  author={Zhang, Qinghua and Liu, Jun and Duan, Yuping},
  journal={IEEE Signal Processing Letters},
  publisher={IEEE},
  volume={32},
  pages={1690-1694},
  year={2025},
}























