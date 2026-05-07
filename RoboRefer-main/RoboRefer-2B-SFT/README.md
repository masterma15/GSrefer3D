---
license: apache-2.0
library_name: transformers
pipeline_tag: robotics
base_model:
- Efficient-Large-Model/NVILA-Lite-2B
---

# 🌏 RoboRefer


  <a href="https://zhoues.github.io/RoboRefer"><img src="https://img.shields.io/badge/%F0%9F%8F%A0%20Project-Homepage-blue" alt="HomePage"></a>
  <a href="https://arxiv.org/abs/2506.04308"><img src="https://img.shields.io/badge/arXiv%20paper-2506.04308-b31b1b.svg?logo=arxiv" alt="arXiv"></a>
  <a href="https://github.com/Zhoues/RoboRefer"><img src="https://img.shields.io/badge/Code-RoboRefer-black?logo=github" alt="Project Homepage"></a>

  
  <a href="https://huggingface.co/datasets/JingkunAn/RefSpatial"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Dataset-RefSpatial%20Dataset-brightgreen" alt="Dataset"></a>
  <a href="https://huggingface.co/datasets/JingkunAn/RefSpatial-Bench"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Benchmark-RefSpatial%20Bench-green" alt="Benchmark"></a>
  <a href="https://huggingface.co/collections/Zhoues/roborefer-and-refspatial-6857c97848fab02271310b89"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Weights-RoboRefer%20Model-yellow" alt="Weights"></a>


> This is the official checkpoint of our work: **RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics**





## Overview
RoboRefer-2B-SFT is an open-source vision-language model that is instruction-tuned on a mixture of RefSpatial datasets, instruction tuning, and referring datasets. 


## How to use

RoboRefer-2B-SFT has strong spatial understanding capability and achieves SOTA performance across diverse benchmarks. Given an image with instructions, it can not only answer your questions in both qualitative and quantitative ways using its spatial knowledge, but also output precise points for spatial referring to guide robotic control. For more details, please visit our [official repo](https://github.com/Zhoues/RoboRefer).


## Resources for More Information
- Paper: https://arxiv.org/abs/2506.04308
- Code: https://github.com/Zhoues/RoboRefer
- Dataset: https://huggingface.co/datasets/JingkunAn/RefSpatial
- Benchmark: https://huggingface.co/datasets/BAAI/RefSpatial-Bench
- Website: https://zhoues.github.io/RoboRefer/


## Date
This model was trained in June 2025.



## 📝 Citation
If you find our code or models useful in your work, please cite [our paper](https://arxiv.org/pdf/2505.06111):


```
@article{zhou2025roborefer,
    title={RoboRefer: Towards Spatial Referring with Reasoning in Vision-Language Models for Robotics},
    author={Zhou, Enshen and An, Jingkun and Chi, Cheng and Han, Yi and Rong, Shanyu and Zhang, Chi and Wang, Pengwei and Wang, Zhongyuan and Huang, Tiejun and Sheng, Lu and others},
    journal={arXiv preprint arXiv:2506.04308},
    year={2025}
}
```