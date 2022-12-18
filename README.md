# JointXplore - Testing and Exploring Joint Visual-NLP Networks

<p align="center"><img width="350" src="./docs/model.png"></p>

This is the official repository of my report [**JointXplore - Testing and Exploring Joint Visual-NLP Networks**](./docs/3D_Visual_Question_Answering.pdf) by Leonard Schenk for the course Testing and Verification in Machine Learning.
## Abstract
There is no software engineering without testing. This
sentence has always been true and is even more important
in the light of probabilistic, potentially safety-critical neural
networks. Few work has been conducted [^robust][^measure] to test
multimodal networks on tasks such as Visual Question Answering
(VQA). To this end, this work presents three different
experiments that measure accuracy, coverage and robustness
on two different multimodal neural network architectures.
Additionally, this work examines the effect of using
only the textual input to perform VQA in each of these settings.
The results reveal that both architectures have a relatively
high performance when using only text. Furthermore,
different coverage metrics show that the text input alone discovers
less internal states compared to the combined visionlanguage
input. Finally, using state of the art adversarial
attack methods point out the vulnerability of multimodal
neural networks.

[^robust]: Kim, Jaekyum, et al. "Robust deep multi-modal learning based on gated information fusion network." Asian Conference on Computer Vision. Springer, Cham, 2018.
[^measure]: Wang, Xuezhi, Haohan Wang, and Diyi Yang. "Measure and Improve Robustness in NLP Models: A Survey." arXiv preprint arXiv:2112.08313 (2021).
## Installation

1. Install requirements with:
```shell
pip install -r requirements.txt
```
2. In root folder, install [LAVIS](https://github.com/salesforce/lavis) as described in in the [official repository](https://github.com/salesforce/lavis#installation)

## Dataset

1. Download VQA 2.0 train and validation set incl. images from the [official webpage](https://visualqa.org/download.html) and save it under  `data/`

2. run
```shell
python load_helper.py
```
to create pre-filtered datasets without greyscale images and with smaller size

## Usage

The code can be run with the following command:
```shell
python run.py --data_path <data_path="./data/">  
--task <["coverage_regions", "coverage", "adversarial_text"]> --model <["vilt", "albef"]> 
--use_rnd
--num_samples <[2500 (coverage), 5000 (coverage_regions)]> --activations_file <path to file that was saved after coverage regions>
```

Examples:

Coverage regions with ViLT and full images:
```shell
python run.py --task "coverage_regions" --model "vilt" --num_samples 5000
```

Coverage metrics with ALBEF and random images:
```shell
python run.py --task "coverage" --model "albef" --num_samples 2500 --use_rnd
```

Adversarial Attack with ViLT and random images:
```shell
python run.py --task "adversarial_text" --model "vilt" --num_attacks 80 --use_rnd
```



## Acknowledgements
I would like to thank [Salesforce/LAVIS](https://github.com/salesforce/LAVIS) for the ALBEF model, [dandelin/ViLT](https://github.com/dandelin/ViLT) ViLT model on [huggingface](https://huggingface.co/dandelin/vilt-b32-finetuned-vqa) and [visualqa](https://visualqa.org/download.html) for the dataset.

