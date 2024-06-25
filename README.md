# Blur Image via Prompts

## Introduction

We implement a module that blurs objects (in an image) determined by the user (text) prompts. While constructing the module, we utilized the pretrained models [OWLViT-v2](https://arxiv.org/abs/2306.09683) provided by [HuggingFace](https://huggingface.co/docs/transformers/en/tasks/zero_shot_object_detection) and [mobile-SAM](https://arxiv.org/pdf/2306.14289.pdf) provided by [ultralytics](https://docs.ultralytics.com/models/mobile-sam/).

## Setting Up the Environment

1. Install [Conda](https://conda.io/projects/conda/en/latest/user-guide/install/index.html), if not already installed.
2. Clone the repository
    ~~~
    git clone https://github.com/byrkbrk/blurring-image-via-prompts.git
    ~~~
3. Change the directory:
    ~~~
    cd blurring-image-via-prompts
    ~~~
4. For macos, run:
    ~~~
    conda env create -f blurring-via-prompts_macos.yaml
    ~~~
    For linux or windows, run:
    ~~~
    conda env create -f blurring-via-prompts_linux.yaml
    ~~~
5. Activate the environment:
    ~~~
    conda activate blurring-via-prompts
    ~~~