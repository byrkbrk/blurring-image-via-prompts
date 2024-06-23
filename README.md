# Blur Image via Prompts

## Introduction

We implement a module that blurs objects (in an image) determined by the user (text) prompts. While constructing the module, we utilized the pretrained models [OWLViT-v2](https://arxiv.org/abs/2306.09683) provided by [HuggingFace](https://huggingface.co/docs/transformers/en/tasks/zero_shot_object_detection) and [mobile-SAM](https://arxiv.org/pdf/2306.14289.pdf) provided by [ultralytics](https://docs.ultralytics.com/models/mobile-sam/).