# Open Remove Background Model (ormbg)

[>>> DEMO <<<](https://huggingface.co/spaces/schirrmacher/ormbg)

Join our [Research Discord Group](https://discord.gg/j94rUgSx)!

This model is a **fully open-source background remover** optimized for images with humans. It is based on [Highly Accurate Dichotomous Image Segmentation research](https://github.com/xuebinqin/DIS).

## Training

Install dependencies:

```
conda env create -f environment.yaml

conda activate ormbg

python3 ormbg/train_teacher.py
```
