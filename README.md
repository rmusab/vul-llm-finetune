# Finetuning Large Language Models for Vulnerability Detection

[![arXiv](https://img.shields.io/badge/arXiv-2401.17010-B31B1B)](https://arxiv.org/abs/2401.17010)

This repository contains the necessary code for reproducing the experiments conducted in the paper "Finetuning Large Language Models for Vulnerability Detection":
[https://arxiv.org/abs/2401.17010](https://arxiv.org/abs/2401.17010)

## Folder structure

You can find the following folders in the root directory:
- Folder `ContraBERT` contains the code of the ContraBERT model, which was used in the paper;
- Folder `Datasets` contains two versions of the dataset used in the paper: `X_1` with `P_3` and `X_1` without `P_3`. Part `P_3` stands for a set of random unchanged functions from vulnerability fixing commits;
- Folder `LineVul` contains the code of the LineVul model, which can be used to run with the ContraBERT model;
- Folder `LLM` contains the code for finetuning LLMs using the next token prediction or classification losses with the function batch packing technique.

## Citing our work

Please, make sure to use the following BibTeX entry when citing our work:

```bibtex
@ARTICLE{10908394,
  author={Shestov, Aleksei and Levichev, Rodion and Mussabayev, Ravil and Maslov, Evgeny and Zadorozhny, Pavel and Cheshkov, Anton and Mussabayev, Rustam and Toleu, Alymzhan and Tolegen, Gulmira and Krassovitskiy, Alexander},
  journal={IEEE Access}, 
  title={Finetuning Large Language Models for Vulnerability Detection}, 
  year={2025},
  volume={13},
  number={},
  pages={38889-38900},
  keywords={Training;Codes;Standards;Large language models;Predictive models;Limiting;Java;Entropy;Adaptation models;Reviews;Large language models;vulnerability detection;cybersecurity;finetuning;StarCoder;WizardCoder;PEFT;LoRA},
  doi={10.1109/ACCESS.2025.3546700}
}
```

