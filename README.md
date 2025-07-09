

<!-- <div align="center">

## Vision-R1: Evolving Human-Free Alignment in Large Vision-Language Models via Vision-Guided Reinforcement Learning

</div> -->

<div align="center">

<h1> Learnable Retrieval Enhanced Visual-Text Alignment and Fusion for Radiology Report Generation</h1>

<h5 align="center"> If you find this project useful, please give us a star🌟.

</div>

## Framework

<div align=center>
<img width="650" alt="image" src="figure1.png">
</div>

Automated radiology report generation is essential for improving diagnostic efficiency and reducing the workload of medical professionals. However, existing methods face significant challenges, such as disease class imbalance and insufficient cross-modal fusion. To address these issues, we propose the learnable Retrieval Enhanced Visual-Text Alignment and Fusion (REVTAF) framework, which effectively tackles both class imbalance and visual-text fusion in report generation. REVTAF incorporates two core components: (1) a Learnable Retrieval Enhancer (LRE) that utilizes semantic hierarchies from hyperbolic space and intra-batch context through a ranking-based metric. LRE adaptively retrieves the most relevant reference reports, enhancing image representations, particularly for underrepresented (tail) class inputs; and (2) a fine-grained visual-text alignment and fusion strateg  that ensures consistency across multi-source cross-attention maps for precise alignment. This component further employs an optimal transport-based cross-attention mechanism to dynamically integrate task-relevant textual knowledge for improved report generation. By combining adaptive retrieval with multi-source alignment and fusion, REVTAF achieves fine-grained visual-text integration under weak image-report level supervision while effectively mitigating data imbalance issues. The experiments demonstrate that REVTAF outperforms state-of-the-art methods, achieving an average improvement of 7.4% on the MIMIC-CXR dataset and 2.9% on the IU X-Ray dataset. Comparisons with mainstream multimodal LLMs (e.g., GPT-series models), further highlight its superiority in radiology report generation.

## Setup
```bash
# Clone the repo
git clone git@github.com:banbooliang/REVTAF-RRG
# Create Env and install basic packages
conda create -n reportenv python=3.10
pip install -r requirements.txt
```

## Training
- Download the **MIMIC-CXR** dataset from the [physionet](https://www.physionet.org/content/mimic-cxr-jpg/2.0.0/), and the annotation file can be downloaded from the [Google Drive](https://drive.google.com/file/d/1qR7EJkiBdHPrskfikz2adL-p9BjMRXup/view?usp=sharing). Put them into ./data/mimic_cxr/ forder.

- Download the **IU X-Ray** model from the [R2Gen](https://github.com/zhjohnchan/R2Gen), and the annotation file can be downloaded from the [Google Drive](https://drive.google.com/file/d/1zV5wgi5QsIp6OuC1U95xvOmeAAlBGkRS/view?usp=sharing). Put them into ./data/iu_xray/ forder.

```bash 
bash train_mimic_cxr.sh 
```
The model will be saved into the results/mimic_cxr forder after you run the above code.

## Test

```bash
bash test_mimic_cxr.sh 
bash test_iu_xray.sh 
```

## Citation
If you find our repository useful, please star this repo and cite our paper.
```bibtex
@misc{liang2025,
      title={Learnable Retrieval Enhanced Visual-Text Alignment and Fusion for Radiology Report Generation}, 
      author={Qin Zhou and Guoyan Liang and Xindi Li and Jingyuan Chen and Zhe Wang and Chang Yao and Sai Wu},
      year={2025},
      archivePrefix={ICCV}, 
}
```

## Acknowledgment
* [R2Gen](https://github.com/zhjohnchan/R2Gen)
* [PromptMRG](https://github.com/jhb86253817/PromptMRG)
* [OTSeg](https://github.com/cubeyoung/OTSeg)
* [GW-Regularization](https://github.com/yyf1217/GW-Regularization)

