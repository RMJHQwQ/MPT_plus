<img width="698" height="87" alt="image" src="https://github.com/user-attachments/assets/504714f7-0ed1-4e15-9a79-06889aa717d0" />

# MPT+
This is the repository of paper: MPT+: Multi-Perspective Thinking from Large Multimodal Models for Effective Multimodal Sentiment Analysis.

The overview of our proposed model is given as follows:

<img width="1911" height="1037" alt="image" src="https://github.com/user-attachments/assets/de884012-3514-4588-8ac5-9b6e210d6732" />



 Download URL: [Overview of our framework](https://github.com/user-attachments/files/24592117/fig111.pdf)

# Requirements
If you want to re-run the code, the versions of packages needed are recommended as follows:
```python
python                         3.11.8
pytorch                        2.2.1+cu121
einops                         0.8.1
pillow                         10.2.0
transformers                   4.49.0
```


# Dataset Preparation
>[!NOTE]
>All other websites are unauthorized third-party websites. Please carefully use them.

Building upon the MVSA, Memotion, and CH-Mits benchmarks, we have curated four specialized datasets enriched with multi-perspective reasoning from Large Multimodal Models (LMMs). These datasets are specifically designed to enhance multimodal sentiment analysis and are provided in .tsv format within this repository: https://github.com/RMJHQwQ/MPT_plus/tree/main/modified_datasets

For the original source data, please refer to the following resources:

MVSA-Single & MVSA-Multiple: https://mcrlab.net/research/mvsa-sentiment-analysis-on-multi-view-social-data/

Memotion Dataset: "Task Report: Memotion Analysis 1.0 @SemEval 2020: The Visuo-Lingual Metaphor!" https://aclanthology.org/2020.semeval-1.99/

CH-Mits Dataset: https://github.com/Marblrdumdore/CH-Mits

# To Run the Code
Before running the code, you should download the pre-trained parameters of the textual encoder and the visual encoder. Due to the large size of the pretrained models, we give the downloading links so that you can access them. If the paper could be accepted, we will attach the Baidu Disk and Google Drive url here for the convenience of downloading.

For the textual encoder, you can refer to the open-source BERT or BGE-m3 model through https://huggingface.co/google-bert/bert-base-uncased or https://huggingface.co/BAAI/bge-m3

For the visual encoder, you can access the ViT model through https://huggingface.co/google/vit-base-patch16-224-in21k.

After getting the pre-trained models mentioned above, you can put the textual encoder in the folder ```bert/bert-base/``` and the visual encoder in the folder ```models/weight/```.

Then we can seamlessly run the code:

```python
cd MPT_plus
python main.py
```

>[!TIP]
>If you want to change the dataset, you can modify the dataset path in the lines 10 and 135 in the file ```data.py```

# GenAI Disclosure
>[!IMPORTANT]
>There was no use of GenAI tools whatsoever in any stage of the research.

