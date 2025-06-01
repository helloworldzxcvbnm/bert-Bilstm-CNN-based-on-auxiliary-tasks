# 基于辅助任务的bert-Bilstm-CNN新闻文本分类模型
## 依赖
- python3.13.2
- pytorch 2.6.0+cu126 
- CUDA 12.6
- numpy 2.2.6
- pandas 2.2.3
- scikit-learn 1.6.1
- NVIDIA RTX 5880 Ada Generation 48GB  四张
- NVIDIA RTX 3060 6GB 一张（用于small版本）
## 项目结构
├─cnews（数据集）</br>
    &emsp;&emsp;├─cnews.test.txt</br>
    &emsp;&emsp;├─cnews.train.txt</br>
    &emsp;&emsp;└─cnews.val.txt</br>
├─README.md</br>
├─run.py   （全标签版本）</br>
├─run_small.py    （只含前三个标签，笔记本可跑版本）</br>
└─run_unbalanced.py   （全标签，训练集每类标签数量不平衡版本）</br>

### 这里不提供数据集文件
