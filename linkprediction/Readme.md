# Lab 3. Link Prediction

**Author**: Hongjie Fang (方泓杰)

**Student ID**: 518030910150

**Email**: galaxies@sjtu.edu.cn

## Preparation

This code has been tested under macOS Big Sur 11.2.3 using Python 3.8.8. Use the following command to install the requirements.

```bash
pip3 install -r requirements.txt
```

**NOTE**. `sklearn` package is only used during training, since we use the metrics `AUC` in this process to measure the performance of the model.

Put the data in `data/` folder with name `course3_edge.csv` and `course3_test.csv`.

## Training (optional)

Use the following commands to train the node2vec model. Make sure `checkpoint/` folder is empty when you start a new training process. A training process takes about 20~30 minutes using an NVIDIA RTX2080 GPU.

```bash
python src/train.py --cfg [Config File]
```

where the optional  `--cfg [Config File]` specifies the configuration file, the default configuration file is `configs/default.yaml`.

## Testing / Inference

Then execute the main program (inference program) using the following scripts. Make sure `checkpoint/` folder has a checkpoint file named `checkpoint.tar`. A testing process takes a few seconds (usually 3-4 seconds) using an ordinary CPU.

```bash
python3 src/test.py --cfg [Config File]
```

where the optional  `--cfg [Config File]` specifies the configuration file, the default configuration file is `configs/default.yaml`.

The inference program will generate an `submission.csv` which corresponds to the answer to the `course3_edge.csv` test file, whose filename is specified in the configuration file.

## Reference

1. Li, Aaron Q., et al. "Reducing the sampling complexity of topic models.", Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
2. Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016. 
