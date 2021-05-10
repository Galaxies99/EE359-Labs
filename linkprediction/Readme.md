# Lab 3. Link Prediction

**Author**: Hongjie Fang (方泓杰)

**Student ID**: 518030910150

**Email**: galaxies@sjtu.edu.cn

## Preparation & Run

This code has been tested under macOS Big Sur 11.2.3 using Python 3.8.8. Use the following command to install the requirements.

```bash
pip3 install -r requirements.txt
```

Put the data in `data/` folder with name `course3_edge.csv` and `course3_test.csv`. Then execute the main program using the following scripts.

```bash
python3 src/main.py
```

## Reference

1. Li, Aaron Q., et al. "Reducing the sampling complexity of topic models.", Proceedings of the 20th ACM SIGKDD international conference on Knowledge discovery and data mining. 2014.
2. Grover, Aditya, and Jure Leskovec. "node2vec: Scalable feature learning for networks." Proceedings of the 22nd ACM SIGKDD international conference on Knowledge discovery and data mining. 2016.
