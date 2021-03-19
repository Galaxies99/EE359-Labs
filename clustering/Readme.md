# Lab 1. Clustering

**Author**: Hongjie Fang (方泓杰)

**Student ID**: 518030910150

## Preparation & Run

This code has been tested under macOS Big Sur 11.2.3. Use the following command to install the requirements.

```bash
pip3 install -r requirements.txt
```

Put the data in `data/` folder with name `data.csv`. Then execute the main program using the following scripts.

```bash
sh run.sh
```

The result will be stored in `data/labels.csv`.

## Reference

1. Lloyd, Stuart. *Least squares quantization in PCM*. Initially published as *Bell Telephone Laboratories Paper* (1958), and later published in journal *IEEE transactions on information theory* 28.2 (1982): 129-137. 
2. Arthur, David, and Sergei Vassilvitskii. *k-means++: The advantages of careful seeding*. Stanford, 2006.
3. Ayoosh Kathuria. *Speed Up K-Means Clustering by 70x*. Online, 2020. Available: https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/