# Seq2Seq4ATE

## Codes for the paper "Exploring Sequence-to-Sequence Learning for Aspect Term Extraction".

### (1).Menu:  
> ./code/train.py -> for training  
> ./code/model.py -> for model details  
> ./code/evaluation.py -> for testing  
> ./code/A.jar -> offical script for restaurant domain  
> ./code/eval.jar -> offical scirpt for laptop domain  
> ./best_model/restaurant -> this is the best model trained by our model for restaurant domain  
> ./best_model/laptop -> this is the best model trained by our model for laptop domain  
> ./data -> store necessary data  


### (2).Enviroment
> OS         : Ubuntu 16.04.4 LTS  
> Python : 3.6.8  
> Pytorch: 1.0.0  
> Numpy: 1.16.3  

### (3).Processing data  
> 1. We adopt the data processing method from the paper: 'Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction' (https://arxiv.org/abs/1805.04601).  
> 2. If you want to process your own data, please follow their introduction (https://github.com/howardhsu/DE-CNN).  
3. We have preprocessed the dataset by their method, and all data are stored in dir: /data/pre_data/  


### (4).Training
> CUDA_VISIBLE_DEVICES=0 python code/train.py laptop/restaurant

### (5).Testing
> CUDA_VISIBLE_DEVICES=0 python code/evaluation.py laptop/restaurant


### (6).Acknowledge
> We must thank all authors from this paper: 'Double Embeddings and CNN-based Sequence Labeling for Aspect Extraction'. We adopt many codes from their projects. Thank a lot!

### (7).Apology
> I apologize to all readers that I can not get the original results in the paper for some reason. I fine-tune on two datasets and get new results. It is unbelievable that the new results are higher than the results reported in the paper. 

> Restaurnat: 75.14 -> 76.15  
> Laptop    : 80.31 -> 80.62  

### (8).Others
> If you think the codes & paper are helpful, please cite this paper. Thank you!  

> @inproceedings{ma2019exploring,  
>   title={Exploring Sequence-to-Sequence Learning in Aspect Term Extraction},  
>              author={Ma, Dehong and Li, Sujian and Wu, Fangzhao and Xie, Xing and Wang, Houfeng},  
>              booktitle={Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics},  
>              pages={3538--3547},  
>             year={2019}  
> }



