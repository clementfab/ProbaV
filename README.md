# ProbaV Challenge submission

Pytorch implementation of a [Proba V ESA competition](https://kelvins.esa.int/proba-v-super-resolution/home/) submission, reaching a score of 0.990704683785194 over the whole training set. 
Implementation based on [FSRCNN](https://arxiv.org/pdf/1608.00367.pdf). 
This code uses the [`embiggen`](https://github.com/lfsimoes/probav) module, which was specifically made for this competition.

The `PROBAV report.pdf` contains a detailed rundown of this submission.


#### Preping [data](https://kelvins.esa.int/proba-v-super-resolution/data/) and aggregating images

- Download the data from the [competition](https://kelvins.esa.int/proba-v-super-resolution/home/) and extract it in a `data/` folder

- Run the dataprocess.py script to aggregate the low-res images into mean and median aggregates

```  
python dataprocess.py
```  

### Training a new model

- To train a new model with default parameters, use main.py
  Options are contained within the main.py file and can be edited

```  
python main.py --train
```  

### Visualize training

- Training can be monitored easily through Tensorboard, with scores and visualization of super resolved images

```
tensorboard --logdir='runs/'
```  

The images/reconstruction folder allows access to outputs produced by the current model

### Evaluate model

- To evaluate a model over the whole training set, set the load_path of the model to use, and then use :

```  
python main.py --eval
```  

If load_path is left untouched, then this program will evaluate the current submission (checkpoints/trained_model)
