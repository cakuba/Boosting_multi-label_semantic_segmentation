[license]: https://github.com/cakuba/Boosting_multi-label_semantic_segmentation/blob/master/LICENSE
[weights]: https://1drv.ms/u/s!Av8_YAWBQpg7eZMDQ0OMwGG3qTk

# Boosting Multi-label Semantic Segmentation

A boosting framework, consisting of a DCNN for multi-label semantic segmentation with a customized logarithmic-Dice loss function, a fusion module combining the original labels and the corresponding predictions from the DCNN, and a boosting algorithm to sequentially update the sample weights during network training iterations, is proposed to systematically improve the quality of the annotated data, resulting in eventually the performance improvement of the segmentation tasks.

The framework was verified in MOST for the segmentation task of soma and vessel structures in mouse brain.

## Quickstart

0. fully tested in a docker image running on Ubunti 16.04 and Docker version 18.06.1-ce

1. developed in Python 3.6 with libraries as
```Bash
   Tensorflow 2.2.0
   Keras 2.2.4
```

2. obtain the MOST sample data and save in the local directory
```Bash
   wget 'https://1drv.ms/u/s!Av8_YAWBQpg7enAkjXEdJ8mw37Q'
   unzip most_sample_dataset.zip
   mv Train/ ./data/
```
3. pre-process the training data with augmentation
```Bash
   python training_data_augment.py 
``` 

4. run the boosting framework in the background
```Bash
   python boosting_training.py >  boosting_training.log 2>&1 &
``` 
NOTE: quite a few hyper-parameters of the boosting framework can be changed as need. See the file "boosting_training.py" for more details.

5. predict the test data with boosted network
```Bash
   python boosting_pred.py
``` 

## Who are we?

The Boosting framework for multi-label semantic segmentation is proposed and maintained by researchers from <a href="https://www.wit.edu.cn/" target="_blank">WIT</a> and <a href="http://www.wnlo.cn/"  target="_blank">HUST</a>.

## License

See [LICENSE][license]
