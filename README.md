
# Invasive Ductal Carcinoma detection

Convolutional Neural Networks and Vision Transformer used for IDC classification. My final degree project for the Universitat Politécnica de Valencia.


## How to train a CNN or the ViT

To train a CNN or the ViT run the following command on a bash terminal

```bash
python main.py -tr <TRAIN_PATH> -te <TEST_PATH> -n <MODEL_NAME> -e <EPOCHS> -b <BATCH_SIZE> -o <OPTIMIZER> --name <FILE_NAME>
```

Where:
   - <TRAIN_PATH>: The absolute path where the training data is stored.
   - <TEST_PATH>: The absolute path where the test data is stored.
   - <MODEL_NAME>: Name of the model you want to train, must be one of the available models.
   - <EPOCHS>: The number of epochs in training.
   - <BATCH_SIZE>: Batch size to load. (A number)
   - <OPTIMIZER>: The learning rate optimizer. Must be one of the available optimizers.
   - <FILE_NAME>: Name of the file where the plots and the weights of the model will be stored. Optional, defaults to "output".

Example:

If your training data is stored at "data/train/", the test data is at "data/test/", and you want to train a EfficientNetB0 for 50 epochs, with AdamW and a 64 batch size, and save your results in a file named "EfficientNetB0" run:
```bash
python main.py -tr "data/train/" -te data/test/ -n efficientnetb0 -e 50 -b 64 -o adamw --name "EfficientNetB0"
```

## How to test a CNN or the ViT
To test a CNN or the ViT run the following command on a bash terminal

```bash
python main.py -te <TEST_PATH> -n <MODEL_NAME> -b <BATCH_SIZE> --test --name <FILE_NAME>
```

Where:
   - <TEST_PATH>: The absolute path where the test data is stored.
   - <BATCH_SIZE>: Batch size to load. (A number)
   - <MODEL_NAME>: Name of the model you want to test, must be one of the available models and checkpoint must be stored at the pretrained folder.
   - <FILE_NAME>: Name of the file that contains the checkpoint of the model.

Example:

If your test data is at "data/test/", and you want to test a EfficientNetB0 and your checkpoint is saved in a file named "EfficientNetB0" at the pretrained folder, run:
```bash
python main.py -te "data/test/" -n efficientnetb0 --test --name "EfficientNetB0"
```