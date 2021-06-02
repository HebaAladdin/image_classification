class argHandler(dict):
    #A super duper fancy custom made CLI argument handler!!
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    _descriptions = {'help, --h, -h': 'show this super helpful message and exit'}

    def setDefaults(self):
        self.define('train_csv', '../classification_datasets/wallball_corridor_training/data.csv', 'path to training directory with folders containig the images of each class')
        self.define('test_csv', '../classification_datasets/wallball_corridor_testing/data.csv', 'path to testing directory with folders containig the images of each class')
        self.define('image_directory', '',
'path to folder containing the patient folders which containg the images')
        self.define('visual_model_name', 'EfficientNetB4', 'select from (VGG16, VGG19, DenseNet121, DenseNet169, DenseNet201, Xception, ResNet50, ResNet101, ResNet152, ResNet50V2, ResNet101V2, ResNet152V2, InceptionV3, InceptionResNetV2, NASNetMobile, NASNetLarge, MobileNet, MobileNetV2, EfficientNetB0 to EfficientNetB7). Note that the classifier layer is removed by default.')
        self.define('image_target_size', (224, 224, 3), 'the target size to resize the image')
        self.define('num_epochs', 100, 'maximum number of epochs')
        self.define('classes', ['LF','RF'], 'the classes that will be classified, should match the folders in train_folder and test_folder')
        self.define('flipped_classes', ['RF','LF'], 'The flipped version of the classes when flipping horizontally.')
        
        self.define('csv_label_columns', ['label'], 'the name of the label columns in the csv')
        self.define('classifier_layer_sizes', [], 'a list describing the hidden layers of the classifier. Example [10,0.4,5] will create a hidden layer with size 10 then dropout wth drop prob 0.4, then hidden layer with size 5. If empty it will connect to output nodes directly.')
        self.define('conv_layers_to_train', -1, 'the number of layers that should be trained in the visual model counting from the end. -1 means train all and 0 means freezing the visual model')
        self.define('use_imagenet_weights', True, 'initialize the visual model with pretrained weights on imagenet')
        self.define('pop_conv_layers', 0, 'number of layers to be popped from the visual model. Note that the imagenet classifier is removed by default so you should not take them into considaration')
        self.define('final_layer_pooling', 'avg', 'the pooling to be used as a final layer to the visual model')
        self.define('load_model_path', '', 'a path containing the checkpoints. If provided with load_model_name the system will continue the training from that point or use it in testing.')
        self.define('save_model_path', './WC_EfficientNetB4/', 'where to save the checkpoints. The path will be created if it does not exist. The system saves every epoch by default')
        self.define('learning_rate', 1e-4, 'The optimizer learning rate')    
        self.define('learning_rate_decay_factor', 0.1, 'Learning rate decay factor when validation loss stops decreasing')
        self.define('gpu_percentage', 0.95, 'gpu utilization. If 0 it will use the cpu')        
        self.define('batch_size', 16, 'batch size for training and testing')        
        self.define('optimizer_type', 'Adam', 'Choose from (Adam, SGD, RMSprop, Adagrad, Adadelta, Adamax, Nadam)')
        self.define('class_weights_balancing', False, 'Should be true when using imbalanced datasets. It automatically calculates class loss weights and use them')
        self.define('generator_workers', 4, 'The number of cpu workers generating batches.')
        self.define('generator_queue_length', 12, 'The maximum number of batches in the queue to be trained on.')
        self.define('minimum_learning_rate', 1e-7, 'The minimum possible learning rate when decaying')
        self.define('reduce_lr_patience', 3,
                    'The number of epochs to reduce the learning rate when validation loss is not decreasing')
        self.define('show_model_summary', True, 'A flag to show or hide the model summary')
        
    def define(self, argName, default, description):
        self[argName] = default
        self._descriptions[argName] = description

    def help(self):
        print('Arguments:')
        spacing = max([len(i) for i in self._descriptions.keys()]) + 2
        for item in self._descriptions:
            currentSpacing = spacing - len(item)
            print('  --' + item + (' ' * currentSpacing) + self._descriptions[item])
        print('')
        exit()


