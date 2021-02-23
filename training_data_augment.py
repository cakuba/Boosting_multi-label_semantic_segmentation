from keras.preprocessing.image import ImageDataGenerator
def augumented_training_data(orig_data_dir, new_data_dir, instances=2000, size=(1024,1024), random_seed=2020):
    """
    To generate more training data based on augmentation algorithms in Keras. By default,
    a total number of 2000 instances will be generated.
    
    Arguments:
        orig_data_dir: data/Train
        new_data_dir:  data/Train_aug
        instances: number of instances after augmentation
        size: image size after augmentation
    
    [ref] https://keras.io/api/models/model_training_apis/#fit-method
    """
    
    # parameters used during augmentation
    batch_size = 64
    epochs = int(instances/batch_size)
    
    # to create the directory of augmentation data
    os.makedirs(new_data_dir, exist_ok=True)
    assert os.path.isdir(new_data_dir), "directory to save augmentation instances does not exist!"
    
    rmtree(new_data_dir)
    os.makedirs(os.path.join(new_data_dir, 'images/0'))
    os.makedirs(os.path.join(new_data_dir, 'labels/soma'))
    os.makedirs(os.path.join(new_data_dir, 'labels/vessel'))
    
    data_gen_args = dict(rotation_range=30,
                         width_shift_range=0.2,
                         height_shift_range=0.2,
                         shear_range=0.2,
                         zoom_range=0.2,
                         fill_mode='wrap',   # be careful!
                         horizontal_flip=True,
                         vertical_flip=True)
    
    image_datagen = ImageDataGenerator(**data_gen_args)
    soma_label_datagen = ImageDataGenerator(**data_gen_args)
    vessel_label_datagen = ImageDataGenerator(**data_gen_args)
    
    image_generator = image_datagen.flow_from_directory(
                                    os.path.join(orig_data_dir,'images'),
                                    target_size=size,
                                    class_mode=None,
                                    seed=random_seed,
                                    save_to_dir=os.path.join(new_data_dir,'images/0'),
                                    save_prefix="",
                                    save_format="jpg",
                                    batch_size=batch_size)
    soma_label_datagen = soma_label_datagen.flow_from_directory(
                                    os.path.join(orig_data_dir,'labels/soma'),
                                    target_size=size,
                                    class_mode=None,
                                    seed=random_seed,
                                    save_to_dir=os.path.join(new_data_dir,'labels/soma'),
                                    save_prefix="",
                                    save_format="jpg",
                                    batch_size=batch_size)
    vessel_label_datagen = vessel_label_datagen.flow_from_directory(
                                    os.path.join(orig_data_dir,'labels/vessel'),
                                    target_size=size,
                                    class_mode=None,
                                    seed=random_seed,
                                    save_to_dir=os.path.join(new_data_dir,'labels/vessel'),
                                    save_prefix="",
                                    save_format="jpg",
                                    batch_size=batch_size)
    
    for i in range(epochs):
        image_generator.next()
        soma_label_datagen.next()
        vessel_label_datagen.next()
