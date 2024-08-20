import os
import urllib.request as request
from zipfile import ZipFile
import tensorflow as tf
from cnnClassifier.entity.config_entity import PrepareBaseModelConfig
from pathlib import Path

class PrepareBaseModel:
    def __init__(self,config:PrepareBaseModelConfig) -> None:
        self.config=config

    def get_base_model(self):
        self.model = tf.keras.applications.vgg16.VGG16(
            input_shape=self.config.params_image_size,
            weights=self.config.params_weigths,
            include_top=self.config.params_include_top
        )
        self.save_model(path=self.config.base_model_path,model=self.model)
    
    @staticmethod
    def _prepare_full_model(model,classes,freeze_all,freeze_till,lr):
        if freeze_all:
            for layer in model.layers:
                model.trainable = False
        elif (freeze_till is not None) and (freeze_till>0):
            for layer in model.layers[:-freeze_till]:
                model.trainable = False
        
        flatten_in = tf.keras.layers.Flatten()(model.output)
        
        pred = tf.keras.layers.Dense(units=classes,activation="softmax")(flatten_in)

        full_model = tf.keras.models.Model(inputs=model.input,outputs=pred)

        full_model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr),loss=tf.keras.losses.CategoricalCrossentropy(),metrics=["accuracy"])

        full_model.summary()
        
        return full_model
    
    def update_base_model(self):
        self.full_model = self._prepare_full_model(
            model=self.model,
            classes=self.config.params_classes,
            freeze_all=True,
            freeze_till=None,
            lr=self.config.params_lr
        )
        self.save_model(path=self.config.updated_base_model_path,model=self.full_model)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        # Modelin mimarisini JSON formatında kaydet
        model_json = model.to_json()
        json_path = path.with_suffix('.json')
        with open(json_path, "w") as json_file:
            json_file.write(model_json)
        print(f"Model architecture saved to {json_path}")

        # Ağırlıkları HDF5 formatında kaydet
        weights_path = path.with_suffix('.h5')
        model.save_weights(weights_path)
        print(f"Model weights saved to {weights_path}")
