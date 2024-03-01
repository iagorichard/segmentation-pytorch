from src.unet import UNet
## e outros modelos...

class ModelGenerator:

    def __init__(self, model_list, n_channels, n_classes):
        self.model_list       = model_list
        self.n_channels       = n_channels
        self.n_classes        = n_classes
        self.available_models = ["unet"]
        self.models           = self.__get_models_dict__()

    def __model_mapper__(self, model_name):
        if model_name == 'unet':
            return UNet(n_channels=self.n_channels, n_classes=self.n_classes).cuda()

    def __get_models_dict__(self):

        models_dict = {}

        for model_name in self.model_list:
            assert model_name in self.available_models, f"Model '{model_name}' is not valid. Please use some of these models: {self.available_models}."
            
            models_dict[model_name] = self.__model_mapper__(model_name)

        return models_dict