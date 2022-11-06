from jarmuiranyitas_1.agents.misc.mlp import MultiLayerPerceptron


class NNInitializer:

    def __init__(self, nn_type: str, size: list, seed: int):
        self.size = size
        self.seed = seed

        if nn_type == 'mlp':
            self.generate_nn = self.generate_mlp
        elif nn_type == 'cnn':
            self.generate_nn = self.generate_cnn
        else:
            raise NotImplementedError

    def generate_mlp(self):
        return MultiLayerPerceptron(self.size, self.seed)

    def generate_cnn(self):
        # TODO: implement CNN network initialization
        raise NotImplementedError
