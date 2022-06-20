import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import accuracy_score
from torch.nn.modules import CrossEntropyLoss

from captchami.loaders import CaptchaDataset, ImgToTensor
from captchami.model import NetModel


class NeuralNet:

    def __init__(self, l_i: int, classes: int, loaders: CaptchaDataset):
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self.loaders = loaders
        self.learning_rate = 0.001
        self.num_epochs = 200
        self.best_model = None

        self.model = NetModel(in_channels=loaders.get_num_channels(), classes=classes,
                              batch_size=loaders.get_batch_size(), linear_input=l_i)

    def train(self,):
        """
        This method trains the neural network.

        Returns: None
        """
        train_loader = self.loaders.get_trainloader()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate, weight_decay=5e-4)
        loss_function = CrossEntropyLoss()
        self.model.to(self.device)
        best_loss = float("inf")

        print("Start training")
        stats = pd.DataFrame(columns=["epoch", "loss", "accuracy"])
        self.model.train()
        for epoch in range(self.num_epochs):

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                optimizer.zero_grad()

                output = self.model(inputs)

                loss = loss_function(output, labels)
                loss.backward()
                optimizer.step()

                loss = loss.item()
                accuracy = self.__accuracy(output, labels)
                print("Epoch: {} Loss {} Accuracy {}".format(epoch, loss, accuracy))
                stats = stats.append({"epoch": epoch, "loss": loss, "accuracy": accuracy}, ignore_index=True)
                if loss < best_loss:
                    best_loss = loss
                    self.best_model = self.model

        print("Finished Training")
        self.model = self.best_model

        print("Start Testing")
        test_loader = self.loaders.get_testloader()
        self.model.eval()

        labels_list = []
        output_list = []
        for inputs, labels in test_loader:
            output = self.model(inputs.to(self.device))
            output = torch.max(output, 1)[1]
            output_list.extend(output.cpu())
            labels_list.extend(labels.cpu())

        print("Accuracy on test set: {}".format(accuracy_score(labels_list, output_list)))
        f, ax = plt.subplots(nrows=2)
        sns.lineplot(data=stats, x="epoch", y="accuracy", ax=ax[0])
        sns.lineplot(data=stats, x="epoch", y="loss", ax=ax[1])
        plt.savefig("stats.pdf")

    def save(self, path: str) -> None:
        """
        Save the current neural network to file

        Args:
            path: the path of the file to save

        Returns: None
        """
        torch.save(self.model.state_dict(), path)

    def classify_file(self, path: str) -> int:
        """
        Classify an image file using the current neural network.
        It returns the value of the class expressed in the dataset with which it has been trained

        Args:
            path: the path to the file to classify

        Returns: the value of the class for the given image

        """
        self.model.eval()
        img_tensor = ImgToTensor(path).get_img_tensor()
        img_tensor = torch.reshape(img_tensor, (1,) + tuple(img_tensor.shape))
        output = self.model(img_tensor)
        return int(torch.max(output, 1)[1])

    def classify_img(self, img: torch.Tensor) -> int:
        """
        Classify a given image tensor using the current neural network.
        It returns the value of the class expressed in the dataset with which it has been trained

        Args:
            img: the 32x32 image converted to a tensor

        Returns: the value of the class for the given image
        """
        self.model.eval()
        img = torch.reshape(img, (1, 1) + tuple(img.shape))
        output = self.model(img)
        return int(torch.max(output, 1)[1])

    @staticmethod
    def __accuracy(output: torch.Tensor, labels: np.ndarray) -> float:
        """
        Calculate the accuracy of a given couple of tensors

        Args:
            output: the output tensor from the neural network
            labels: the numpy array containing the right classes

        Returns: the accuracy value
        """
        predictions = torch.max(output, 1)[1]
        correct = predictions.eq(labels)
        correct = correct.sum()
        return correct.item() / len(labels)

    def load(self, path: str) -> None:
        """
        Load the state dictionary of the neural network from file and set it to be used on "CPU"

        Args:
            path: the file path containing the .pt of the neural network

        Returns: None

        """
        self.model.load_state_dict(torch.load(path, map_location="cpu"))
        self.model.eval()
