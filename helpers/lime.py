from lime import lime_image
from helpers.cnn import ConvolutionalNeuralNetwork
from torch.utils.data import DataLoader
from helpers.dataset import BarkVN50Dataset
from numpy import array
from torch import no_grad, device
from torch.nn.functional import softmax


def lime_evaluate_cnn(
    model: ConvolutionalNeuralNetwork, test_dataset: BarkVN50Dataset, device: device
):
    # TODO: implement LIME analysis

    def batch_predict(images):
        m = model.eval().to(device)
        images = images.to(device)
        
        logits = m(images)
        probs = softmax(logits, dim=1)
        return probs.detach().cpu().numpy()

    with no_grad():
        test_loader = DataLoader(test_dataset, batch_size=40)
        for image_batch, _ in test_loader:
            print(image_batch.shape)

            
            explainer = lime_image.LimeImageExplainer()
            _ = explainer.explain_instance(
                array(image_batch),
                batch_predict,
                top_labels=5,
                hide_color=0,
                num_samples=1000,
            )
