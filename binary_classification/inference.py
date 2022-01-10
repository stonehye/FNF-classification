from torchvision.transforms import transforms

import argparse
import glob
from PIL import Image

from binary_classification.model import *


def inference(ckpt, img):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
    image = transform(img)
    image = torch.unsqueeze(image, 0)

    # model = MobileNet(n_class=2).cuda() # TODO
    model = Resnet50(n_class=3).cuda()  # TODO
    model.load_state_dict(ckpt['model_state_dict'])
    model = nn.DataParallel(model)
    softmax = nn.Softmax()
    model.eval()

    with torch.no_grad():
        outputs = model(image.cuda())
        outputs = softmax(outputs)
        prob, indice = torch.topk(outputs, 1)

        return indice.item()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description="Test for Food/Non-food Binary Classification")
    # parser.add_argument('-i', '--image', type=str, required=True)
    # args = parser.parse_args()

    torch.cuda.empty_cache()

    # pt_path = '/hdd/food_classification_output/20220105/203444/best.pt'  # TODO
    pt_path = '/hdd/food_classification_output/20220107/110534/best.pt'
    ckpt = torch.load(pt_path)

    dataset_path = '/hdd/KoreanFood/*.jpg'
    # dataset_path = '/hdd/multiple_object/*.jpg'
    print(len(glob.glob(dataset_path)))
    for img_path in glob.glob(dataset_path): # TODO
        image = Image.open(img_path)
        predict = inference(ckpt, image)
        print(img_path, predict)
