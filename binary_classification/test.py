import torch
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

import json
import os
from tqdm import tqdm

from binary_classification.loader import Food_5K_dataset
from binary_classification.model import *
from binary_classification.Measure import AverageMeter


if __name__ == '__main__':
    torch.cuda.empty_cache()

    # pt_path = '/hdd/food_classification_output/20220106/125501/best.pt' # TODO
    pt_path = '/hdd/food_classification_output/20220105/203444/best.pt'  # TODO
    # pt_path = '/hdd/food_classification_output/20220105/212147/best.pt'  # TODO
    ckpt = torch.load(pt_path)

    print(f"Validation Acc: {ckpt['acc']}")
    print(f"Final Loss: {ckpt['loss']}")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    # test_loader = DataLoader(
    #     Food_5K_dataset('/hdd/Food-5K/evaluation/', transform=transform, phase='test'),
    #     batch_size=1,
    #     num_workers=4, shuffle=True) # TODO: class 개수 수정
    test_loader = DataLoader(
        Food_5K_dataset('/hdd/MergeFood/', transform=transform, phase='test'),
        batch_size=1,
        num_workers=4, shuffle=True)

    model = Resnet50(n_class=3).cuda() # TODO
    # model = MobileNet(n_class=2).cuda()
    model.load_state_dict(ckpt['model_state_dict'])
    model = nn.DataParallel(model)

    softmax = nn.Softmax()

    result = dict()
    result['model'] = '' # TODO
    result['ckpt'] = pt_path
    result['precision'] = ckpt['acc']
    result['loss'] = ckpt['loss'].item()
    result['epoch'] = ckpt['epoch']
    result['test_acc'] = .0
    result['classes'] = test_loader.dataset.classes
    result['result'] = []

    pbar = tqdm(test_loader, ncols=150)
    correct = AverageMeter()
    model.eval()

    with torch.no_grad():
        for i, (label, image, path) in enumerate(test_loader, start=1):
            outputs = model(image.cuda())
            outputs = softmax(outputs)

            prob, indice = torch.topk(outputs.cpu(), k=1)
            correct.update(torch.sum(indice[:, 0:1] == label.reshape(-1, 1)).item())
            pbar.set_description(
                f'accuracy: {correct.val / test_loader.batch_size:.4f}({correct.sum / (i * test_loader.batch_size):.4f}) '
            )

            labels = [[result['classes'][i] for i in l] for l in indice]
            result['result'].extend([{'image': i[0],
                                      'gt_index': int(i[1]),
                                      'gt': result['classes'][i[1]],
                                      'predict': i[5],
                                      'predict_index': [int(j) for j in i[4]],
                                      'probability': [float(j) for j in i[3]]
                                      } for i in zip(path, label.numpy(), label, prob.numpy(), indice.numpy(), labels)])

            pbar.update()

        log = (f'accuracy: {correct.sum / test_loader.dataset.__len__():.4f}')
        pbar.set_description(log)
        pbar.close()

        result['test_acc'] = correct.sum / test_loader.dataset.__len__()

    if not os.path.isdir('./result/'): # TODO
        os.makedirs('./result/')
    json.dump(result, open('result/result.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
