import warnings
warnings.simplefilter("ignore", UserWarning)
import logging
import torch
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import LADataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
import models
from config import opts
import os
import ipdb


def load_weights(model, weight_path, args):
    if weight_path != "random":
        pretrained_weights = torch.load(weight_path, map_location=torch.device('cpu'))
        model_weights = model.state_dict()
        
        if args.name != "m3d":
            pretrained_weights = pretrained_weights['model']


        load_weights = {}
        for k, v in pretrained_weights.items():
            if "visual_extractor.model." + k in model_weights:
                load_weights["visual_extractor.model." + k] = v

        print(len(load_weights))
        # print("load weights")
        # for k, _ in load_weights.items():
        #     print(k)

        model_weights.update(load_weights)
        model.load_state_dict(model_weights)
    return model


def main():
    # parse arguments
    # args = parse_agrs()
    # print the number of visible GPUs
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    args = opts.parse_opt()
    logging.info(str(args))

    # fix random seeds
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(args.seed)

    # create tokenizer
    tokenizer = Tokenizer(args)

    # create data loader
    train_dataloader = LADataLoader(args, tokenizer, split='train', shuffle=True)
    val_dataloader = LADataLoader(args, tokenizer, split='val', shuffle=False)
    test_dataloader = LADataLoader(args, tokenizer, split='test', shuffle=False)

    # build model architecture
    model_name = f"LAMRGModel_{args.version}"
    logging.info(f"Model name: {model_name} \tModel Layers:{args.num_layers}")

    model = getattr(models, model_name)(args, tokenizer)

    model = load_weights(model, args.pretrained_path, args)

    # for (images_id, images, reports_ids, reports_masks, labels) in train_dataloader:
    #     model(images, reports_ids, labels, mode='train')

    # get function handles of loss and metrics
    criterion = compute_loss
    metrics = compute_scores

    # build optimizer, learning rate scheduler
    optimizer = build_optimizer(args, model)
    lr_scheduler = build_lr_scheduler(args, optimizer)

    # build trainer and start to train
    trainer = Trainer(model, criterion, metrics, optimizer, args, lr_scheduler, train_dataloader, val_dataloader, test_dataloader)
    trainer.train()
    logging.info(str(args))


if __name__ == '__main__':
    main()
