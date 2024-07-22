from pathlib import Path
import random
import argparse
import sys

import torch
import torch.nn.functional as F
import numpy as np 
import pandas as pd
import math

from datasets.utils import get_splits
from models.alexnet import CaffeNet
from models.resnet import Resnet
from models.heads import CaffeNetDiscriminator, ResNetDiscriminator, BrainCancerDiscriminator
from models.discriminator import Discriminator
from models.brain_cancer import BrainCancer
from models.entropyLoss import HLoss
from datasets.datasets import Augmentation
from datasets.utils import normed_tensors, random_color_jitter, \
    matsuura_augmentation
from models.utils import set_random_seed, train_adversarial_examples, \
    evaluate, evaluate_regression
    
from wilds.common.data_loaders import get_train_loader, get_eval_loader

optim_select=''


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--verbose',
                        help='Print log file',
                        action='store_true')
    parser.add_argument('--gpu', type=int, help='GPU idx to run', default=0)
    parser.add_argument('--save_dir',
                        type=str,
                        help='Write directory',
                        default='output')
    parser.add_argument('--model',
                        type=str,
                        choices=['caffenet', 'resnet', 'brainCancer'],
                        help='Model',
                        default='brainCancer')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['PACS', 'VLCS', 'OfficeHome', 'OfficeHome_larger', 'openBHB'],
        help='Dataset',
        default='OpenBHB')
    parser.add_argument('--leave_out_domain',
                        type=str,
                        help='Domain to leave out for hyper-param selection',
                        default=None)
    parser.add_argument(
        '--single_target',
        type=str,
        help='If a single target is required, specify it here.',
        default=None)
    parser.add_argument('--features_lr',
                        type=float,
                        help='Feature extractor learning rate',
                        default=1e-3)
    parser.add_argument('--classifier_lr',
                        type=float,
                        help='Classifier learning rate',
                        default=1e-2)
    parser.add_argument('--num_epochs',
                        type=int,
                        help='Number of epochs',
                        default=50)
    parser.add_argument('--batch_size',
                        type=int,
                        help='Batch size',
                        default=11)
    parser.add_argument('--lr_step',
                        type=int,
                        help='Steps between LR decrease',
                        default=24)
    parser.add_argument('--momentum',
                        type=float,
                        help='Momentum for SGD',
                        default=0.9)
    parser.add_argument('--weight_decay',
                        type=float,
                        help='Weight Decay',
                        default=1e-2)
    parser.add_argument(
        '--adversarial_examples',
        help='Train with examples adversarial to Feature Extractor',
        action='store_true')
    parser.add_argument('--nnet_generator',
                        help='Use a nnet to generate, instead of SGD.',
                        action='store_true')
    parser.add_argument(
        '--ablate_blur',
        help='Check to make the boost is not coming from blurring.',
        action='store_true')
    parser.add_argument('--adversarial_examples_lr',
                        type=float,
                        help='Learning rate for adversarial examples',
                        default=1e-2)
    parser.add_argument('--adversarial_examples_wd',
                        type=float,
                        help='Weight Decay for adversarial examples',
                        default=1e-2)
    parser.add_argument('--adversarial_train_steps',
                        type=int,
                        help='Steps to train adversarial examples',
                        default=50)
    parser.add_argument('--adversarial_examples_ratio',
                        type=float,
                        help='Ratio of adversarial examples',
                        default=0.5)
    parser.add_argument('--adv_blur_step',
                        type=int,
                        help='How many steps between blurring',
                        default=4)
    parser.add_argument('--adv_smoothing_step',
                        type=int,
                        help='How many steps between smoothing',
                        default=4)
    parser.add_argument('--adv_smoothing_strength',
                        type=int,
                        help='Kernel size for smoothing',
                        default=3)
    parser.add_argument('--adv_blur_sigma',
                        type=float,
                        help='Size of sigma in blurring',
                        default=1)
    parser.add_argument('--domain_adversary',
                        help='Discriminate domains adversarially',
                        action='store_true')
    parser.add_argument('--domain_adversary_weight',
                        type=float,
                        help='Weight for domain adversarial loss',
                        default=1.0)
    parser.add_argument('--domain_adversary_lr',
                        type=float,
                        help='Learning rate for domain adversary',
                        default=1e-2)
    parser.add_argument('--early_adversary_supression',
                        help='Suppress adversaries 2/{1+exp{-k*p}}-1',
                        action='store_true')
    parser.add_argument('--supression_decay',
                        type=float,
                        help='Weight to use in supression',
                        default=10.0)
    parser.add_argument('--color_jitter',
                        help='Apply random color jitter',
                        action='store_true')
    parser.add_argument('--matsuura_augmentation',
                        help='Use matsuuras augmentation. ' \
                            'Takes precedence over --color_jitter. '\
                            'Ignores --cj_mag.',
                        action='store_true')
    parser.add_argument('--wandb',
                        type=str,
                        help='Plot on wandb ',
                        default=None)
    parser.add_argument('--entropy',
                        help='entropy loss; default is False',
                        action='store_true')
    parser.add_argument('--entropy_weight',
                        type=float,
                        help='entropy loss weight; default is 1',
                        default=1)
    parser.add_argument('--random_seed',
                        type=int,
                        help='random seed',
                        default=None)
    parser.add_argument('--dann_size',
                        type=float,
                        help='Size of DANN Network',
                        default=1)
    parser.add_argument('--dann_depth',
                        type=int,
                        help='Depth of DANN Network',
                        default=1)
    parser.add_argument('--dann_conv_layers',
                        help='Use conv layers as input to DANN',
                        action='store_true')
    parser.add_argument('--adv_kl_weight',
                        type=float,
                        help='Use kl loss on classes for adv examples',
                        default=1.0)
    parser.add_argument('--save_adversarial_examples',
                        help='Save examples adversarial to DANN',
                        action='store_true')
    parser.add_argument('--only_augment_half',
                        help='Only augment 50 percent of the images',
                        action='store_true')
    parser.add_argument('--no_adversary_on_original',
                        help='adversary on color jitter only',
                        action='store_true')
    parser.add_argument('--use_original_train_set',
                        help='whether to use the original train set.',
                        action='store_true')
    parser.add_argument('--cj_mag',
                        type=int,
                        help='magnitude of color jitter',
                        default=1)
    parser.add_argument('--add_val',
                        help='add validation into pipeline',
                        action='store_true')
    parser.add_argument('--classify_adv_exp',
                        help='classify adversarial examples',
                        action='store_true')
    parser.add_argument('--blur_at_last_step',
                        help='blur at the last step',
                        action='store_true')
    parser.add_argument('--smooth_at_last_step',
                        help='smooth at the last step',
                        action='store_true')
    parser.add_argument('--even_lower_lr_vlcs',
                        help='even lower lr for vlcs',
                        action='store_true')

    args = parser.parse_args()
    if args.dataset == 'VLCS':
        args.features_lr /= 10
        args.classifier_lr /= 10
        args.domain_adversary_lr /= 10
        if args.even_lower_lr_vlcs:
            args.features_lr /= 10
            args.classifier_lr /= 10
            args.domain_adversary_lr /= 10

    print('args: ', args)
    if args.random_seed is not None:
        set_random_seed(args.random_seed)

    if args.wandb is not None:
        import wandb
        wandb.init(project=args.wandb, name=args.save_dir)
    else:
        wandb = None

    args.use_rgb_convert = (args.dataset == 'VLCS')

    if args.entropy:
        print('using entropy')
        print(f'entropy weight:{args.entropy_weight}')

    # print('************************   model split   **************')
    splits, num_classes, num_domains = get_splits(
        args.dataset,
        leave_out=args.leave_out_domain,
        original=args.use_original_train_set)
    # print('************************   model split Done   **************')

    device = torch.device(
        f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")

    for heldout in splits.keys():
        # print('************************   heldout loop   **************')

        if args.single_target is not None and heldout != args.single_target:
            # print('************************   continue?  **************')
            continue

        _save_dir = '../results/' + args.save_dir + '/' + heldout
        Path(_save_dir).mkdir(parents=True, exist_ok=True)
        if not args.verbose:
            sys.stdout = open(f'{_save_dir}/log.txt', 'w')
        # print('************************   result folder done   **************')

        print('args: ', args)
        print('nums_domain: ', num_domains)

        lr_groups = []
        # print('************************   model start   **************')

        if args.model == 'caffenet':
            print('Using caffenet')
            model = CaffeNet(num_classes=num_classes).to(device)
            DiscHead = CaffeNetDiscriminator
            lr_groups.extend([(model.features.parameters(), args.features_lr),
                              (model.classifier.parameters(),
                               args.classifier_lr)])
            c_loss_fn = torch.nn.CrossEntropyLoss()
        elif args.model == 'resnet':
            print('Using resnet')
            model = Resnet(num_classes=num_classes).to(device)
            DiscHead = ResNetDiscriminator
            lr_groups.extend([
                (model.base_model.conv1.parameters(), args.features_lr),
                (model.base_model.bn1.parameters(), args.features_lr),
                (model.base_model.layer1.parameters(), args.features_lr),
                (model.base_model.layer2.parameters(), args.features_lr),
                (model.base_model.layer3.parameters(), args.features_lr),
                (model.base_model.layer4.parameters(), args.features_lr),
                (model.base_model.fc.parameters(), args.classifier_lr)
            ])
            c_loss_fn = torch.nn.CrossEntropyLoss()
        elif args.model == 'brainCancer':
            print('Using brainCancer')
            model = BrainCancer()
            model.to('cuda:2')
            # net = torch.nn.DataParallel(model)
            DiscHead = BrainCancerDiscriminator
            lr_groups.extend([
                (model.conv1.parameters(), args.features_lr),
                (model.conv2.parameters(), args.features_lr),
                (model.conv3.parameters(), args.features_lr),
                (model.conv4.parameters(), args.features_lr),
                (model.conv5.parameters(), args.features_lr),
                (model.conv6.parameters(), args.features_lr),
                (model.output.parameters(), args.features_lr)
            ])
            c_loss_fn = torch.nn.MSELoss()
            
            

        else:
            raise Exception(f'{args.model} not supported.')

        if args.domain_adversary:
            print('Using domain adversary')
            domain_incr = 1
            if args.classify_adv_exp:
                domain_incr *= 2
            d_loss_fn = torch.nn.CrossEntropyLoss()
            if args.model == 'caffenet':
                domain_adversary = Discriminator(
                    DiscHead(num_domains * domain_incr,
                             size=args.dann_size,
                             depth=args.dann_depth,
                             conv_input=args.dann_conv_layers)).to(device)
            elif args.model == 'resnet':
                domain_adversary = Discriminator(
                    DiscHead(num_domains * domain_incr)).to(device)
            elif args.model == 'brainCancer':
                domain_adversary = Discriminator(
                    DiscHead(64,  num_domains * domain_incr)).to(device)
                d_loss_fn = torch.nn.CrossEntropyLoss()
                d_loss_fn.double()
            else:
                raise AssertionError("model unrecognized")
            lr_groups.append(
                (domain_adversary.parameters(), args.domain_adversary_lr))
        
        # print('************************   optim start   **************')
        if(optim_select=='Adam'):
            optimizers = [
                torch.optim.Adam(params,
                                lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
                for params, lr in lr_groups
            ]
            
        else:
            optimizers = [
                torch.optim.SGD(params,
                                lr,
                                momentum=args.momentum,
                                nesterov=True,
                                weight_decay=args.weight_decay)
                for params, lr in lr_groups
            ]

        schedulers = [
            torch.optim.lr_scheduler.StepLR(optim,
                                            step_size=args.lr_step,
                                            gamma=1e-1) for optim in optimizers
        ]

        if (args.dataset=='OpenBHB'):
            trainset = splits[heldout]['train']
            valset = splits[heldout]['val']
            testset = splits[heldout]['test']
        else:
            # print('************************   train start   **************')

            trainset = splits[heldout]['train']()
            # print('************************   train complete   **************')
            valset = splits[heldout]['val']()

            # print('************************   vali complete   **************')
            testset = splits[heldout]['test']()
            # print('************************   test complete   **************')
            
        if(args.dataset=='OpenBHB'):
            train = get_train_loader("standard", trainset, batch_size=args.batch_size, drop_last=True, num_workers = 10)
            val = get_eval_loader("standard", valset, batch_size=args.batch_size, drop_last=True, num_workers = 10)
            test = get_eval_loader("standard", testset, batch_size=args.batch_size, drop_last=True, num_workers = 10)
        else:
            if args.matsuura_augmentation:
                trainset = Augmentation(trainset,
                                        baseline=normed_tensors(),
                                        augmentation=matsuura_augmentation(),
                                        augment_half=args.only_augment_half,
                                        use_rgb_convert=args.use_rgb_convert)
            elif args.color_jitter:
                print('Using random augmentation')
                trainset = Augmentation(
                    trainset,
                    baseline=normed_tensors(),
                    augmentation=random_color_jitter(magnitude=args.cj_mag),
                    augment_half=args.only_augment_half,
                    use_rgb_convert=args.use_rgb_convert)
            else:
                print('Using no augmentation')
                trainset = Augmentation(trainset,
                                        baseline=normed_tensors(),
                                        augment_half=args.only_augment_half,
                                        use_rgb_convert=args.use_rgb_convert)

            train = torch.utils.data.DataLoader(trainset,
                                                batch_size=args.batch_size,
                                                drop_last=False,
                                                num_workers=6,
                                                shuffle=True)
            if args.add_val:
                valset = Augmentation(splits[heldout]['val'](),
                                    baseline=normed_tensors(),
                                    use_rgb_convert=args.use_rgb_convert)

                val = torch.utils.data.DataLoader(valset,
                                                batch_size=args.batch_size,
                                                drop_last=False,
                                                num_workers=6,
                                                shuffle=True)

            testset = Augmentation(splits[heldout]['test'](),
                                baseline=normed_tensors(),
                                use_rgb_convert=args.use_rgb_convert)

            test = torch.utils.data.DataLoader(testset,
                                            batch_size=args.batch_size,
                                            drop_last=False,
                                            num_workers=6,
                                            shuffle=True)

        print(f'Starting {heldout}...')
        print('train dataset size: ',len(train.dataset))
        print('val dataset size: ',len(val.dataset))
        print('test dataset size: ',len(test.dataset))

        best_val_acc = 0
        best_val_acc_epoch = 0
        best_test_acc = 0
        best_test_acc_epoch = 0
        anomaly = False

        generator_state = None
        
        Train_MAE = []
        Validation_MAE = []
        Test_MAE = []
        MAE_func = torch.nn.L1Loss()

        Train_MSE = []
        Validation_MSE = []
        Test_MSE = []
        
        Epoch = []
        # Batch = []
        model_device = 'cuda:2'

        for epoch in range(0, args.num_epochs):

            running_loss_class = 0.0
            running_loss_model = 0.0
            running_loss_domain = 0.0
            running_loss_entropy = 0.0
            running_accuracy = []
            if args.save_adversarial_examples:
                args.adv_img_saved_this_epoch = False

            print(f'Starting epoch {epoch}...')
            Epoch.append(epoch)
            p = epoch / args.num_epochs
            supression = (2.0 / (1. + np.exp(-args.supression_decay * p)) - 1)
            print(f'supression={supression:.2f}')

            if args.domain_adversary:
                beta = 1 * args.domain_adversary_weight
                # print("domain_adversary_weight: ", args.domain_adversary_weight)
                if args.early_adversary_supression:
                    beta *= supression
                print(f'beta={beta:.2f}')
                domain_adversary.set_beta(beta)

            for i, batch in enumerate(train):
                torch.cuda.empty_cache()
                print(f'batch={i}')

                if(args.dataset=='OpenBHB'):                 
                    x = batch[0].to(device)
                    # print("dataset shape is: ", batch[0].shape)
                    y = batch[1][:, 0].double().to(device)
                    # print("reuslt label shape is: ", batch[1][:, 0].shape)
                    d =  batch[2][:, 0].clone().detach().to(device)#  torch.tensor().unsqueeze(0).to(device)
                    # print("domain shape is: ", batch[2][:, 0].shape)

                    a = False
                
                else:
                    x = batch['img'].to(device)
                    y = batch['label'].to(device)
                    d = batch['domain'].to(device)
                    a = batch['augmented'].to(device)


                # print(batch)
                # print('======================shape outside:' ,x.shape)
                if args.adversarial_examples and not args.nnet_generator:
                    assert args.domain_adversary
                    if args.classify_adv_exp:
                        x, d = train_adversarial_examples(x,
                                                          d,
                                                          a,
                                                          num_domains,
                                                          args,
                                                          model,
                                                          domain_adversary,
                                                          d_loss_fn,
                                                          beta,
                                                          device,
                                                          epoch=epoch,
                                                          i=i,
                                                          wandb=wandb,
                                                          heldout=heldout)
                    else:
                        x, _ = train_adversarial_examples(x,
                                                          d,
                                                          a,
                                                          num_domains,
                                                          args,
                                                          model,
                                                          domain_adversary,
                                                          d_loss_fn,
                                                          beta,
                                                          device,
                                                          epoch=epoch,
                                                          i=i,
                                                          wandb=wandb,
                                                          heldout=heldout)
                    args.adv_img_saved_this_epoch = True

                print("\n\n\n\n\n\n\n---------------------------- TRAIN ADVERSARIAL EXAMPLES DONE \n\n\n\n\n\n\n")
                model.train()
                # print("---------------========== x: ", x.shape)
                z_conv = model.conv_features(x.to(model_device)).to(device)
                z = model.dense_features(z_conv.to(model_device)).to(device)
                yhat = model.classifier(z.to(model_device)).to(device)
                yhat = yhat.view(-1)
                # print("::::::::::::::::::::::::  yhat: ", yhat.shape, "y: ", y.shape)
                # print("type of yhat and y: ", yhat.dtype, y.dtype)
                c_loss = c_loss_fn(yhat.to(device), y.to(device))
                running_loss_class += c_loss
                # print("type of c_loss", c_loss.dtype)
                print(f'class_loss={c_loss:.2f}')
                # for p in model.parameters():
                #     print("parameter dtype: ", p.dtype)
                # for p in domain_adversary.parameters():
                #     print("parameter dtype: ", p.dtype)
                if args.entropy:
                    entropy_loss = HLoss()
                    e_loss = entropy_loss(yhat)
                    print(f'e_loss={e_loss:.2f}')
                    if args.early_adversary_supression:
                        e_loss = e_loss * (supression * args.entropy_weight)

                if args.domain_adversary:
                    domain_adversary.train()
                    # domain adversary should only work on sources,
                    # not synthetic examples
                    if args.dann_conv_layers:
                        dhat = domain_adversary(z_conv.double().to(device))
                        # print("type of dhat: ", dhat.dtype)

                    else:
                        dhat = domain_adversary(z.double().to(device))
                        # print("type of dhat: ", dhat.dtype)

                    d_loss = d_loss_fn(dhat.to(device), d.long().to(device))

                loss = c_loss
                # print("type of d_loss", d_loss.dtype)
                # print("type of e_loss", e_loss.dtype)

                if args.domain_adversary:
                    loss += d_loss
                if args.entropy:
                    loss += e_loss

                print(f'model_loss={loss:.2f}')

                if torch.isnan(loss):
                    anomaly = True
                    break

                running_loss_model += loss

                if args.domain_adversary:
                    running_loss_domain += d_loss  # * x.size(0)
                if args.entropy:
                    running_loss_entropy += e_loss

                for optim in optimizers:
                    optim.zero_grad()
                # print("loss dtype", loss.dtype)
                loss.backward()
                running_accuracy.append(MAE_func(yhat.to(device), y.to(device)).item())
                for optim in optimizers:
                    optim.step()

            Train_MAE.append(sum(running_accuracy)/len(running_accuracy))
            Train_MSE.append(running_loss_model.item()/len(running_accuracy))
            if anomaly:
                print('Found anomaly. Terminating.')
                break

            for scheduler in schedulers:
                scheduler.step()

            epoch_loss_class = running_loss_class / i
            print(f'epoch{epoch}_class_loss={epoch_loss_class:.2f}')
            if args.domain_adversary:
                epoch_loss_domain = running_loss_domain / i
                print(f'epoch{epoch}_domain_loss={epoch_loss_domain:.2f}')
            if args.entropy:
                epoch_loss_entropy = running_loss_entropy / i
                print(f'epoch{epoch}_entropy_loss={epoch_loss_entropy:.2f}')
            epoch_loss_model = running_loss_model / i  # len(train.dataset)
            print(f'epoch{epoch}_model_loss={epoch_loss_model:.2f}')

            if args.add_val:
                print(f'Starting validation...')
                if(args.dataset == 'OpenBHB'):
                    val_acc, val_loss = evaluate_regression(model, val, device, c_loss_fn,
                                                _save_dir) 
                else:            
                    val_acc, val_loss = evaluate(model, val, device, c_loss_fn,
                                                _save_dir)
                Validation_MSE.append(val_loss.item())
                Validation_MAE.append(val_acc.item())
                if val_acc > best_val_acc:
                    torch.save(model.state_dict(), f'{_save_dir}/val_model.pt')
                    best_val_acc = val_acc
                    best_val_acc_epoch = epoch
                print(f'val_class_loss={val_loss:.2f}')
                print(f'vall_accuracy={val_acc:.2f}')
                print(f'finished validation')

            # results NOT used in AISTATS
            # previously used to compare against models who report best test accuracy
            # AISTATS paper reports LAST test accuracy
            print(f'Starting test evaluation...')
            if(args.dataset == 'OpenBHB'):
                test_acc, test_loss = evaluate_regression(model, test, device, c_loss_fn,
                                            _save_dir)
            else:
                test_acc, test_loss = evaluate(model, test, device, c_loss_fn,
                            _save_dir)
            Test_MSE.append(test_loss.item())
            Test_MAE.append(test_acc.item())
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_acc_epoch = epoch
            print(f'test_class_loss={test_loss:.2f}')
            print(f'test_accuracy={test_acc:.2f}')
            print(f'Finished testing')

            print(f'Finished epoch {epoch}')
            if args.wandb is not None:
                wandb.log({
                    f"{heldout}_epoch":
                    epoch,
                    f"{heldout}_class_loss":
                    epoch_loss_class,
                    f"{heldout}_domain_loss":
                    epoch_loss_domain if args.domain_adversary else None,
                    f"{heldout}_entropy_loss":
                    epoch_loss_entropy if args.entropy else None,
                    f"{heldout}_model_loss":
                    epoch_loss_model,
                    f"{heldout}_test_class_loss (MSE)":
                    test_loss,
                    f"{heldout}_test_accuracy (MAE)":
                    test_acc,
                    f"{heldout}_val_class_loss (MSE)":
                    val_loss if args.add_val else None,
                    f"{heldout}_val_accuracy (MAE)":
                    val_acc if args.add_val else None
                })

        print(f'Saving the model used to test...')
        torch.save(model.state_dict(), f'{_save_dir}/model.pt')

        if args.add_val:
            print('Starting testing on best val model...')
            model.load_state_dict(torch.load(f'{_save_dir}/val_model.pt'))
            if(args.dataset == 'OpenBHB'):
                val_test_acc, val_test_loss = evaluate_regression(model, test, device,
                                                    c_loss_fn, _save_dir)            
            else:
                val_test_acc, val_test_loss = evaluate(model, test, device,
                                                    c_loss_fn, _save_dir)
            
            print(f'test(val)_class_loss={val_test_loss:.4f}')
            print(f'test(val)_accuracy={val_test_acc:.2f}')

        # results used in AISTATS paper based on last model
        print('Starting testing on last model...')
        model.load_state_dict(torch.load(f'{_save_dir}/model.pt'))
        if(args.dataset == 'OpenBHB'):
            acc, loss = evaluate_regression(model, test, device, c_loss_fn, _save_dir)
        else:
            acc, loss = evaluate(model, test, device, c_loss_fn, _save_dir)
        with open(f'../results/{args.save_dir}/results.txt', 'a') as res:
            res.write(f'{heldout}: acc={acc:.4f} loss={loss:.4f}\n')
        print(f'test_class_loss={loss:.2f}')
        print(f'test_accuracy={acc:.2f}')
        print(f'Finished testing')

        print(f'Finished {heldout}')
        if args.wandb is not None:
            wandb.log({
                f"{heldout}_final_test_class_loss (MSE)":
                loss,
                f"{heldout}_final_test_accuracy (MAE)":
                acc,
                f"{heldout}_best_test_acc (MAE)":
                best_test_acc,
                f"{heldout}_best_test_acc_epoch":
                best_test_acc_epoch,
                f"{heldout}_final_test(val)_class_loss (MSE)":
                val_test_loss if args.add_val else None,
                f"{heldout}_final_test(val)_test_accuracy (MAE)":
                val_test_acc if args.add_val else None
            })
    
    train_dataset = pd.DataFrame({'epoch': Epoch, 'MSE': Train_MSE, 'MAE': Train_MAE})
    val_dataset = pd.DataFrame({'epoch': Epoch, 'MSE': Validation_MSE, 'MAE': Validation_MAE})
    test_dataset = pd.DataFrame({'epoch': Epoch, 'MSE': Test_MSE, 'MAE': Test_MAE})
    train_dataset.to_csv(_save_dir+'/train_dataset.csv', index=False)
    val_dataset.to_csv(_save_dir+'/validation_dataset.csv', index=False)
    test_dataset.to_csv(_save_dir+'/test_dataset.csv', index=False)

    
