import os
import numpy as np
import sys
import cPickle as pickle
import random
import cv2
import theano
import theano.tensor as T
import lasagne
from sympy.utilities.iterables import cartes
from tqdm import tqdm
from models.model_salgan import ModelSALGAN
from models.model_bce import ModelBCE
from evaluation import Evaluation
import pdb
import matplotlib
import argparse
import glob

#####################################
#To bypass X11 for matplotlib in tmux
matplotlib.use('Agg')
import matplotlib.pyplot as plt
#####################################

def bce_batch_iterator(model, train_data, validation_data, epochs = 10, fig=False):
    num_epochs = epochs+1
    n_updates = 1
    nr_batches_train = int(len(train_data) / model.batch_size)

    for current_epoch in tqdm(range(num_epochs), ncols=20):
        e_cost = 0.
        random.shuffle(train_data)

        for currChunk in chunks(train_data, model.batch_size):
            if len(currChunk) != model.batch_size:
                continue

            #Prepare data
            batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],dtype=theano.config.floatX)
            batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],dtype=theano.config.floatX)
            batch_output = np.expand_dims(batch_output, axis=1)

            #Feed data to the model
            G_cost = model.G_trainFunction(batch_input, batch_output)

            e_cost += G_cost;
            n_updates += 1

        e_cost /= nr_batches_train

        print('\tEpoch: [{0:02}/{1}]\t'
              'TrainCost: {2}'.format(current_epoch, num_epochs, e_cost))

        if current_epoch % 5 == 0:
            np.savez('../weights/gen_modelWeights{:04d}.npz'.format(current_epoch),
                     *lasagne.layers.get_all_param_values(model.net['output']))

# def salgan_batch_iterator(model, train_data, validation_data,epochs = 20, fig=False):
#     num_epochs = epochs+1
#     nr_batches_train = int(len(train_data) / model.batch_size)
#     train_loss_plt, train_acc_plt, val_loss_plt, val_acc_plt = [[] for i in range(4)]
#     n_updates = 1
#     for current_epoch in tqdm(range(num_epochs), ncols=20):
#     g_cost = 0.; d_cost = 0.; e_cost = 0.
#         random.shuffle(train_data)
#         for currChunk in chunks(train_data, model.batch_size):
#             if len(currChunk) != model.batch_size:
#                 continue
#             batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],dtype=theano.config.floatX)
#             batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],dtype=theano.config.floatX)
#             batch_output = np.expand_dims(batch_output, axis=1)
#             if n_updates % 2 == 0:
#                 G_obj, D_obj, G_cost = model.G_trainFunction(batch_input, batch_output)
#                 d_cost += D_obj; g_cost += G_obj; e_cost += G_cost
#             else:
#                 G_obj, D_obj, G_cost = model.D_trainFunction(batch_input, batch_output)
#                 d_cost += D_obj; g_cost += G_obj; e_cost += G_cost
#             n_updates += 1
#         g_cost /= nr_batches_train
#     d_cost /= nr_batches_train
#     e_cost /= nr_batches_train
#     #Compute the Jaccard Index on the Validation
#     v_cost, v_acc = bce_feedforward(model,validation_data,True)
#
#     if current_epoch % 5  == 0:
#             np.savez('./' + DIR_TO_SAVE + '/gen_modelWeights{:04d}.npz'.format(current_epoch),
#                      *lasagne.layers.get_all_param_values(model.net['output']))
#             np.savez('./' + DIR_TO_SAVE + '/disrim_modelWeights{:04d}.npz'.format(current_epoch),
#                      *lasagne.layers.get_all_param_values(model.discriminator['fc5']))
#     return v_acc

# def bce_feedforward(model, validation_data, bPrint=False):
#     nr_batches_val = int(len(validation_data) / model.batch_size)
#     v_cost = 0.
#     v_acc = 0.
#     for currChunk in chunks(validation_data, model.batch_size):
#         if len(currChunk) != model.batch_size:
#             continue
#         batch_input = np.asarray([x.image.data.astype(theano.config.floatX).transpose(2, 0, 1) for x in currChunk],dtype=theano.config.floatX)
#         batch_output = np.asarray([y.saliency.data.astype(theano.config.floatX) / 255. for y in currChunk],dtype=theano.config.floatX)
#         batch_output = np.expand_dims(batch_output, axis=1)
#         val_loss, val_accuracy = model.G_valFunction(batch_input,batch_output)
#         v_cost += val_loss
#         v_acc += val_accuracy
#     v_cost /= nr_batches_val
#     v_acc /= nr_batches_val
#     if bPrint is True:
#         print "  validation_accuracy -->", v_acc
#     print "  validation_loss -->", v_cost
#     print "-----------------------------------------"
#     return v_cost, v_acc


def load_weights(net, path, epochtoload):
    with np.load(path  + "modelWeights{:04d}.npz".format(epochtoload)) as f:
        param_values = [f['arr_%d' % i] for i in range(len(f.files))]
    lasagne.layers.set_all_param_values(net['output'], param_values)

def main(args):
    if args.mode == 'train':
        print 'Loading training data...'
        with open(args.trainset, 'rb') as f:
            train_data = pickle.load(f)
        print '-->done!'

        print 'Loading test data...'
        with open(args.valset, 'rb') as f:
            validation_data = pickle.load(f)
        print '-->done!'

        # Create network
        if args.model == 'salgan':

            model_args = [args.width, args.height, args.batch_size, args.lr, args.regul_term, args.momentum]
            model = ModelSALGAN(*model_args)

            if args.resume:
                load_weights(net=model.net['output'], path="weights/gen_", epochtoload=args.resume)
                load_weights(net=model.discriminator['fc5'], path="weights/disrim_", epochtoload=args.resume)
            #salgan_batch_iterator(model, train_data, validation_data,epochs=args.num_epochs)

        elif args.model == 'bce':
            model_args = [args.width, args.height, args.batch_size, args.lr, args.regul_term, args.momentum]
            model = ModelBCE(*model_args)

            if args.resume:
                load_weights(net=model.net['output'], path='weights/gen_', epochtoload=args.resume)
            bce_batch_iterator(model, train_data, validation_data,epochs=args.num_epochs)

        else:
            print "Invalid Model Argument."

    elif args.mode == 'test':
        model = ModelBCE()
        with np.load('../weights/gen_' + "modelWeights{:04d}.npz".format(args.test_epoch)) as f:
            param_values = [f['arr_%d' % i] for i in range(len(f.files))]
        lasagne.layers.set_all_param_values(model.net['output'], param_values)

        list_img_files = [k.split('/')[-1].split('.')[0] for k in glob.glob(os.path.join(args.imgdir, 'val_*.bmp'))]
        for curr_file in tqdm(list_img_files):
            img = cv2.cvtColor(cv2.imread(os.path.join(args.imgdir, curr_file + '.bmp'), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)

            img = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_AREA)

            blob = np.zeros((1, 3, args.height, args.width), theano.config.floatX)
            blob[0, ...] = (img.astype(theano.config.floatX).transpose(2, 0, 1))

            result = np.squeeze(model.predictFunction(blob))
            seg_map = (result * 255).astype(np.uint8)

            seg_map = cv2.resize(seg_map , (args.width, args.height), interpolation=cv2.INTER_CUBIC)
            seg_map = np.clip(seg_map , 0, 255)

            cv2.imwrite(os.path.join(args.resdir, curr_file + '_'+ args.arch +'.bmp'), seg_map)

    elif args.mode == 'eval':
        evaluator = Evaluation()

        evaluator(args.gtdir, args.resdir, 'unet')
        evaluator.print_vals()





# def cross_val(args):
#     # Load data
#     print 'Loading training data...'
#     with open(TRAIN_DATA_DIR_CROSS, 'rb') as f:
#         train_data = pickle.load(f)
#     print '-->done!'
#
#     print 'Loading validation data...'
#     with open(VAL_DATA_DIR, 'rb') as f:
#         validation_data = pickle.load(f)
#     print '-->done!'
#
#     if args.model == 'bce':
#         lr_list = [0.1,0.01,0.001,0.05]
#         regterm_list = [1e-1,1e-2,1e-3,1e-4,1e-5]
#         momentum_list = [0.9,0.99]
#         lr,regterm,mom,acc = [[] for i in range(4)]
#         for config_list in list(cartes(lr_list,regterm_list,momentum_list)):
#             model = ModelBCE(INPUT_SIZE[0], INPUT_SIZE[1],16,config_list[0],config_list[1],config_list[2])
#             val_accuracy = bce_batch_iterator(model, train_data, validation_data,epochs=10)
#             lr.append(config_list[0])
#             regterm.append(config_list[1])
#             mom.append(config_list[2])
#             acc.append(val_accuracy)
#         for l,r,m,a in zip(lr,regterm,mom,acc):
#             print ("lr: {}, lambda: {}, momentum: {}, accuracy: {}").format(l,r,m,a)
#             print('------------------------------------------------------------------')
#
#             print('--------------------------------The Best--------------------------')
#             best_idx = np.argmax(acc)
#             print ("lr: {}, lambda: {}, momentum: {}, accuracy: {}").format(lr[best_idx],regterm[best_idx],mom[best_idx],acc[best_idx])
#     elif args.model == 'salgan':
#         G_lr_list = [0.1,0.01,0.05]
#         regterm_list = [1e-1,1e-2,1e-3,1e-4,1e-5]
#         D_lr_list = [0.1,0.01,0.05]
#         alpha_list = [1/5., 1/10., 1/20.]
#         G_lr,regterm,D_lr,alpha,acc = [[] for i in range(5)]
#         for config_list in list(cartes(G_lr_list,regterm_list,D_lr_list,alpha_list)):
#             model = ModelSALGAN(INPUT_SIZE[0], INPUT_SIZE[1],9,config_list[0],config_list[1],config_list[2],config_list[3])
#             val_accuracy = salgan_batch_iterator(model, train_data, validation_data,epochs=10)
#             G_lr.append(config_list[0])
#             regterm.append(config_list[1])
#             D_lr.append(config_list[2])
#             alpha.append(config_list[3])
#             acc.append(val_accuracy)
#         for g_l,r,d_l,al,a in zip(G_lr,regterm,D_lr,alpha,acc):
#             print ("G_lr: {}, lambda: {}, D_lr: {}, alpha: {}, accuracy: {}").format(g_l,r,d_l,al,a)
#             print('------------------------------------------------------------------')
#
#         print('--------------------------------The Best--------------------------')
#         best_idx = np.argmax(acc)
#         print ("G_lr: {}, lambda: {}, D_lr: {}, alpha: {}, accuracy: {}").format(G_lr[best_idx],regterm[best_idx],D_lr[best_idx],alpha[best_idx],acc[best_idx])
#     else:
#         print("Please provide a correct argument")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='bce')

    subparsers = parser.add_subparsers(dest='mode')
    subparsers.required = True

    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--trainset', default='../data/pickle320x240/trainData.pickle')
    parser_train.add_argument('--valset', default='../data/pickle320x240/validationData.pickle')
    parser_train.add_argument('--lr', default=0.05, type=float)
    parser_train.add_argument('--momentum', default=0.99, type=float)
    parser_train.add_argument('--regul-term', default=1e-05, type=float)
    parser_train.add_argument('--alpha', default=0.2, type=float)
    parser_train.add_argument('--batch-size', default=10, type=int)
    parser_train.add_argument('--width', default=320, type=int)
    parser_train.add_argument('--num-epochs', default=10, type=int)
    parser_train.add_argument('--height', default=240, type=int)
    parser_train.add_argument('--resume', type=int, required=False)

    parser_test = subparsers.add_parser('test')
    parser_test.add_argument('--test-epoch', default=10, type=int)
    parser_test.add_argument('--imgdir', default='../data/image320x240/', type=str)
    parser_test.add_argument('--resdir', default='../data/results', type=str)
    parser_test.add_argument('--width', default=320, type=int)
    parser_test.add_argument('--height', default=240, type=int)
    parser_test.add_argument('--arch', default='unet', type=str)

    parser_eval = subparsers.add_parser('eval')
    parser_eval.add_argument('--resdir', type=str, default='../data/results')
    parser_eval.add_argument('--gtdir', type=str, default='../data/mask320x240')
    


    main(parser.parse_args())






    #parser_train = subparsers.add_parser('crossval')



