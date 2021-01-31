# -*- coding: utf-8 -*-
import os
import sys

import torch
from torch import optim, nn
from sklearn.metrics import f1_score, recall_score, precision_score

import settings.parameters as para


class LowResEDModel(nn.Module):

    def __init__(self, support_sentence_encoder, query_sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        self.cost: loss function
        '''
        nn.Module.__init__(self)
        self.support_sentence_encoder = support_sentence_encoder
        self.query_sentence_encoder = query_sentence_encoder
        self.cost = nn.CrossEntropyLoss()
        self.loss_for_td = nn.MSELoss()

        self.fc_trigger = nn.Sequential(
            nn.Dropout(para.DROPOUT_RATE),
            nn.Linear(para.SIZE_EMB_WORD, para.SIZE_TRIGGER_LABEL, bias=True),
            nn.ReLU(),
        )

    def forward(self, support, query, scope_support, scope_query, N, R, K, Q):
        '''
        support: Inputs of the support set.
        query: Inputs of the query set.
        N: Num of classes
        R: Ratio of instances for each class
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        return: logits
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def evaluation_metric(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Evaluation Metric]
        '''
        # average=None, get the P, R, and F1 value of a single class
        if para.CUDA:
            pred = pred.cpu()
            label = label.cpu()

        Precision = precision_score(y_true=label.view(-1), y_pred=pred.view(-1), average="macro")
        Recall = recall_score(y_true=label.view(-1), y_pred=pred.view(-1), average="macro")
        F1_score = f1_score(y_true=label.view(-1), y_pred=pred.view(-1), average="macro")
        F1_score_micro = f1_score(y_true=label.view(-1), y_pred=pred.view(-1), average="micro")
        return Precision, F1_score, Recall, F1_score_micro

    def concat_support_query_items(self, items_support_trigger, items_query_trigger, item_scale=2):
        if item_scale == 2:
            Max_Num = items_support_trigger.size(-1)
            items_support_trigger = items_support_trigger.view(para.SIZE_BATCH, -1, Max_Num)
            items_query_trigger = items_query_trigger.view(para.SIZE_BATCH, -1, Max_Num)
            items_trigger = torch.cat((items_support_trigger, items_query_trigger), dim=1).view(-1, Max_Num)
        elif item_scale == 3:
            D = items_support_trigger.size(-1)
            Max_Num = items_support_trigger.size(-2)
            items_support_trigger = items_support_trigger.view(para.SIZE_BATCH, -1, Max_Num, D)
            items_query_trigger = items_query_trigger.view(para.SIZE_BATCH, -1, Max_Num, D)
            items_trigger = torch.cat((items_support_trigger, items_query_trigger), dim=1).view(-1, Max_Num, D)
        return items_trigger


class LowResEDFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        return x.item()

    def train(self, model, model_name,
              B, N_for_train, N_for_eval, R_for_train, R_for_eval, K, Q,
              ckpt_dir=para.CHECKPOINT_DIRECTORY,
              test_result_dir=para.DATA_OUTPUT_DIRECTORY,
              learning_rate=para.LR,
              lr_step_size=para.SIZE_LR_STEP,
              weight_decay=para.WEIGHT_DECAY,
              train_iter=para.TRAIN_ITER,
              val_iter=para.VAL_ITER,
              val_step=para.VAL_STEP,
              test_iter=para.TEST_ITER,
              cuda=para.CUDA,
              pretrain_model=None,
              optimizer=optim.SGD,
              noise_rate=0):
        '''
        model: a LowResEDModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")

        # Init
        parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        if cuda:
            model = model.cuda()
        model.train()

        # Training
        best_acc = 0
        not_best_count = 0  # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_precision = 0.0
        iter_f1 = 0.0
        iter_f1_micro = 0.0
        iter_recall = 0.0
        iter_loss_ec = 0.0
        iter_loss_td = 0.0
        iter_right_td = 0.0
        iter_precision_td = 0.0
        iter_f1_td = 0.0
        iter_f1_td_micro = 0.0
        iter_recall_td = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            support, query, label, label_support_trigger, label_query_trigger, scope_support, scope_query = \
                self.train_data_loader.next_batch(B, N_for_train, R_for_train, K, Q, noise_rate=noise_rate)

            logits, pred, logits_support_trigger, pred_support_trigger, logits_query_trigger, pred_query_trigger = \
                model(support, query, scope_support, scope_query, N_for_train, R_for_train, K, Q)
            # logits.size() -> (B, N_for_train * Q, N_for_train)
            # pred.size() -> (B * N_for_train * Q)
            # logits_support_trigger.size() -> (B, N_for_train * K, W, 2)
            # pred_support_trigger.size() -> (B, N_for_train * K, W)
            # logits_query_trigger.size() -> (B, N_for_train * Q, W, 2)
            # pred_query_trigger.size() -> (B, N_for_train * Q, W)
            # support_trigger_label.size() -> (B, N_for_train * K, W)
            # query_trigger_label.size() -> (B, N_for_train * Q, W)

            if torch.cuda.device_count() > 1:
                logits_trigger = model.module.concat_support_query_items(logits_support_trigger, logits_query_trigger, 3)
                label_trigger = model.module.concat_support_query_items(label_support_trigger, label_query_trigger, 2)
                pred_trigger = model.module.concat_support_query_items(pred_support_trigger, pred_query_trigger, 2)
                loss_td = model.module.loss(logits_trigger, label_trigger)
                loss_ec = model.module.loss(logits, label)
                loss = para.LOSS_RATIO_FOR_EC * loss_ec + para.LOSS_RATIO_FOR_TD * loss_td

                right_td = model.module.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.module.evaluation_metric(pred_trigger, label_trigger)

                right = model.module.accuracy(pred, label)
                Precision, F1_score, Recall, F1_score_micro = model.module.evaluation_metric(pred, label)
            else:
                logits_trigger = model.concat_support_query_items(logits_support_trigger, logits_query_trigger, 3)
                label_trigger = model.concat_support_query_items(label_support_trigger, label_query_trigger, 2)
                pred_trigger = model.concat_support_query_items(pred_support_trigger, pred_query_trigger, 2)
                loss_td = model.loss(logits_trigger, label_trigger)
                loss_ec = model.loss(logits, label)
                loss = para.LOSS_RATIO_FOR_EC * loss_ec + para.LOSS_RATIO_FOR_TD * loss_td

                right_td = model.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.evaluation_metric(pred_trigger, label_trigger)

                right = model.accuracy(pred, label)
                Precision, F1_score, Recall, F1_score_micro = model.evaluation_metric(pred, label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_loss += self.item(loss.data)
            iter_loss_ec += loss_ec
            iter_precision += Precision
            iter_recall += Recall
            iter_f1 += F1_score
            iter_f1_micro += F1_score_micro
            iter_right += self.item(right.data)
            iter_loss_td += loss_td
            iter_precision_td += Precision_td
            iter_recall_td += Recall_td
            iter_f1_td += F1_score_td
            iter_f1_td_micro += F1_score_td_micro
            iter_right_td += self.item(right_td.data)
            iter_sample += 1

            sys.stdout.write(
                '[TRAIN] step: {0:4} | loss: {1:2.6f}, '
                'loss_ec: {2:2.6f}, precision: {3:2.6f}, recall: {4:2.6f}, '
                'f1: {5:2.6f}, f1_micro: {6:2.6f}, accuracy: {7:3.2f}%, '
                'loss_td: {8:2.6f}, precision_td: {9:2.6f}, recall_td: {10:2.6f}, '
                'f1_td: {11:2.6f}, f1_td_micro: {12:2.6f}, accuracy_td: {13:3.2f}%'.format(
                    it + 1, iter_loss / iter_sample, iter_loss_ec / iter_sample, iter_precision / iter_sample, iter_recall / iter_sample,
                    iter_f1 / iter_sample, iter_f1_micro / iter_sample, 100 * iter_right / iter_sample,
                    iter_loss_td / iter_sample, iter_precision_td / iter_sample, iter_recall_td / iter_sample,
                    iter_f1_td / iter_sample, iter_f1_td_micro / iter_sample, 100 * iter_right_td / iter_sample) + '\r')
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_loss_ec = 0.
                iter_precision = 0.
                iter_recall = 0.
                iter_f1 = 0.
                iter_right = 0.
                iter_loss_td = 0.
                iter_precision_td = 0.
                iter_recall_td = 0.
                iter_f1_td = 0.
                iter_right_td = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                precision, recall, f1, f1_micro, acc, precision_td, recall_td, f1_td, f1_td_micro, acc_td,  = self.eval(model, model_name, B, N_for_eval, R_for_eval, K, Q, val_iter, noise_rate=noise_rate)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + '_' + str(N_for_train) + "Way-" + str(R_for_train) + "Ratio-Max" + str(K) + "Shot" + ".pth.tar")
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = acc

        print("\n======================================================================================\n")
        print("Finish training " + model_name)
        test_precision, test_recall, test_f1, test_f1_micro, test_acc, test_precision_td, test_recall_td, test_f1_td,  test_f1_td_micro, test_acc_td = \
            self.eval(model, model_name, B, N_for_eval, R_for_eval, K, Q, test_iter,
                                      ckpt=os.path.join(ckpt_dir, model_name + '_' + str(N_for_train) + "Way-" + str(R_for_train) + "Ratio-Max" + str(K) + "Shot" + '.pth.tar'), noise_rate=noise_rate)
        print("\n======================================================================================\n")
        print("Finish testing " + model_name)
        print("LR:", para.LR, "LOSS_RATIO_FOR_TD:", para.LOSS_RATIO_FOR_TD, "LOSS_RATIO_FOR_EC:",
              para.LOSS_RATIO_FOR_EC, "LOSS_RATIO_FOR_ERE:", para.LOSS_RATIO_FOR_ERE)
        print(model_name + '_' + str(N_for_train) + "Way-" + str(R_for_train) + "Ratio-Max" + str(K) + "Shot")
        print("Test precision_td: {}".format(test_precision_td))
        print("Test recall_td: {}".format(test_recall_td))
        print("Test f1_td: {}".format(test_f1_td))
        print("Test f1_td_micro: {}".format(test_f1_td_micro))
        print("Test accuracy_td: {}".format(test_acc_td))
        print("Test precision: {}".format(test_precision))
        print("Test recall: {}".format(test_recall))
        print("Test f1: {}".format(test_f1))
        print("Test f1_micro: {}".format(test_f1_micro))
        print("Test accuracy: {}".format(test_acc))

    def eval(self, model, model_name, B, N, R, K, Q, eval_iter, ckpt=None, noise_rate=0):
        '''
        model: a FewShotEDModel instance
        B: Batch size
        N: Num of classes for each batch
        R: Ratio of instances
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")

        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader

        iter_precision = 0.0
        iter_recall = 0.0
        iter_f1 = 0.0
        iter_f1_micro = 0
        iter_right = 0.0
        iter_precision_td = 0.0
        iter_recall_td = 0.0
        iter_f1_td = 0.0
        iter_f1_td_micro = 0.0
        iter_right_td = 0.0
        iter_sample = 0.0

        for it in range(eval_iter):
            support, query, label, label_support_trigger, label_query_trigger, scope_support, scope_query = \
                eval_dataset.next_batch(B, N, R, K, Q, noise_rate=noise_rate)
            logits, pred, logits_support_trigger, pred_support_trigger, logits_query_trigger, pred_query_trigger = \
                model(support, query, scope_support, scope_query, N, R, K, Q)
            if torch.cuda.device_count() > 1:
                label_trigger = model.module.concat_support_query_items(label_support_trigger, label_query_trigger, 2)
                pred_trigger = model.module.concat_support_query_items(pred_support_trigger, pred_query_trigger, 2)
                right_td = model.module.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.module.evaluation_metric(pred_trigger, label_trigger)
                right = model.module.accuracy(pred, label)
                Precision, F1_score, Recall, F1_score_micro = model.module.evaluation_metric(pred, label)
            else:
                label_trigger = model.concat_support_query_items(label_support_trigger, label_query_trigger, 2)
                pred_trigger = model.concat_support_query_items(pred_support_trigger, pred_query_trigger, 2)
                right_td = model.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.evaluation_metric(pred_trigger, label_trigger)
                right = model.accuracy(pred, label)
                Precision, F1_score, Recall, F1_score_micro = model.evaluation_metric(pred, label)

            iter_precision += Precision
            iter_recall += Recall
            iter_f1 += F1_score
            iter_f1_micro += F1_score_micro
            iter_right += self.item(right.data)
            iter_precision_td += Precision_td
            iter_recall_td += Recall_td
            iter_f1_td += F1_score_td
            iter_f1_td_micro += F1_score_td_micro
            iter_right_td += self.item(right_td.data)
            iter_sample += 1
            sys.stdout.write('[EVAL] step: {0:4} | '
                             'precision: {1:2.6f}, recall: {2:2.6f}, f1: {3:2.6f}, f1_micro: {4:2.6f}, accuracy: {5:3.2f}%, '
                             'precision_td: {6:2.6f}, recall_td: {7:2.6f}, f1_td: {8:2.6f}, f1_td_micro: {9:2.6f}, accuracy_td: {10:3.2f}%'.format(
                it + 1, iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample, iter_f1_micro / iter_sample,
                100 * iter_right / iter_sample, iter_precision_td / iter_sample, iter_recall_td / iter_sample,
                iter_f1_td / iter_sample, iter_f1_td_micro / iter_sample, 100 * iter_right_td / iter_sample) + '\r')

            sys.stdout.flush()
        print("")

        return iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample, iter_f1_micro / iter_sample, iter_right / iter_sample, iter_precision_td / iter_sample, iter_recall_td / iter_sample, iter_f1_td / iter_sample, iter_f1_td_micro / iter_sample, iter_right_td / iter_sample


class OverallEDModel(nn.Module):

    def __init__(self, sentence_encoder):
        '''
        sentence_encoder: Sentence encoder
        self.cost: loss function
        '''
        nn.Module.__init__(self)
        self.sentence_encoder = sentence_encoder
        self.cost = nn.CrossEntropyLoss()
        # MSELoss, NLLLoss, KLDivLoss
        self.loss_for_td = nn.MSELoss()

        self.fc_trigger = nn.Sequential(
            nn.Dropout(para.DROPOUT_RATE),
            nn.Linear(para.SIZE_EMB_WORD, para.SIZE_TRIGGER_LABEL, bias=True),
            nn.ReLU(),
        )

    def forward(self, inputs):
        '''
        inputs: Inputs of the overall set.
        return: logits
        '''
        raise NotImplementedError

    def loss(self, logits, label):
        '''
        logits: Logits with the size (..., class_num)
        label: Label with whatever size.
        return: [Loss] (A single value)
        '''
        N = logits.size(-1)
        return self.cost(logits.view(-1, N), label.view(-1))

    def accuracy(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Accuracy] (A single value)
        '''
        return torch.mean((pred.view(-1) == label.view(-1)).type(torch.FloatTensor))

    def evaluation_metric(self, pred, label):
        '''
        pred: Prediction results with whatever size
        label: Label with whatever size
        return: [Evaluation Metric]
        '''
        # average=None, get the P, R, and F1 value of a single class
        if para.CUDA:
            pred = pred.cpu()
            label = label.cpu()
        Precision = precision_score(y_true=label.view(-1), y_pred=pred.view(-1), average="macro")
        Recall = recall_score(y_true=label.view(-1), y_pred=pred.view(-1), average="macro")
        F1_score = f1_score(y_true=label.view(-1), y_pred=pred.view(-1), average="macro")
        F1_score_micro = f1_score(y_true=label.view(-1), y_pred=pred.view(-1), average="micro")
        return Precision, F1_score, Recall, F1_score_micro


class OverallEDFramework:

    def __init__(self, train_data_loader, val_data_loader, test_data_loader):
        '''
        train_data_loader: DataLoader for training.
        val_data_loader: DataLoader for validating.
        test_data_loader: DataLoader for testing.
        '''
        self.train_data_loader = train_data_loader
        self.val_data_loader = val_data_loader
        self.test_data_loader = test_data_loader

    def __load_model__(self, ckpt):
        '''
        ckpt: Path of the checkpoint
        return: Checkpoint dict
        '''
        if os.path.isfile(ckpt):
            checkpoint = torch.load(ckpt)
            print("Successfully loaded checkpoint '%s'" % ckpt)
            return checkpoint
        else:
            raise Exception("No checkpoint found at '%s'" % ckpt)

    def item(self, x):
        '''
        PyTorch before and after 0.4
        '''
        return x.item()

    def train(self, model, model_name,
              B,
              ckpt_dir=para.CHECKPOINT_DIRECTORY,
              test_result_dir=para.DATA_OUTPUT_DIRECTORY,
              learning_rate=para.LR,
              lr_step_size=para.SIZE_LR_STEP,
              weight_decay=para.WEIGHT_DECAY,
              train_iter=para.TRAIN_ITER,
              val_iter=para.VAL_ITER,
              val_step=para.VAL_STEP,
              test_iter=para.TEST_ITER,
              cuda=para.CUDA,
              pretrain_model=None,
              optimizer=optim.SGD,
              noise_rate=0):
        '''
        model: a LowResEDModel instance
        model_name: Name of the model
        B: Batch size
        N: Num of classes for each batch
        K: Num of instances for each class in the support set
        Q: Num of instances for each class in the query set
        ckpt_dir: Directory of checkpoints
        test_result_dir: Directory of test results
        learning_rate: Initial learning rate
        lr_step_size: Decay learning rate every lr_step_size steps
        weight_decay: Rate of decaying weight
        train_iter: Num of iterations of training
        val_iter: Num of iterations of validating
        val_step: Validate every val_step steps
        test_iter: Num of iterations of testing
        cuda: Use CUDA or not
        pretrain_model: Pre-trained checkpoint path
        '''
        print("Start training...")

        # Init
        parameters_to_optimize = filter(lambda x: x.requires_grad, model.parameters())
        optimizer = optimizer(parameters_to_optimize, learning_rate, weight_decay=weight_decay)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=lr_step_size)
        if pretrain_model:
            checkpoint = self.__load_model__(pretrain_model)
            model.load_state_dict(checkpoint['state_dict'])
            start_iter = checkpoint['iter'] + 1
        else:
            start_iter = 0

        if torch.cuda.device_count() > 1:
            print("Let's use", torch.cuda.device_count(), "GPUs!")
            model = nn.DataParallel(model)

        if cuda:
            model = model.cuda()
        model.train()

        # Training
        best_acc = 0
        not_best_count = 0  # Stop training after several epochs without improvement.
        iter_loss = 0.0
        iter_right = 0.0
        iter_precision = 0.0
        iter_f1 = 0.0
        iter_f1_micro = 0
        iter_recall = 0.0
        iter_loss_ec = 0.0
        iter_loss_td = 0.0
        iter_right_td = 0.0
        iter_precision_td = 0.0
        iter_f1_td = 0.0
        iter_f1_td_micro = 0
        iter_recall_td = 0.0
        iter_sample = 0.0
        for it in range(start_iter, start_iter + train_iter):
            scheduler.step()
            overall, label_event, label_trigger = self.train_data_loader.next_batch_for_overall(B, noise_rate=noise_rate)

            logits, pred, logits_trigger, pred_trigger = model(overall)
            # logits.size() -> (B, #All_Instances, N)
            # pred.size() -> (B * #All_Instances)
            # logits_trigger.size() -> (B, #All_Instances, W, 2)
            # pred_trigger.size() -> (B, #All_Instances, W)
            # trigger_label.size() -> (B, #All_Instances, W)

            if torch.cuda.device_count() > 1:
                loss_td = model.module.loss(logits_trigger, label_trigger)
                loss_ec = model.module.loss(logits, label_event)
                loss = para.LOSS_RATIO_FOR_EC * loss_ec + para.LOSS_RATIO_FOR_TD * loss_td

                right_td = model.module.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.module.evaluation_metric(pred_trigger, label_trigger)

                right = model.module.accuracy(pred, label_event)
                Precision, F1_score, Recall, F1_score_micro = model.module.evaluation_metric(pred, label_event)
            else:
                loss_td = model.loss(logits_trigger, label_trigger)
                loss_ec = model.loss(logits, label_event)
                loss = para.LOSS_RATIO_FOR_EC * loss_ec + para.LOSS_RATIO_FOR_TD * loss_td

                right_td = model.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.evaluation_metric(pred_trigger, label_trigger)

                right = model.accuracy(pred, label_event)
                Precision, F1_score, Recall, F1_score_micro = model.evaluation_metric(pred, label_event)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_loss += self.item(loss.data)
            iter_loss_ec += loss_ec
            iter_precision += Precision
            iter_recall += Recall
            iter_f1 += F1_score
            iter_f1_micro += F1_score_micro
            iter_right += self.item(right.data)
            iter_loss_td += loss_td
            iter_precision_td += Precision_td
            iter_recall_td += Recall_td
            iter_f1_td += F1_score_td
            iter_f1_td_micro += F1_score_td_micro
            iter_right_td += self.item(right_td.data)
            iter_sample += 1

            sys.stdout.write(
                '[TRAIN] step: {0:4} | loss: {1:2.6f}, '
                'loss_ec: {2:2.6f}, precision: {3:2.6f}, recall: {4:2.6f}, '
                'f1: {5:2.6f}, f1_micro: {6:2.6f}, accuracy: {7:3.2f}%, '
                'loss_td: {8:2.6f}, precision_td: {9:2.6f}, recall_td: {10:2.6f}, '
                'f1_td: {11:2.6f}, f1_td_micro: {12:2.6f}, accuracy_td: {13:3.2f}%'.format(it + 1,
                     iter_loss / iter_sample, iter_loss_ec / iter_sample, iter_precision / iter_sample, iter_recall / iter_sample,
                     iter_f1 / iter_sample, iter_f1_micro / iter_sample, 100 * iter_right / iter_sample,
                     iter_loss_td / iter_sample, iter_precision_td / iter_sample, iter_recall_td / iter_sample,
                     iter_f1_td / iter_sample, iter_f1_td_micro / iter_sample, 100 * iter_right_td / iter_sample) + '\r')
            sys.stdout.flush()
            sys.stdout.flush()

            if it % val_step == 0:
                iter_loss = 0.
                iter_loss_ec = 0.
                iter_precision = 0.
                iter_recall = 0.
                iter_f1 = 0.
                iter_right = 0.
                iter_loss_td = 0.
                iter_precision_td = 0.
                iter_recall_td = 0.
                iter_f1_td = 0.
                iter_right_td = 0.
                iter_sample = 0.

            if (it + 1) % val_step == 0:
                precision, recall, f1, f1_micro, acc, precision_td, recall_td, f1_td, f1_td_micro, acc_td,  = self.eval(model, model_name, B, val_iter, noise_rate=noise_rate)
                model.train()
                if acc > best_acc:
                    print('Best checkpoint')
                    if not os.path.exists(ckpt_dir):
                        os.makedirs(ckpt_dir)
                    save_path = os.path.join(ckpt_dir, model_name + '_Overall' + '.pth.tar')
                    torch.save({'state_dict': model.state_dict()}, save_path)
                    best_acc = acc

        print("\n======================================================================================\n")
        print("Finish training " + model_name)
        test_precision, test_recall, test_f1, test_f1_micro, test_acc, test_precision_td, test_recall_td, test_f1_td, test_f1_td_micro, test_acc_td = \
            self.eval(model, model_name, B, test_iter, ckpt=os.path.join(ckpt_dir, model_name + '_Overall' + '.pth.tar'), noise_rate=noise_rate)
        print("\n======================================================================================\n")
        print("Finish testing " + model_name)
        print("LR:", para.LR, "LOSS_RATIO_FOR_TD:", para.LOSS_RATIO_FOR_TD, "LOSS_RATIO_FOR_EC:",
              para.LOSS_RATIO_FOR_EC, "LOSS_RATIO_FOR_ERE:", para.LOSS_RATIO_FOR_ERE)
        print(model_name + '_Overall')
        print("Test precision_td: {}".format(test_precision_td))
        print("Test recall_td: {}".format(test_recall_td))
        print("Test f1_td: {}".format(test_f1_td))
        print("Test f1_td_micro: {}".format(test_f1_td_micro))
        print("Test accuracy_td: {}".format(test_acc_td))
        print("Test precision: {}".format(test_precision))
        print("Test recall: {}".format(test_recall))
        print("Test f1: {}".format(test_f1))
        print("Test f1_micro: {}".format(test_f1_micro))
        print("Test accuracy: {}".format(test_acc))

    def eval(self, model, model_name, B, eval_iter, ckpt=None, noise_rate=0):
        '''
        model: a FewShotEDModel instance
        B: Batch size
        N: Num of classes for each batch
        eval_iter: Num of iterations
        ckpt: Checkpoint path. Set as None if using current model parameters.
        return: Accuracy
        '''
        print("")

        model.eval()
        if ckpt is None:
            eval_dataset = self.val_data_loader
        else:
            checkpoint = self.__load_model__(ckpt)
            model.load_state_dict(checkpoint['state_dict'])
            eval_dataset = self.test_data_loader

        iter_precision = 0.0
        iter_recall = 0.0
        iter_f1 = 0.0
        iter_f1_micro = 0.0
        iter_right = 0.0
        iter_precision_td = 0.0
        iter_recall_td = 0.0
        iter_f1_td = 0.0
        iter_f1_td_micro = 0.0
        iter_right_td = 0.0
        iter_sample = 0.0

        for it in range(eval_iter):
            overall, label_event, label_trigger = eval_dataset.next_batch_for_overall(B, noise_rate=noise_rate)
            logits, pred, logits_trigger, pred_trigger = model(overall)
            if torch.cuda.device_count() > 1:
                right_td = model.module.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.module.evaluation_metric(pred_trigger, label_trigger)
                right = model.module.accuracy(pred, label_event)
                Precision, F1_score, Recall, F1_score_micro = model.module.evaluation_metric(pred, label_event)
            else:
                right_td = model.accuracy(pred_trigger, label_trigger)
                Precision_td, F1_score_td, Recall_td, F1_score_td_micro = model.evaluation_metric(pred_trigger, label_trigger)
                right = model.accuracy(pred, label_event)
                Precision, F1_score, Recall, F1_score_micro = model.evaluation_metric(pred, label_event)
            iter_precision += Precision
            iter_recall += Recall
            iter_f1 += F1_score
            iter_f1_micro += F1_score_micro
            iter_right += self.item(right.data)
            iter_precision_td += Precision_td
            iter_recall_td += Recall_td
            iter_f1_td += F1_score_td
            iter_f1_td_micro += F1_score_td_micro
            iter_right_td += self.item(right_td.data)
            iter_sample += 1
            sys.stdout.write('[EVAL] step: {0:4} | '
                             'precision: {1:2.6f}, recall: {2:2.6f}, f1: {3:2.6f}, f1_micro: {4:2.6f}, accuracy: {5:3.2f}%, '
                             'precision_td: {6:2.6f}, recall_td: {7:2.6f}, f1_td: {8:2.6f}, f1_td_micro: {9:2.6f}, accuracy_td: {10:3.2f}%'.format(
                it + 1, iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample,
                iter_f1_micro / iter_sample, 100 * iter_right / iter_sample, iter_precision_td / iter_sample,
                iter_recall_td / iter_sample, iter_f1_td / iter_sample, iter_f1_td_micro / iter_sample,
                100 * iter_right_td / iter_sample) + '\r')
            sys.stdout.flush()
        print("")

        return iter_precision / iter_sample, iter_recall / iter_sample, iter_f1 / iter_sample, iter_f1_micro / iter_sample, iter_right / iter_sample, iter_precision_td / iter_sample, iter_recall_td / iter_sample, iter_f1_td / iter_sample, iter_f1_td_micro / iter_sample, iter_right_td / iter_sample
