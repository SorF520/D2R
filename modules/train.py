import torch
from torch import optim
from tqdm import tqdm
import random
from sklearn.metrics import classification_report
from transformers.optimization import get_linear_schedule_with_warmup
from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score, precision_recall_fscore_support
import sklearn.metrics as metrics

import shutil


# def get_four_metrics(labels, predicted_labels):
#     confusion = metrics.confusion_matrix(labels, predicted_labels)
#     total = confusion[0][0] + confusion[0][1] + confusion[1][0] + confusion[1][1]
#     acc = (confusion[0][0] + confusion[1][1]) / total
#     # about sarcasm
#     recall = confusion[1][1] / (confusion[1][1] + confusion[1][0])
#     precision = confusion[1][1] / (confusion[1][1] + confusion[0][1])
#     f1 = 2 * recall * precision / (recall + precision)
#     return acc, recall, precision, f1

def get_four_metrics(labels, predicted_labels, type='weighted'):

    acc = accuracy_score(labels, predicted_labels)
    f1 = f1_score(labels, predicted_labels, average=type)
    recall = recall_score(labels, predicted_labels, average=type)
    precision = precision_score(labels, predicted_labels, average=type)

    return acc, recall, precision, f1


# def get_four_metrics(labels, predicted_labels, type='weighted'):
#
#     acc = accuracy_score(labels, predicted_labels)
#     precision, recall, f1, support_macro \
#         = precision_recall_fscore_support(labels, predicted_labels, average=type)
#
#     return acc, recall, precision, f1


class BaseTrainer(object):
    def train(self):
        raise NotImplementedError()

    def evaluate(self):
        raise NotImplementedError()

    def test(self):
        raise NotImplementedError()


class MSDTrainer(BaseTrainer):
    def __init__(self, train_data=None, dev_data=None, test_data=None, model=None,
                 args=None, logger=None, writer=None) -> None:
        self.train_data = train_data
        self.dev_data = dev_data
        self.test_data = test_data
        self.model = model
        self.logger = logger
        self.writer = writer
        self.args = args
        self.step = 0
        self.refresh_step = 2
        self.best_dev_metric = 0
        self.best_test_metric = 0
        self.best_train_metric = 0
        self.best_dev_epoch = None
        self.best_test_epoch = None
        self.best_train_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * args.num_epochs

        self.multiModal_before_train()

    def train(self, clip_model_dict=None, bert_model_dict=None):
        self.step = 0
        self.model.train()
        self.logger.info("***** Running training *****")
        self.logger.info("  Num instance = %d", len(self.train_data) * self.args.batch_size)
        self.logger.info("  Num epoch = %d", self.args.num_epochs)
        self.logger.info("  Batch size = %d", self.args.batch_size)
        self.logger.info("  Learning rate = {}".format(self.args.lr))
        self.logger.info("  Evaluate begin = %d", self.args.eval_begin_epoch)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load model successful!")

        vision_names, text_names = [], []
        model_dict = self.model.state_dict()

        for name in model_dict:
            if 'vision' in name:
                clip_name = name.replace('vision_', '').replace('model.', '')
                if clip_name in clip_model_dict:
                    vision_names.append(clip_name)
                    model_dict[name] = clip_model_dict[clip_name]

            elif 'text' in name:
                text_name = name.replace('text_', '').replace('model.', '')

                if text_name in bert_model_dict:
                    text_names.append(text_name)
                    model_dict[name] = bert_model_dict[text_name]

        assert len(vision_names) == len(clip_model_dict) and len(text_names) == len(bert_model_dict), \
            (len(vision_names), len(clip_model_dict), len(text_names), len(bert_model_dict))
        self.model.load_state_dict(model_dict)

        with tqdm(total=self.train_num_steps, postfix='loss:{0:<6.5f}', leave=False, dynamic_ncols=True,
                  initial=self.step) as pbar:
            self.pbar = pbar
            avg_loss = 0
            for epoch in range(1, self.args.num_epochs + 1):
                pbar.set_description_str(desc="Epoch {}/{}".format(epoch, self.args.num_epochs))
                for batch in self.train_data:
                    self.step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    (loss, logits), labels = self._step(batch, mode="train")
                    avg_loss += loss.detach().cpu().item()

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step()
                    self.optimizer.zero_grad()

                    if self.step % self.refresh_step == 0:
                        avg_loss = float(avg_loss) / self.refresh_step
                        print_output = "loss:{:<6.5f}".format(avg_loss)
                        pbar.update(self.refresh_step)
                        pbar.set_postfix_str(print_output)
                        if self.writer:
                            self.writer.add_scalar(tag='train_loss', scalar_value=avg_loss,
                                                   global_step=self.step)  # tensorbordx
                        avg_loss = 0

                if epoch >= self.args.eval_begin_epoch:
                    self.evaluate(epoch)  # generator to dev.
                    # self.test(epoch)

            # 取最好的模型进行测试     './output/'
            self.args.load_path = self.args.save_path + "best_model.pth"
            self.test(epoch)

            # 递归删除文件夹
            shutil.rmtree("./output")

            torch.cuda.empty_cache()
            pbar.close()

            self.pbar = None
            # self.logger.info("Get best dev performance at epoch {}, best dev f1 score is {}".format(self.best_dev_epoch,
            #                                                                                         self.best_dev_metric))
            # self.logger.info(
            #     "Get best test performance at epoch {}, best test f1 score is {}".format(self.best_test_epoch,
            #                                                                              self.best_test_metric))

    def evaluate(self, epoch):
        self.model.eval()
        self.logger.info("***** Running evaluate *****")
        self.logger.info("  Num instance = %d", len(self.dev_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        step = 0
        true_labels, pred_labels = [], []

        with torch.no_grad():
            with tqdm(total=len(self.dev_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Dev")
                total_loss = 0
                for batch in self.dev_data:
                    step += 1
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    (loss, logits), labels = self._step(batch, mode="dev")  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())
                    pbar.update()
                # evaluate done
                pbar.close()
                # result = classification_report(y_true=true_labels, y_pred=pred_labels, digits=4)
                acc, recall, precision, micro_f1 = get_four_metrics(true_labels, pred_labels, type='weighted')
                result = {
                    'eval_accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f_score': micro_f1,
                    'global_step': epoch,
                    'loss': total_loss
                }
                self.logger.info("***** Dev Eval results *****")
                for key in sorted(result.keys()):
                    self.logger.info("  %s = %s", key, str(result[key]))

                if self.writer:
                    self.writer.add_scalar(tag='dev_acc', scalar_value=acc, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_f1', scalar_value=micro_f1, global_step=epoch)  # tensorbordx
                    self.writer.add_scalar(tag='dev_loss', scalar_value=total_loss / len(self.dev_data),
                                           global_step=epoch)  # tensorbordx

                self.logger.info("Epoch {}/{}, best dev f1: {}, best epoch: {}, current dev f1 score: {}, acc: {}." \
                                 .format(epoch, self.args.num_epochs, self.best_dev_metric, self.best_dev_epoch,
                                         micro_f1, acc))
                if micro_f1 >= self.best_dev_metric:  # this epoch get best performance
                    self.logger.info("Get better performance at epoch {}".format(epoch))
                    self.best_dev_epoch = epoch
                    self.best_dev_metric = micro_f1  # update best metric(f1 score)
                    if self.args.save_path is not None:
                        torch.save(self.model.state_dict(), self.args.save_path + "best_model.pth")
                        self.logger.info("Save best model at {}".format(self.args.save_path))

        self.model.train()

    def test(self, epoch):
        self.model.eval()
        self.logger.info("\n***** Running testing *****")
        self.logger.info("  Num instance = %d", len(self.test_data) * self.args.batch_size)
        self.logger.info("  Batch size = %d", self.args.batch_size)

        if self.args.load_path is not None:  # load model from load_path
            self.logger.info("Loading best model from {}".format(self.args.load_path))
            self.model.load_state_dict(torch.load(self.args.load_path))
            self.logger.info("Load best model successful!")

        true_labels, pred_labels = [], []

        with torch.no_grad():
            with tqdm(total=len(self.test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Testing")
                total_loss = 0
                for batch in self.test_data:
                    batch = (tup.to(self.args.device) if isinstance(tup, torch.Tensor) else tup for tup in
                             batch)  # to cpu/cuda device
                    (loss, logits), labels = self._step(batch, mode="test")  # logits: batch, 3
                    total_loss += loss.detach().cpu().item()

                    preds = logits.argmax(-1)
                    true_labels.extend(labels.view(-1).detach().cpu().tolist())
                    pred_labels.extend(preds.view(-1).detach().cpu().tolist())

                    pbar.update()
                # evaluate done
                pbar.close()

                acc, recall, precision, micro_f1 = get_four_metrics(true_labels, pred_labels, type='weighted')
                result = {
                    'eval_accuracy': acc,
                    'precision': precision,
                    'recall': recall,
                    'f_score': micro_f1,
                    'global_step': epoch,
                    'loss': total_loss
                }
                # result = classification_report(y_true=true_labels, y_pred=pred_labels, digits=4)
                self.logger.info("***** Test Eval results *****")
                for key in sorted(result.keys()):
                    self.logger.info("  %s = %s", key, str(result[key]))

                if self.writer:
                    self.writer.add_scalar(tag='test_acc', scalar_value=acc)  # tensorbordx
                    self.writer.add_scalar(tag='test_f1', scalar_value=micro_f1)  # tensorbordx
                    self.writer.add_scalar(tag='test_loss',
                                           scalar_value=total_loss / len(self.test_data))  # tensorbordx
                total_loss = 0
                # self.logger.info("Epoch {}/{}, best test f1: {}, best epoch: {}, current test f1 score: {}, acc: {}." \
                #                  .format(epoch, self.args.num_epochs, self.best_test_metric, self.best_test_epoch,
                #                          micro_f1, acc))
                # if micro_f1 >= self.best_test_metric:  # this epoch get best performance
                #     self.best_test_metric = micro_f1
                #     self.best_test_epoch = epoch

        self.model.train()

    def _step(self, batch, mode="train"):
        input_ids, input_mask, segment_ids, img_mask, labels, images = batch

        outputs = self.model(input_ids=input_ids, attention_mask=input_mask, token_type_ids=segment_ids,
                             labels=labels, images=images)
        return outputs, labels

    def multiModal_before_train(self):

        parameters = []
        # other parameter
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'vision' not in name and 'text' not in name and not name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        # bert lr
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'text' in name:
                params['params'].append(param)
        parameters.append(params)

        # vit lr
        params = {'lr': self.args.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if 'vision' in name:
                params['params'].append(param)
        parameters.append(params)

        # fc lr
        params = {'lr': 5e-2, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if name.startswith('fc'):
                params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(parameters)

        self.model.to(self.args.device)

        self.scheduler = get_linear_schedule_with_warmup(optimizer=self.optimizer,
                                                         num_warmup_steps=self.args.warmup_ratio * self.train_num_steps,
                                                         num_training_steps=self.train_num_steps)
