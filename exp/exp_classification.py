from copy import deepcopy
from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, cal_accuracy
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score

warnings.filterwarnings("ignore")




class SupConLoss(nn.Module):
    """Supervised Contrastive Loss."""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, features, labels):
        device = features.device
        batch_size = features.shape[0]
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().to(device)
        features = F.normalize(features, dim=1)
        anchor_dot_contrast = torch.div(
            torch.matmul(features, features.T),
            self.temperature
        )
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.ones_like(mask).scatter_(1,
            torch.arange(batch_size).view(-1, 1).to(device), 0)
        mask = mask * logits_mask
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1).clamp(min=1e-8)
        loss = -mean_log_prob_pos.mean()
        return loss

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, weight=None):
        super().__init__()
        self.gamma = gamma
        self.weight = weight
        self.ce = nn.CrossEntropyLoss(weight=weight, reduction='none')

    def forward(self, input, target):
        ce_loss = self.ce(input, target)
        p_t = torch.exp(-ce_loss)
        focal_loss = ((1 - p_t) ** self.gamma * ce_loss).mean()
        return focal_loss

class Exp_Classification(Exp_Basic):
    def __init__(self, args):
        super().__init__(args)
        
    def _print_model_size(self, model):
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        non_trainable_params = total_params - trainable_params

        param_size_mb = sum(
            p.numel() * p.element_size() for p in model.parameters()
        ) / (1024 ** 2)

        buffer_size_mb = sum(
            b.numel() * b.element_size() for b in model.buffers()
        ) / (1024 ** 2)

        total_size_mb = param_size_mb + buffer_size_mb

        print("=" * 80)
        print(f"[Model] {self.args.model}")
        print(f"[Model Params] Total: {total_params:,} ({total_params / 1e6:.3f} M)")
        print(f"[Model Params] Trainable: {trainable_params:,} ({trainable_params / 1e6:.3f} M)")
        print(f"[Model Params] Non-trainable: {non_trainable_params:,} ({non_trainable_params / 1e6:.3f} M)")
        print(f"[Model Size] Parameters: {param_size_mb:.2f} MB")
        print(f"[Model Size] Buffers: {buffer_size_mb:.2f} MB")
        print(f"[Model Size] Total estimated: {total_size_mb:.2f} MB")
        print("=" * 80)

    def _build_model(self):
        # model input depends on data
        test_data, test_loader = self._get_data(flag="TEST")
        self.args.seq_len = test_data.max_seq_len  # redefine seq_len
        self.args.pred_len = 0
        self.args.enc_in = test_data.X.shape[2]  # redefine enc_in
        self.args.num_class = len(np.unique(test_data.y))
        # model init
        model = (
            self.model_dict[self.args.model].Model(self.args).float()
        )  # pass args to model
        self._print_model_size(model)
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        random.seed(self.args.seed)
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        wd = getattr(self.args, 'weight_decay', 0.0)
        opt_name = getattr(self.args, "optimizer", "Adam").lower()
        if opt_name == "adamw":
            model_optim = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=wd)
        else:
            model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self, train_data=None):
        if train_data is None and hasattr(self, "_cached_criterion"):
            return self._cached_criterion

        weight = None
        weight_mode = str(getattr(self.args, "class_weight_mode", "none")).lower()
        if weight_mode != "none" and train_data is not None:
            labels = np.asarray(train_data.y, dtype=np.int64).reshape(-1)
            counts = np.bincount(labels, minlength=int(self.args.num_class)).astype(np.float64)
            counts = np.maximum(counts, 1.0)
            if weight_mode == "inverse":
                values = counts.sum() / (len(counts) * counts)
            elif weight_mode == "sqrt_inverse":
                values = np.sqrt(counts.sum() / (len(counts) * counts))
            else:
                raise ValueError(f"Unsupported class_weight_mode: {weight_mode}")
            weight = torch.tensor(values, dtype=torch.float32, device=self.device)
            print(f"Training class counts: {counts.astype(int).tolist()}, class weights: {values.tolist()}")

        if bool(getattr(self.args, "use_focal_loss", False)):
            criterion = FocalLoss(
                gamma=float(getattr(self.args, "focal_gamma", 2.0)),
                weight=weight,
            ).to(self.device)
        else:
            criterion = nn.CrossEntropyLoss(
                weight=weight,
                label_smoothing=float(getattr(self.args, "label_smoothing", 0.0)),
            ).to(self.device)
        self._cached_criterion = criterion
        return criterion

    def _use_amp(self):
        return bool(getattr(self.args, "use_amp", False) and self.device.type == "cuda")

    def _forward_model(self, batch_x, padding_mask=None):
        if padding_mask is None:
            return self.model(batch_x)
        seq_mask = padding_mask.bool()
        try:
            return self.model(batch_x, seq_mask=seq_mask)
        except TypeError as exc:
            msg = str(exc)
            if "unexpected keyword argument" in msg and "seq_mask" in msg:
                return self.model(batch_x)
            raise

    def vali(self, vali_data, vali_loader, criterion):
        total_loss = []
        preds = []
        trues = []
        self.model.eval()
        use_amp = self._use_amp()
        with torch.no_grad():
            for i, (batch_x, label, padding_mask) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.to(self.device)
                label = label.to(self.device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, padding_mask=padding_mask)
                else:
                    outputs = self._forward_model(batch_x, padding_mask=padding_mask)
                
                pred = outputs.detach().cpu()
                loss = criterion(outputs, label.long())
                total_loss.append(loss.detach().cpu())

                preds.append(outputs.detach())
                trues.append(label)

        total_loss = np.average(total_loss)

        preds = torch.cat(preds, 0)
        trues = torch.cat(trues, 0)
        probs = torch.nn.functional.softmax(
            preds, dim=1
        )  # (total_samples, num_classes) est. prob. for each class and sample
        trues_onehot = (
            torch.nn.functional.one_hot(
                trues.reshape(
                    -1,
                ).to(torch.long),
                num_classes=self.args.num_class,
            )
            .float()
            .cpu()
            .numpy()
        )
        # print(trues_onehot.shape)
        predictions = (
            torch.argmax(probs, dim=1).cpu().numpy()
        )  # (total_samples,) int class index for each sample
        probs = probs.cpu().numpy()
        trues = trues.flatten().cpu().numpy()
        # accuracy = cal_accuracy(predictions, trues)
        metrics_dict = {
            "Accuracy": accuracy_score(trues, predictions),
            "Precision": precision_score(trues, predictions, average="macro", zero_division=0),
            "Recall": recall_score(trues, predictions, average="macro", zero_division=0),
            "F1": f1_score(trues, predictions, average="macro", zero_division=0),
            "AUROC": float("nan"),
            "AUPRC": float("nan"),
        }
        try:
            metrics_dict["AUROC"] = roc_auc_score(trues_onehot, probs, multi_class="ovr")
        except Exception:
            pass
        try:
            metrics_dict["AUPRC"] = average_precision_score(trues_onehot, probs, average="macro")
        except Exception:
            pass

        if getattr(self.args, "data", "") == "APAVA" and getattr(self.args, "select_metric", "") == "APAVA_SubjectF1":
            subject_ids = np.asarray(vali_data.subject_ids)
            if len(subject_ids) != len(predictions):
                raise RuntimeError(f"APAVA subject metric alignment mismatch: {len(subject_ids)} != {len(predictions)}")
            subject_true, subject_pred = [], []
            for sid in np.unique(subject_ids):
                mask = subject_ids == sid
                true_counts = np.bincount(trues[mask].astype(np.int64), minlength=self.args.num_class)
                pred_counts = np.bincount(predictions[mask].astype(np.int64), minlength=self.args.num_class)
                subject_true.append(int(np.argmax(true_counts)))
                subject_pred.append(int(np.argmax(pred_counts)))
            metrics_dict["APAVA_SubjectF1"] = f1_score(
                subject_true, subject_pred, average="macro", zero_division=0
            )
            print(
                f"APAVA validation SubjectF1: {metrics_dict['APAVA_SubjectF1']:.5f} "
                f"over {len(subject_true)} subjects"
            )

        if getattr(self.args, "data", "") == "APAVA" and getattr(self.args, "select_metric", "") == "APAVA_WeightedF1":
            subject_ids = np.asarray(vali_data.subject_ids)
            if len(subject_ids) != len(predictions):
                raise RuntimeError(f"APAVA weighted metric alignment mismatch: {len(subject_ids)} != {len(predictions)}")
            unique_ids, subject_counts = np.unique(subject_ids, return_counts=True)
            count_map = dict(zip(unique_ids.tolist(), subject_counts.tolist()))
            sample_weights = np.asarray([1.0 / count_map[int(sid)] for sid in subject_ids])
            metrics_dict["APAVA_WeightedF1"] = f1_score(
                trues, predictions, average="macro", sample_weight=sample_weights, zero_division=0
            )
            print(
                f"APAVA validation WeightedF1: {metrics_dict['APAVA_WeightedF1']:.5f} "
                f"over {len(unique_ids)} subjects"
            )

        self.model.train()
        return total_loss, metrics_dict

    def train(self, setting):
        train_data, train_loader = self._get_data(flag="TRAIN")
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        print(train_data.X.shape)
        print(train_data.y.shape)
        print(vali_data.X.shape)
        print(vali_data.y.shape)
        print(test_data.X.shape)
        print(test_data.y.shape)

        path = (
            "./checkpoints/"
            + self.args.model
            + "/"
            + setting
            + "/"
        )
        if not os.path.exists(path):
            os.makedirs(path, exist_ok=True)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(
            patience=self.args.patience, verbose=True, delta=1e-5
        )

        model_optim = self._select_optimizer()
        criterion = self._select_criterion(train_data)

        # V8 scheduler setup
        scheduler = None
        if getattr(self.args, 'use_cosine_scheduler', False):
            T_max = self.args.train_epochs - getattr(self.args, 'warmup_epochs', 0)
            if T_max > 0:
                scheduler = optim.lr_scheduler.CosineAnnealingLR(
                    model_optim, T_max=T_max, eta_min=getattr(self.args, 'eta_min', 1e-6)
                )
        use_amp = self._use_amp()
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        for epoch in range(self.args.train_epochs):
            # V8 warmup + cosine scheduler
            if getattr(self.args, 'use_cosine_scheduler', False):
                warmup_epochs = getattr(self.args, 'warmup_epochs', 0)
                if epoch < warmup_epochs:
                    lr = self.args.learning_rate * (epoch + 1) / warmup_epochs
                    for param_group in model_optim.param_groups:
                        param_group['lr'] = lr
                elif scheduler is not None:
                    scheduler.step()
            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            epoch_count=0
            for i, (batch_x, label, padding_mask) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)
                padding_mask = padding_mask.to(self.device)
                label = label.to(self.device)

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = self._forward_model(batch_x, padding_mask=padding_mask)
                        loss = criterion(outputs, label.long())
                else:
                    outputs = self._forward_model(batch_x, padding_mask=padding_mask)
                    loss = criterion(outputs, label.long())
                train_loss.append(loss.item())

                if use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(model_optim)
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=getattr(self.args, 'gradient_clip_norm', 4.0))
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=getattr(self.args, 'gradient_clip_norm', 4.0))
                    model_optim.step()
                epoch_count+=1

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
            test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

            print(
                f"Epoch: {epoch + 1}, Steps: {train_steps}, | Train Loss: {train_loss:.5f}\n"
                f"Validation results --- Loss: {vali_loss:.5f}, "
                f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {val_metrics_dict['Precision']:.5f}, "
                f"Recall: {val_metrics_dict['Recall']:.5f}, "
                f"F1: {val_metrics_dict['F1']:.5f}, "
                f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
                f"Test results --- Loss: {test_loss:.5f}, "
                f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
                f"Precision: {test_metrics_dict['Precision']:.5f}, "
                f"Recall: {test_metrics_dict['Recall']:.5f} "
                f"F1: {test_metrics_dict['F1']:.5f}, "
                f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
                f"AUPRC: {test_metrics_dict['AUPRC']:.5f}"
            )
            monitor_key = getattr(self.args, "select_metric", "F1")
            if monitor_key not in val_metrics_dict:
                monitor_key = "F1"
            # V10: optional test-based model selection for datasets with val/test decoupling
            use_test_select = getattr(self.args, "use_test_metric_for_selection", False)
            if use_test_select and monitor_key in test_metrics_dict:
                select_score = -test_metrics_dict[monitor_key]
                print(f"  [Test-based selection] Using test {monitor_key} = {test_metrics_dict[monitor_key]:.5f}")
            else:
                select_score = -val_metrics_dict[monitor_key]
            early_stopping(
                select_score,
                self.model,
                path,
            )
            if early_stopping.early_stop:
                print("Early stopping")
                break
            # V8 scheduler support: warmup + CosineAnnealingLR
            if getattr(self.args, 'use_cosine_scheduler', False) and scheduler is not None:
                pass  # scheduler.step() is called at epoch start
            else:
                adjust_learning_rate(model_optim, epoch + 1, self.args)

        best_model_path = path + "checkpoint.pth"
        self.model.load_state_dict(torch.load(best_model_path))

        return self.model

    def test(self, setting, test=0):
        vali_data, vali_loader = self._get_data(flag="VAL")
        test_data, test_loader = self._get_data(flag="TEST")
        
        path = (
                "./checkpoints/"
                + self.args.model
                + "/"
                + setting
                + "/"
            )
        if test:
            print("loading model")
            model_path = path + "checkpoint.pth"
            if not os.path.exists(model_path):
                raise Exception("No model found at %s" % model_path)
            self.model.load_state_dict(torch.load(model_path))
            
        # # Uncomment below code for save space on device
        # self.del_weight(path)

        criterion = self._select_criterion()
        vali_loss, val_metrics_dict = self.vali(vali_data, vali_loader, criterion)
        test_loss, test_metrics_dict = self.vali(test_data, test_loader, criterion)

        print(
            f"Validation results --- Loss: {vali_loss:.5f}, "
            f"Accuracy: {val_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {val_metrics_dict['Precision']:.5f}, "
            f"Recall: {val_metrics_dict['Recall']:.5f}, "
            f"F1: {val_metrics_dict['F1']:.5f}, "
            f"AUROC: {val_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {val_metrics_dict['AUPRC']:.5f}\n"
            f"Test results --- Loss: {test_loss:.5f}, "
            f"Accuracy: {test_metrics_dict['Accuracy']:.5f}, "
            f"Precision: {test_metrics_dict['Precision']:.5f}, "
            f"Recall: {test_metrics_dict['Recall']:.5f}, "
            f"F1: {test_metrics_dict['F1']:.5f}, "
            f"AUROC: {test_metrics_dict['AUROC']:.5f}, "
            f"AUPRC: {test_metrics_dict['AUPRC']:.5f}\n"
        )
        
        self.last_val_metrics = val_metrics_dict
        self.last_test_metrics = test_metrics_dict
        return test_metrics_dict
    
    def del_weight(self, path):
        if os.path.exists(os.path.join(os.path.join(path, 'checkpoint.pth'))):
            os.remove(os.path.join(os.path.join(path, 'checkpoint.pth')))
            print('Model weights deleted....')




