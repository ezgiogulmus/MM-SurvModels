from argparse import Namespace
import os
import numpy as np
import random
from sksurv.metrics import concordance_index_censored
import torch

from datasets.dataset_generic import save_splits
from models.model_genomic import SNN
from models.model_set_mil import MIL_Sum_FC_surv, MIL_Attention_FC_surv
from models.model_coattn import MCAT_Surv
from models.model_motcat import MOTCAT_Surv
from models.model_porpoise import PorpoiseAMIL, PorpoiseMMF
from utils.utils import *

device=torch.device("cuda" if torch.cuda.is_available() else "cpu") 


class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, warmup=5, patience=15, stop_epoch=20, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 20
            stop_epoch (int): Earliest epoch possible for stopping
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
        """
        self.warmup = warmup
        self.patience = patience
        self.stop_epoch = stop_epoch
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, epoch, val_loss, model, ckpt_name = 'checkpoint.pt'):

        score = -val_loss

        if epoch < self.warmup:
            pass
        elif self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
        elif score < self.best_score:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience and epoch > self.stop_epoch:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, ckpt_name)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, ckpt_name):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        torch.save(model.state_dict(), ckpt_name)
        self.val_loss_min = val_loss


def train(datasets: tuple, cur: int, args: Namespace):
    """   
        train for a single fold
    """
    print('\nTraining Fold {}!'.format(cur))
    writer_dir = os.path.join(args.results_dir, str(cur))
    if not os.path.isdir(writer_dir):
        os.mkdir(writer_dir)

    if args.log_data:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(writer_dir, flush_secs=15)

    else:
        writer = None

    print('\nInit train/val/test splits...', end=' ')
    train_split, val_split, test_split = datasets
    train_survival = np.array(list(zip(train_split.slide_data["censorship"].values, train_split.slide_data["survival_months"].values)), dtype=[('censorship', bool), ('time', np.float64)])
    max_surv_limit = int(np.min([train_split.slide_data["survival_months"].max(), val_split.slide_data["survival_months"].max(), test_split.slide_data["survival_months"].max()]))
    save_splits(datasets, ['train', 'val', 'test'], os.path.join(args.results_dir, 'splits_{}.csv'.format(cur)))
    print('Done!')
    print("Training on {} samples".format(len(train_split)))
    print("Validating on {} samples".format(len(val_split)))
    print("Testing on {} samples".format(len(test_split)))

    print('\nInit loss function...', end=' ')
    if args.bag_loss == 'ce_surv':
        loss_fn = CrossEntropySurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'nll_surv':
        loss_fn = NLLSurvLoss(alpha=args.alpha_surv)
    elif args.bag_loss == 'cox_surv':
        loss_fn = CoxSurvLoss()
    else:
        raise NotImplementedError

    if args.reg_type == 'omic':
        reg_fn = l1_reg_all
    elif args.reg_type == 'pathomic':
        reg_fn = l1_reg_modules
    else:
        reg_fn = None

    print('Done!')
    
    print('\nInit Model...', end=' ')
    args.fusion = None if args.fusion == 'None' else args.fusion
    args.omic_sizes = train_split.omic_sizes

    if args.model_type =='snn':
        model_dict = {'omic_input_dim': args.omic_input_dim, 'model_size_omic': args.model_size_omic, 'n_classes': args.n_classes}
        model = SNN(**model_dict)
    elif args.model_type == 'deepset':
        model_dict = {"path_input_dim": args.path_input_dim, 'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Sum_FC_surv(**model_dict)
    elif args.model_type =='amil':
        model_dict = {'path_input_dim': args.path_input_dim, 'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes}
        model = MIL_Attention_FC_surv
    elif args.model_type == 'mcat':
        model_dict = {"path_input_dim": args.path_input_dim, 'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MCAT_Surv(**model_dict)
    elif args.model_type == 'motcat':
        model_dict = {'path_input_dim': args.path_input_dim, 'ot_reg': args.ot_reg, 'ot_tau': args.ot_tau, 'ot_impl': args.ot_impl,'fusion': args.fusion, 'omic_sizes': args.omic_sizes, 'n_classes': args.n_classes}
        model = MOTCAT_Surv(**model_dict)
    elif args.model_type == 'porpoise_mmf':
        model_dict = {'path_input_dim': args.path_input_dim, 'omic_input_dim': args.omic_input_dim, 'fusion': args.fusion, 'n_classes': args.n_classes, 
        'gate_path': args.gate_path, 'gate_omic': args.gate_omic, 'scale_dim1': args.scale_dim1, 'scale_dim2': args.scale_dim2, 
        'skip': args.skip, 'dropinput': args.dropinput, 'path_input_dim': args.path_input_dim, 'use_mlp': args.use_mlp,
        }
        model = PorpoiseMMF(**model_dict)
    elif args.model_type == 'porpoise_amil':
        model_dict = {'path_input_dim': args.path_input_dim, 'n_classes': args.n_classes}
        model = PorpoiseAMIL(**model_dict)
    else:
        raise NotImplementedError
    
    if hasattr(model, "relocate"):
        model.relocate()
    else:
        model = model.to(torch.device('cuda'))
    print('Done!')
    if cur == 0:
        print_network(model)

    print('\nInit optimizer ...', end=' ')
    optimizer = get_optim(model, args)
    print('Done!')
    
    print('\nInit Loaders...', end=' ')
    train_loader = get_split_loader(train_split, training=True, weighted = args.weighted_sample, mode=args.mode, batch_size=args.batch_size)
    val_loader = get_split_loader(val_split, mode=args.mode, batch_size=args.batch_size)
    test_loader = get_split_loader(test_split, batch_size=args.batch_size)
    print('Done!')

    print('\nSetup EarlyStopping...', end=' ')
    if args.early_stopping:
        early_stopping = EarlyStopping(warmup=0, patience=10, stop_epoch=20, verbose = True)
    else:
        early_stopping = None
    print('Done!\n\n')

    for epoch in range(args.max_epochs):
        loop_survival(cur, epoch, model, train_loader, loss_fn, reg_fn, args.lambda_reg, writer, optimizer, args.gc, model_type=args.model_type, bs_micro=args.bs_micro)
        stop = loop_survival(cur, epoch, model, val_loader, loss_fn, reg_fn, args.lambda_reg, writer, model_type=args.model_type, training=False, results_dir=args.results_dir, early_stopping=early_stopping, bs_micro=args.bs_micro)
        if stop:
            break
    
    torch.save(model.state_dict(), os.path.join(args.results_dir, "s_{}_checkpoint.pt".format(cur)))
    if os.path.isfile(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))):
        model.load_state_dict(torch.load(os.path.join(args.results_dir, "s_{}_minloss_checkpoint.pt".format(cur))))
    
    results_val_dict, val_cindex = loop_survival(cur, epoch, model, val_loader, loss_fn, reg_fn, args.lambda_reg, model_type=args.model_type, training=False, return_summary=True, bs_micro=args.bs_micro)
    results_test_dict, test_cindex = loop_survival(cur, epoch, model, test_loader, loss_fn, reg_fn, args.lambda_reg, model_type=args.model_type, training=False, return_summary=True, bs_micro=args.bs_micro)

    print('Val c-Index: {:.4f} | Test c-Index: {:.4f}'.format(val_cindex, test_cindex))
    log = {'val_cindex': val_cindex, 'test_cindex': test_cindex}
    if writer:
        for k, v in log.items():
            writer.add_scalar(k, v)
        writer.close()
    return log, results_val_dict, results_test_dict

def loop_survival(
        cur, epoch, model, loader, 
        loss_fn=None, reg_fn=None, lambda_reg=0., writer=None, optimizer=None, gc=16, 
        model_type="coattn", training=True, results_dir=None, 
        early_stopping=None, return_summary=False, bs_micro=256
    ): 
    model.train() if training else model.eval()
    split_name = "Train" if training else "Validation"
    loss_surv, running_loss = 0., 0.
    if return_summary:
        patient_results = {}
        slide_ids = loader.dataset.slide_data['slide_id']

    all_risk_scores = np.zeros((len(loader)))
    all_censorships = np.zeros((len(loader)))
    all_event_times = np.zeros((len(loader)))

    for batch_idx, data in enumerate(loader):
        
        if model_type == "motcat":
            data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c = list(map(lambda x:x.to(device), data))
            loss = 0.
            all_risk = 0.
            cnt = 0
            index_chunk_list = split_chunk_list(data_WSI, bs_micro)
            for tindex in index_chunk_list:
                wsi_mb = torch.index_select(data_WSI, dim=0, index=torch.LongTensor(tindex).to(data_WSI.device)).cuda()
                with torch.set_grad_enabled(training):
                    hazards, S, _, _  = model(x_path=wsi_mb, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
                
                loss_micro = loss_fn(hazards=hazards, S=S, Y=label, c=c)
                
                loss += loss_micro
                all_risk += -torch.sum(S, dim=1).detach().cpu().numpy().item()
                cnt+=1
            loss = loss / cnt
            risk = all_risk / cnt
        else:
            if model_type == "mcat":
                data_WSI, data_omic1, data_omic2, data_omic3, data_omic4, data_omic5, data_omic6, label, event_time, c = list(map(lambda x:x.to(device), data))
                with torch.set_grad_enabled(training):
                    hazards, S, Y_hat, A  = model(x_path=data_WSI, x_omic1=data_omic1, x_omic2=data_omic2, x_omic3=data_omic3, x_omic4=data_omic4, x_omic5=data_omic5, x_omic6=data_omic6)
            else:
                data_WSI, data_omic, label, event_time, c = list(map(lambda x:x.to(device), data))
                with torch.set_grad_enabled(training):
                    hazards, S = model(x_path=data_WSI, x_omic=data_omic)
                
            loss = loss_fn(hazards=hazards, S=S, Y=label, c=c)
            risk = -torch.sum(S, dim=1).detach().cpu().numpy()
        
        loss_value = loss.item()

        if reg_fn is None:
            loss_reg = 0
        else:
            loss_reg = reg_fn(model) * lambda_reg

        all_risk_scores[batch_idx] = risk
        all_censorships[batch_idx] = c.item()
        all_event_times[batch_idx] = event_time

        if return_summary:
            slide_id = slide_ids.iloc[batch_idx]
            patient_results.update({
                slide_id: {
                    'slide_id': np.array(slide_id), 
                    'risk': risk, 
                    'disc_label': label.item(), 
                    'survival': event_time.cpu().numpy(), 
                    'censorship': c.cpu().numpy()
            }})

        loss_surv += loss_value
        running_loss += loss_value + loss_reg

        if (batch_idx + 1) % 100 == 0:
            print('batch {}, loss: {:.4f}, label: {}, event_time: {:.4f}, risk: {:.4f}, bag_size: {}'.format(batch_idx, loss_value + loss_reg, label.item(), float(event_time), float(risk), data_WSI.size(0)))
        
        if training:
            # backward pass
            loss = loss / gc + loss_reg
            loss.backward()

            if (batch_idx + 1) % gc == 0: 
                optimizer.step()
                optimizer.zero_grad()

    # calculate loss and error for epoch
    loss_surv /= len(loader)
    running_loss /= len(loader)

    c_index = concordance_index_censored((1-all_censorships).astype(bool), all_event_times, all_risk_scores, tied_tol=1e-08)[0]
    if return_summary:
        return patient_results, c_index
    print('{} | epoch: {}, loss_surv: {:.4f}, loss: {:.4f}, train_c_index: {:.4f}\n'.format(split_name, epoch, loss_surv, running_loss, c_index))
    
    if writer:
        writer.add_scalar(f'{split_name}/loss_surv', loss_surv, epoch)
        writer.add_scalar(f'{split_name}/loss', running_loss, epoch)
        writer.add_scalar(f'{split_name}/c_index', c_index, epoch)

    if early_stopping:
        assert results_dir
        early_stopping(epoch, loss_surv, model, ckpt_name=os.path.join(results_dir, "s_{}_minloss_checkpoint.pt".format(cur)))
        
        if early_stopping.early_stop:
            print("Early stopping")
            return True
    return False


def split_chunk_list(data, batch_size):
    numGroup = data.shape[0] // batch_size + 1
    feat_index = list(range(data.shape[0]))
    random.shuffle(feat_index)
    index_chunk_list = np.array_split(np.array(feat_index), numGroup)
    index_chunk_list = [sst.tolist() for sst in index_chunk_list]
    return index_chunk_list

