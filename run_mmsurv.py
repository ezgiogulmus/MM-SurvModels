import argparse
import os
import sys 
import json
from timeit import default_timer as timer
from mmsurv.main import run

def setup_argparse():
	### Data 
	parser = argparse.ArgumentParser(description='Configurations for Survival Analysis on TCGA Data.')
	parser.add_argument('--run_name',      type=str, default='run')
	parser.add_argument('--data_name',   type=str, default=None)
	parser.add_argument('--feats_dir',   type=str, default=None)

	parser.add_argument('--dataset_dir', type=str, default="./datasets_csv")
	parser.add_argument('--results_dir', type=str, default='./results', help='Results directory (Default: ./results)')
	parser.add_argument('--split_dir', type=str, default="./splits", help='Split directory (Default: ./splits)')

	parser.add_argument('--run_config_file',      type=str, default=None)
	
	### Experiment
	parser.add_argument('--seed', 			 type=int, default=1, help='Random seed for reproducible experiment (default: 1)')
	parser.add_argument('--k', 			     type=int, default=5, help='Number of folds (default: 5)')
	parser.add_argument('--k_start',		 type=int, default=-1, help='Start fold (Default: -1, last fold)')
	parser.add_argument('--k_end',			 type=int, default=-1, help='End fold (Default: -1, first fold)')
	parser.add_argument('--log_data',        action='store_true', default=True, help='Log data using tensorboard')
	parser.add_argument('--overwrite',     	 action='store_true', default=False, help='Whether or not to overwrite experiments (if already ran)')

	### Model Parameters.
	parser.add_argument('--omics', default=None)
	parser.add_argument('--selected_features',     	 action='store_false', default=True)
	parser.add_argument('--n_classes', type=int, default=4)

	parser.add_argument('--model_type',      type=str, choices=['snn', 'deepset', 'amil', 'mi_fcn', 'mcat', "motcat", "porpmmf", "porpamil"], default='mcat', help='Type of model (Default: mcat)')
	parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'], default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
	parser.add_argument('--fusion',          type=str, choices=['None', 'concat', 'bilinear'], default='concat', help='Type of fusion. (Default: concat).')
	parser.add_argument('--apply_sig',		 action='store_true', default=False, help='Use genomic features as signature embeddings.')
	parser.add_argument('--apply_sigfeats',  action='store_true', default=False, help='Use genomic features as tabular features.')
	parser.add_argument('--drop_out',        action='store_true', default=True, help='Enable dropout (p=0.25)')
	parser.add_argument('--model_size_wsi',  type=str, default='small', help='Network size of AMIL model')
	parser.add_argument('--model_size_omic', type=str, default='small', help='Network size of SNN model')

	# MOTCAT Parameters
	parser.add_argument('--bs_micro', type=int, default=256, help='The Size of Micro-batch (Default: 256)')  # new
	parser.add_argument('--ot_impl', type=str, default='pot-uot-l2', help='impl of ot (default: pot-uot-l2)')  # new
	parser.add_argument('--ot_reg', type=float, default=0.1, help='epsilon of OT (default: 0.1)')
	parser.add_argument('--ot_tau', type=float, default=0.5, help='tau of UOT (default: 0.5)')

	# PORPOISE Parameters
	parser.add_argument('--gate_path', action='store_true', default=False)
	parser.add_argument('--gate_omic', action='store_true', default=False)
	parser.add_argument('--scale_dim1', type=int, default=8)
	parser.add_argument('--scale_dim2', type=int, default=8)
	parser.add_argument('--skip', action='store_true', default=False)
	parser.add_argument('--dropinput', type=float, default=0.0)
	parser.add_argument('--use_mlp', action='store_true', default=False)

	### Optimizer Parameters + Survival Loss Function
	parser.add_argument('--opt',             type=str, choices = ['adam', 'sgd'], default='adam')
	parser.add_argument('--batch_size',      type=int, default=1, help='Batch Size (Default: 1, due to varying bag sizes)')
	parser.add_argument('--gc',              type=int, default=32, help='Gradient Accumulation Step.')
	parser.add_argument('--max_epochs',      type=int, default=20, help='Maximum number of epochs to train (default: 20)')
	parser.add_argument('--lr',				 type=float, default=2e-4, help='Learning rate (default: 0.0001)')
	parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv', 'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: ce)')
	parser.add_argument('--bag_weight',      type=float, default=0.7, help='clam: weight coefficient for bag-level loss (default: 0.7)')
	parser.add_argument('--reg', 			 type=float, default=1e-5, help='L2-regularization weight decay (default: 1e-5)')
	parser.add_argument('--alpha_surv',      type=float, default=0.0, help='How much to weigh uncensored patients')
	parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'], default='None', help='Which network submodules to apply L1-Regularization (default: None)')
	parser.add_argument('--lambda_reg',      type=float, default=1e-4, help='L1-Regularization Strength (Default 1e-4)')
	parser.add_argument('--weighted_sample', action='store_true', default=True, help='Enable weighted sampling')
	parser.add_argument('--early_stopping',  type=int, default=20, help='Enable early stopping')

	args = parser.parse_args()
	return args

if __name__ == "__main__" and (__package__ is None or __package__ == ''):
	script_dir = os.path.dirname(os.path.abspath(__file__))
	sys.path.append(os.path.dirname(script_dir))
	args = setup_argparse()
	if args.run_config_file:
		new_run_name = args.run_name
		results_dir = args.results_dir
		feats_dir = args.feats_dir
		cv_fold = args.k
		max_epochs = args.max_epochs
		with open(args.run_config_file, "r") as f:
			config = json.load(f)
		
		parser = argparse.ArgumentParser()
		parser.add_argument("--run_config_file")
		for k, v in config.items():
			if k != "run_config_file":
				parser.add_argument('--' + k, default=v, type=type(v))
		args = parser.parse_args()
		args.run_name = new_run_name
		args.feats_dir = feats_dir
		args.results_dir = results_dir
		args.k = cv_fold
		args.max_epochs = max_epochs
		args.split_dir = args.split_dir.split("/")[-1]
		start = timer()
		run(args)
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	else:
		start = timer()
		run(args)
		end = timer()
		print("finished!")
		print("end script")
		print('Script Time: %f seconds' % (end - start))
	