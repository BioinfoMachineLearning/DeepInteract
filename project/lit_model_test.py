import logging
import os
from pathlib import Path

import pytorch_lightning as pl
import torch.nn as nn
from pytorch_lightning.plugins import DDPPlugin

from project.datasets.PICP.picp_dgl_data_module import PICPDGLDataModule
from project.utils.deepinteract_constants import NODE_COUNT_LIMIT, RESIDUE_COUNT_LIMIT
from project.utils.deepinteract_modules import LitGINI
from project.utils.deepinteract_utils import collect_args, process_args, construct_pl_logger


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/BioinfoMachineLearning/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


def main(args):
    # -----------
    # Test Args
    # -----------
    test_batch_size = 1  # Enforce batch_size=1 when testing on the large complexes in DB5-Plus
    self_loops = True  # Enforce self-loops in graphs to be expected

    # -----------
    # Data
    # -----------
    # Load protein interface contact prediction (PICP) data module
    picp_data_module = PICPDGLDataModule(casp_capri_data_dir=args.casp_capri_data_dir,
                                         db5_data_dir=args.db5_data_dir,
                                         dips_data_dir=args.dips_data_dir,
                                         batch_size=test_batch_size,
                                         num_dataloader_workers=args.num_workers,
                                         knn=args.knn,
                                         self_loops=args.self_loops,
                                         pn_ratio=args.pn_ratio,
                                         casp_capri_percent_to_use=args.casp_capri_percent_to_use,
                                         db5_percent_to_use=args.db5_percent_to_use,
                                         dips_percent_to_use=args.dips_percent_to_use,
                                         training_with_db5=args.training_with_db5,
                                         testing_with_casp_capri=args.testing_with_casp_capri,
                                         process_complexes=args.process_complexes,
                                         input_indep=args.input_indep)
    picp_data_module.setup()

    # ------------
    # Model
    # ------------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)
    use_wandb_logger = args.logger_name.lower() == 'wandb'  # Determine whether the user requested to use WandB

    # Pick model and supply it with a dictionary of arguments
    # Baseline Model - Geometry-Focused Inter-Graph Node Interaction (GINI)
    model = LitGINI(num_node_input_feats=picp_data_module.dips_test.num_node_features,
                    num_edge_input_feats=picp_data_module.dips_test.num_edge_features,
                    gnn_activ_fn=nn.SiLU(),
                    num_classes=picp_data_module.dips_test.num_classes,
                    max_num_graph_nodes=NODE_COUNT_LIMIT,
                    max_num_residues=RESIDUE_COUNT_LIMIT,
                    testing_with_casp_capri=dict_args['testing_with_casp_capri'],
                    training_with_db5=dict_args['training_with_db5'],
                    pos_prob_threshold=0.5,
                    gnn_layer_type=dict_args['gnn_layer_type'],
                    num_gnn_layers=dict_args['num_gnn_layers'],
                    num_gnn_hidden_channels=dict_args['num_gnn_hidden_channels'],
                    num_gnn_attention_heads=dict_args['num_gnn_attention_heads'],
                    knn=dict_args['knn'],
                    interact_module_type=dict_args['interact_module_type'],
                    num_interact_layers=dict_args['num_interact_layers'],
                    num_interact_hidden_channels=dict_args['num_interact_hidden_channels'],
                    use_interact_attention=dict_args['use_interact_attention'],
                    num_interact_attention_heads=dict_args['num_interact_attention_heads'],
                    disable_geometric_mode=dict_args['disable_geometric_mode'],
                    num_epochs=dict_args['num_epochs'],
                    pn_ratio=dict_args['pn_ratio'],
                    dropout_rate=dict_args['dropout_rate'],
                    metric_to_track=dict_args['metric_to_track'],
                    weight_decay=dict_args['weight_decay'],
                    batch_size=test_batch_size,
                    lr=dict_args['lr'],
                    pad=dict_args['pad'],
                    viz_every_n_epochs=dict_args['viz_every_n_epochs'],
                    use_wandb_logger=use_wandb_logger,
                    weight_classes=dict_args['weight_classes'],
                    fine_tune=False,
                    ckpt_path=None)
    args.experiment_name = f'LitGINI-b{test_batch_size}-gl{args.num_gnn_layers}' \
                           f'-n{args.num_gnn_hidden_channels}' \
                           f'-e{args.num_gnn_hidden_channels}' \
                           f'-il{args.num_interact_layers}-i{args.num_interact_hidden_channels}' \
        if not args.experiment_name \
        else args.experiment_name
    litgini_template_ckpt_filename_metric_to_track = f'{args.metric_to_track}:.3f'
    template_ckpt_filename = 'LitGINI-{epoch:02d}-{' + litgini_template_ckpt_filename_metric_to_track + '}'

    # ------------
    # Checkpoint
    # ------------
    ckpt_path = os.path.join(args.ckpt_dir, args.ckpt_name)
    ckpt_provided = args.ckpt_name != ''
    assert ckpt_provided, 'A checkpoint filename must be provided'

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # ------------
    # Logger
    # ------------
    pl_logger = construct_pl_logger(args)  # Log everything to an external logger
    trainer.logger = pl_logger  # Assign specified logger (e.g. TensorBoardLogger) to Trainer instance

    # ------------
    # Restore
    # ------------
    # If using WandB, download checkpoint artifact from their servers if the checkpoint is not already stored locally
    if use_wandb_logger and args.ckpt_name != '' and not os.path.exists(ckpt_path):
        checkpoint_reference = f'{args.entity}/{args.project_name}/model-{args.run_id}:best'
        artifact = trainer.logger.experiment.use_artifact(checkpoint_reference, type='model')
        artifact_dir = artifact.download()
        model = model.load_from_checkpoint(Path(artifact_dir) / 'model.ckpt',
                                           use_wandb_logger=use_wandb_logger,
                                           batch_size=test_batch_size,
                                           lr=args.lr,
                                           weight_decay=args.weight_decay,
                                           dropout_rate=args.dropout_rate)
    else:
        assert ckpt_provided and os.path.exists(ckpt_path), 'A valid checkpoint filepath must be provided'
        model = model.load_from_checkpoint(ckpt_path,
                                           use_wandb_logger=use_wandb_logger,
                                           batch_size=test_batch_size,
                                           lr=args.lr,
                                           weight_decay=args.weight_decay,
                                           dropout_rate=args.dropout_rate)

    # -------------
    # Testing
    # -------------
    # Test the trained model with the provided data module
    trainer.test(model=model, datamodule=picp_data_module)


if __name__ == '__main__':
    # -----------
    # Arguments
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Parse all known and unknown arguments
    args, unparsed_argv = parser.parse_known_args()

    # Let the model add what it wants
    parser = LitGINI.add_model_specific_args(parser)

    # Re-parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = 'dp'  # Test using Data Parallel (DP) and not Distributed Data Parallel (DDP) to avoid PT error
    args.auto_select_gpus = args.auto_choose_gpus
    args.gpus = 1  # Enforce testing to take place on a single GPU
    args.num_nodes = 1  # Enforce testing to take place on a single node
    args.precision = args.gpu_precision
    args.accumulate_grad_batches = args.accum_grad_batches
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo
    args.stochastic_weight_avg = args.stc_weight_avg

    # Finalize all arguments as necessary
    args = process_args(args)

    # Begin execution of model training with given args
    main(args)
