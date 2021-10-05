import logging
import os

import atom3.case as ca
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from absl import flags, app
from dgl.data import DGLDataset
from torch.utils.data import DataLoader

from project.utils.deepinteract_constants import NODE_COUNT_LIMIT, RESIDUE_COUNT_LIMIT
from project.utils.deepinteract_modules import LitGINI
from project.utils.deepinteract_utils import collect_args, process_args, process_pdb_into_graph, dgl_picp_collate

flags.DEFINE_string('left_pdb_filepath', None, 'A filepath to the left input PDB chain.')
flags.DEFINE_string('right_pdb_filepath', None, 'A filepath to the right input PDB chain.')
flags.DEFINE_string('input_dataset_dir', None, 'Directory in which to store generated features and outputs for inputs.')
flags.DEFINE_string('ckpt_dir', '/mnt/checkpoints', 'Directory from which to load checkpoints.')
flags.DEFINE_string('ckpt_name', None, 'Name of trained model checkpoint to use.')
flags.DEFINE_string('psaia_dir', '/home/Programs/PSAIA_1.0_source/bin/linux/psa',
                    'Path to locally-compiled copy of PSAIA (i.e., to PSA, one of its CLIs)')
flags.DEFINE_string('psaia_config', '/app/DeepInteract/project/datasets/builder/psaia_config_file_input_docker.txt',
                    'Path to input config file for PSAIA')
flags.DEFINE_string('hhsuite_db', None, 'Path to downloaded and extracted HH-suite3-compatible database'
                                        ' (e.g., BFD or Uniclust30')
flags.DEFINE_integer('num_gpus', 0, 'How many GPUs to use to make a prediction (num_gpus=0 means use CPU instead)')

FLAGS = flags.FLAGS


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code curated for DeepInteract (https://github.com/amorehead/DeepInteract):
# -------------------------------------------------------------------------------------------------------------------------------------


class InputDataset(DGLDataset):
    r"""A temporary Dataset for processing and presenting an input complex as a Python dictionary of DGLGraphs.

    Parameters
    ----------
    left_pdb_filepath: str
        A filepath to the left input PDB chain. Default: 'test_data/4heq_l_u.pdb'.
    right_pdb_filepath: str
        A filepath to the right input PDB chain. Default: 'test_data/4heq_r_u.pdb'.
    knn: int
        How many nearest neighbors to which to connect a given node. Default: 20.
    geo_nbrhd_size: int
        Size of each edge's neighborhood when updating geometric edge features. Default: 2.
    self_loops: bool
        Whether to connect a given node to itself. Default: True.
    force_reload: bool
        Whether to reload the dataset. Default: False.
    verbose: bool
        Whether to print out progress information. Default: False.

    Notes
    -----
    The input complex (i.e., both the left and right PDB chains) will be preprocessed into local storage first.

    Examples
    --------
    >>> # Get dataset
    >>> input_data = InputDataset()
    >>>
    >>> len(input_data)
    1
    """

    def __init__(self,
                 left_pdb_filepath=os.path.join('test_data', '4heq_l_u.pdb'),
                 right_pdb_filepath=os.path.join('test_data', '4heq_r_u.pdb'),
                 input_dataset_dir=os.path.join('datasets', 'Input'),
                 psaia_dir='~/Programs/PSAIA_1.0_source/bin/linux/psa',
                 psaia_config='datasets/builder/psaia_config_file_input.txt',
                 hhsuite_db='~/Data/Databases/bfd_metaclust_clu_complete_id30_c90_final_seq.sorted_opt',
                 knn=20,
                 geo_nbrhd_size=2,
                 self_loops=True,
                 force_reload=False,
                 verbose=False):
        self.left_pdb_filepath = left_pdb_filepath
        self.right_pdb_filepath = right_pdb_filepath
        self.input_dataset_dir = input_dataset_dir
        self.psaia_dir = psaia_dir
        self.psaia_config = psaia_config
        self.hhsuite_db = hhsuite_db
        self.knn = knn
        self.geo_nbrhd_size = geo_nbrhd_size
        self.self_loops = self_loops
        self.data = {}

        raw_dir = os.path.join(*left_pdb_filepath.split(os.sep)[:-1])
        super(InputDataset, self).__init__(name='InputDataset',
                                           raw_dir=raw_dir,
                                           force_reload=force_reload,
                                           verbose=verbose)
        logging.info(f"Loading complex for prediction,"
                     f" l_chain: {self.left_pdb_filepath}, r_chain: {self.right_pdb_filepath}")

    def download(self):
        """Download an input complex."""
        pass

    def process(self):
        """Process each protein complex into a prediction-ready dictionary representing both chains."""
        # Process the unprocessed protein complex
        left_complex_graph, right_complex_graph = process_pdb_into_graph(self.left_pdb_filepath,
                                                                         self.right_pdb_filepath,
                                                                         self.input_dataset_dir,
                                                                         self.psaia_dir,
                                                                         self.psaia_config,
                                                                         self.hhsuite_db,
                                                                         self.knn, self.geo_nbrhd_size, self.self_loops)
        self.data = {
            'graph1': left_complex_graph,
            'graph2': right_complex_graph,
            'examples': torch.Tensor(),
            # Both 'complex' and 'filepath' are unused during Lightning's predict_step()
            'complex': self.left_pdb_filepath,
            'filepath': self.left_pdb_filepath
        }

    def has_cache(self):
        """Check if the input complex is available for prediction."""
        pass

    def __getitem__(self, _):
        """Return requested complex to DataLoader."""
        return self.data

    def __len__(self) -> int:
        """Number of complexes in the dataset."""
        return 1

    @property
    def num_chains(self) -> int:
        """Number of protein chains in each complex."""
        return 2

    @property
    def num_classes(self) -> int:
        """Number of possible classes for each graph node."""
        return 2

    @property
    def num_node_features(self) -> int:
        """Number of node feature values after encoding them."""
        return 113

    @property
    def num_edge_features(self) -> int:
        """Number of edge feature values after encoding them."""
        return 27

    @property
    def raw_path(self) -> str:
        """Directory in which to locate raw pairs."""
        return self.raw_dir


def main(argv):
    if len(argv) > 1:
        raise app.UsageError('Too many command-line arguments.')

    # -----------
    # ArgParse
    # -----------
    # Collect all arguments
    parser = collect_args()

    # Let the model add what it wants
    parser = LitGINI.add_model_specific_args(parser)

    # Re-parse all known and unknown arguments after adding those that are model specific
    args, unparsed_argv = parser.parse_known_args()

    # Set Lightning-specific parameter values before constructing Trainer instance
    args.max_time = {'hours': args.max_hours, 'minutes': args.max_minutes}
    args.max_epochs = args.num_epochs
    args.profiler = args.profiler_method
    args.accelerator = 'dp'  # Predict using Data Parallel (DP) and not Distributed Data Parallel (DDP) to avoid errors
    args.auto_select_gpus = args.auto_choose_gpus
    args.gpus = FLAGS.num_gpus  # Allow user to choose how many GPUs to use for inference
    args.num_nodes = 1  # Enforce predictions to to take place on a single node
    args.precision = args.gpu_precision
    args.accumulate_grad_batches = args.accum_grad_batches
    args.gradient_clip_val = args.grad_clip_val
    args.gradient_clip_algo = args.grad_clip_algo
    args.stochastic_weight_avg = args.stc_weight_avg

    # Finalize all arguments as necessary
    args = process_args(args)

    # -----------
    # Input
    # -----------
    logging.info(f'Generating features for {args.left_pdb_filepath} and {args.right_pdb_filepath}')
    input_dataset = InputDataset(left_pdb_filepath=FLAGS.left_pdb_filepath,
                                 right_pdb_filepath=FLAGS.right_pdb_filepath,
                                 input_dataset_dir=FLAGS.input_dataset_dir,
                                 psaia_dir=FLAGS.psaia_dir,
                                 psaia_config=FLAGS.psaia_config,
                                 hhsuite_db=FLAGS.hhsuite_db,
                                 knn=20,
                                 geo_nbrhd_size=2,
                                 self_loops=True)
    input_dataloader = DataLoader(input_dataset, batch_size=1, shuffle=False,
                                  num_workers=0, collate_fn=dgl_picp_collate)

    # -----------
    # Model
    # -----------
    # Assemble a dictionary of model arguments
    dict_args = vars(args)

    # Baseline Model - Geometry-Focused Inter-Graph Node Interaction (GINI)
    model = LitGINI(num_node_input_feats=input_dataset.num_node_features,
                    num_edge_input_feats=input_dataset.num_edge_features,
                    gnn_activ_fn=nn.SiLU(),
                    num_classes=input_dataset.num_classes,
                    max_num_graph_nodes=NODE_COUNT_LIMIT,
                    max_num_residues=RESIDUE_COUNT_LIMIT,
                    testing_with_casp_capri=dict_args['testing_with_casp_capri'],
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
                    batch_size=1,
                    lr=dict_args['lr'],
                    pad=dict_args['pad'],
                    viz_every_n_epochs=dict_args['viz_every_n_epochs'],
                    use_wandb_logger=False,
                    weight_classes=args.weight_classes)
    args.experiment_name = f'LitGINI-b{1}-gl{args.num_gnn_layers}' \
                           f'-n{args.num_gnn_hidden_channels}' \
                           f'-e{args.num_gnn_hidden_channels}' \
                           f'-il{args.num_interact_layers}-i{args.num_interact_hidden_channels}' \
        if not args.experiment_name \
        else args.experiment_name

    # ------------
    # Checkpoint
    # ------------
    ckpt_path = os.path.join(FLAGS.ckpt_dir, FLAGS.ckpt_name)
    ckpt_provided = FLAGS.ckpt_name != ''
    assert ckpt_provided and os.path.exists(ckpt_path), 'A valid checkpoint filepath must be provided'
    model = model.load_from_checkpoint(ckpt_path,
                                       use_wandb_logger=False,
                                       batch_size=args.batch_size,
                                       lr=args.lr,
                                       weight_decay=args.weight_decay,
                                       dropout_rate=args.dropout_rate)
    model.freeze()

    # ------------
    # Trainer
    # ------------
    trainer = pl.Trainer.from_argparse_args(args)

    # -----------
    # Prediction
    # -----------
    # Predict with a trained model using the provided input data module
    predict_payload = trainer.predict(model=model, dataloaders=input_dataloader)[0]
    logits = predict_payload[0][0].squeeze()
    g1_nf, g1_ef, g2_nf, g2_ef = predict_payload[1:]

    # Retrieve the positive-class probabilities to construct the predicted contact probability map
    graph_1_len, graph_2_len = logits.shape[1:]
    flattened_logits = torch.flatten(logits, start_dim=1).transpose(1, 0)
    contact_prob_map = torch.softmax(flattened_logits, dim=1)[:, 1]
    contact_prob_map = torch.reshape(contact_prob_map, (graph_1_len, graph_2_len)).cpu().numpy()

    # -----------
    # Saving
    # -----------
    pdb_code = list(ca.get_complex_pdb_codes([args.left_pdb_filepath, args.right_pdb_filepath]))[0]
    input_prefix = os.sep + os.path.join(*args.left_pdb_filepath.split(os.sep)[:-1])
    contact_map_filepath = os.path.join(input_prefix, f'{pdb_code}_contact_prob_map.npy')
    np.save(contact_map_filepath, contact_prob_map)
    logging.info(f'Saved predicted contact probability map for {pdb_code} as {contact_map_filepath}')

    # Save learned node and edge representations
    g1_nf_filepath = os.path.join(input_prefix, f'{pdb_code}_graph1_node_feats.npy')
    g1_ef_filepath = os.path.join(input_prefix, f'{pdb_code}_graph1_edge_feats.npy')
    g2_nf_filepath = os.path.join(input_prefix, f'{pdb_code}_graph2_node_feats.npy')
    g2_ef_filepath = os.path.join(input_prefix, f'{pdb_code}_graph2_edge_feats.npy')
    np.save(g1_nf_filepath, g1_nf), np.save(g1_ef_filepath, g1_ef)
    np.save(g2_nf_filepath, g2_nf), np.save(g2_ef_filepath, g2_ef)
    logging.info(f'Saved learned node representations for the first chain graph of {pdb_code} as {g1_nf_filepath}')
    logging.info(f'Saved learned edge representations for the first chain graph of {pdb_code} as {g1_ef_filepath}')
    logging.info(f'Saved learned node representations for the second chain graph of {pdb_code} as {g2_nf_filepath}')
    logging.info(f'Saved learned edge representations for the second chain graph of {pdb_code} as {g2_ef_filepath}')


if __name__ == '__main__':
    log_fmt = '%(asctime)s %(levelname)s %(process)d: %(message)s'
    logging.basicConfig(level=logging.DEBUG, format=log_fmt)

    # -----------
    # Absl
    # -----------
    flags.mark_flags_as_required([
        'left_pdb_filepath',
        'right_pdb_filepath',
        'input_dataset_dir',
        'ckpt_name',
        'hhsuite_db'
    ])

    # Begin execution of model training with given args
    app.run(main)
