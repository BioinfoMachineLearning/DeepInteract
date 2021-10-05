import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


# -------------------------------------------------------------------------------------------------------------------------------------
# Following code adapted from Graph-Based Protein Design (https://github.com/jingraham/neurips19-graph-protein-design):
# -------------------------------------------------------------------------------------------------------------------------------------
def gather_nodes(nodes, neighbor_idx):
    """Collect node features of neighbor."""
    # Features [B,N,C] at Neighbor indices [B,N,K] => [B,N,K,C]
    # Flatten and expand indices per batch [B,N,K] => [B,NK] => [B,NK,C]
    neighbors_flat = neighbor_idx.view((neighbor_idx.shape[0], -1))
    neighbors_flat = neighbors_flat.unsqueeze(-1).expand(-1, -1, nodes.size(2))
    # Gather and re-pack
    neighbor_features = torch.gather(nodes, 1, neighbors_flat)
    neighbor_features = neighbor_features.view(list(neighbor_idx.shape)[:3] + [-1])
    return neighbor_features


def gather_edges(edges, neighbor_idx):
    """Collect edge features of neighbor."""
    # Features [B,N,N,C] at Neighbor indices [B,N,K] => Neighbor features [B,N,K,C]
    neighbors = neighbor_idx.unsqueeze(-1).expand(-1, -1, -1, edges.size(-1))
    edge_features = torch.gather(edges, 2, neighbors)
    return edge_features


class PositionalEncodings(nn.Module):
    """Encode the positional index of a node."""

    def __init__(self, num_embeddings, period_range=None):
        super(PositionalEncodings, self).__init__()
        if period_range is None:
            period_range = [2, 1000]
        self.num_embeddings = num_embeddings
        self.period_range = period_range

    def forward(self, E_idx):
        # i-j
        N_batch, N_nodes, N_neighbors = E_idx.size(0), E_idx.size(1), E_idx.size(2)
        ii = torch.arange(N_nodes, dtype=torch.float32).view((1, -1, 1))
        d = (E_idx.float() - ii).unsqueeze(-1)
        # Original Transformer frequencies
        frequency = torch.exp(
            torch.arange(0, self.num_embeddings, 2, dtype=torch.float32)
            * -(np.log(10000.0) / self.num_embeddings)
        )
        # Grid-aligned
        # frequency = 2. * np.pi * torch.exp(
        #     -torch.linspace(
        #         np.log(self.period_range[0]),
        #         np.log(self.period_range[1]),
        #         self.num_embeddings / 2
        #     )
        # )
        angles = d * frequency.view((1, 1, 1, -1))
        E = torch.cat((torch.cos(angles), torch.sin(angles)), -1)
        return E


class GeometricProteinFeatures(nn.Module):
    """Extract geometric protein features."""

    def __init__(self, num_positional_embeddings=20, num_rbf=18, dropout_rate=0.1, features_type='full'):
        super(GeometricProteinFeatures, self).__init__()
        assert 0 < num_rbf < 20, 'Number of RBFs to be computed must be between 1 and 19, inclusively'
        self.num_positional_embeddings = num_positional_embeddings
        self.num_rbf = num_rbf
        self.dropout = nn.Dropout(dropout_rate)  # Used on hydrogen bonds and contacts with neighbors
        self.features_type = features_type

    @staticmethod
    def get_masked_neighbors(edge_ids: torch.Tensor, mask: torch.Tensor):
        """Find indices of neighbors after applying masking."""
        mask_2d = torch.unsqueeze(mask, 1) * torch.unsqueeze(mask, 2)
        mask_neighbors = gather_edges(mask_2d.unsqueeze(-1), edge_ids)
        return mask_neighbors

    @staticmethod
    def compute_rbfs(pairwise_dists: torch.Tensor, num_rbf: int):
        """Apply radial basis function on pairwise squared distances."""
        D_min, D_max, D_count = 0., 20., num_rbf
        D_mu = torch.linspace(D_min, D_max, D_count)
        D_mu = D_mu.view([1, 1, 1, -1])
        D_sigma = (D_max - D_min) / D_count
        D_expand = torch.unsqueeze(pairwise_dists, -1)
        RBF = torch.exp(-((D_expand - D_mu) / D_sigma) ** 2)

        # for i in range(D_count):
        #     fig = plt.figure(figsize=(4, 4))
        #     rbf_i = RBF.data.numpy()[0, i, :, :]
        #     # rbf_i = D.data.numpy()[0,0,:,:]
        #     plt.imshow(rbf_i, aspect='equal')
        #     plt.axis('off')
        #     plt.tight_layout()
        #     plt.savefig('rbf{}.pdf'.format(i))
        #     print(np.min(rbf_i), np.max(rbf_i), np.mean(rbf_i))
        # exit(0)
        return RBF

    @staticmethod
    def convert_rotations_into_quaternions(R: torch.Tensor):
        """
        Convert a batch of 3D rotations [R] to quaternions [Q]
        R [...,3,3]
        Q [...,4].
        """
        # For the simple Wikipedia version, see: en.wikipedia.org/wiki/Rotation_matrix#Quaternion
        # For other options, see math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        diag = torch.diagonal(R, dim1=-2, dim2=-1)
        Rxx, Ryy, Rzz = diag.unbind(-1)
        magnitudes = 0.5 * torch.sqrt(torch.abs(1 + torch.stack([
            Rxx - Ryy - Rzz,
            - Rxx + Ryy - Rzz,
            - Rxx - Ryy + Rzz
        ], -1)))
        _R = lambda i, j: R[:, :, :, i, j]
        signs = torch.sign(torch.stack([
            _R(2, 1) - _R(1, 2),
            _R(0, 2) - _R(2, 0),
            _R(1, 0) - _R(0, 1)
        ], -1))
        xyz = signs * magnitudes
        # The relu enforces a non-negative trace
        w = torch.sqrt(F.relu(1 + diag.sum(-1, keepdim=True))) / 2.  # Ensure we only get the real component
        Q = torch.cat((xyz, w), -1)
        Q = F.normalize(Q, dim=-1)

        # Axis of rotation
        # Replace bad rotation matrices with identity
        # I = torch.eye(3).view((1,1,1,3,3))
        # I = I.expand(*(list(R.shape[:3]) + [-1,-1]))
        # det = (
        #     R[:,:,:,0,0] * (R[:,:,:,1,1] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,1])
        #     - R[:,:,:,0,1] * (R[:,:,:,1,0] * R[:,:,:,2,2] - R[:,:,:,1,2] * R[:,:,:,2,0])
        #     + R[:,:,:,0,2] * (R[:,:,:,1,0] * R[:,:,:,2,1] - R[:,:,:,1,1] * R[:,:,:,2,0])
        # )
        # det_mask = torch.abs(det.unsqueeze(-1).unsqueeze(-1))
        # R = det_mask * R + (1 - det_mask) * I

        # DEBUG
        # https://math.stackexchange.com/questions/2074316/calculating-rotation-axis-from-rotation-matrix
        # Columns of this are in rotation plane
        # A = R - I
        # v1, v2 = A[:,:,:,:,0], A[:,:,:,:,1]
        # axis = F.normalize(torch.cross(v1, v2), dim=-1)
        return Q

    @staticmethod
    def get_contacts(D_neighbors: torch.Tensor, mask_neighbors: torch.Tensor, cutoff=8):
        """Find contacts."""
        D_neighbors = D_neighbors.unsqueeze(-1)
        neighbor_C = mask_neighbors * (D_neighbors < cutoff).type(torch.float32)
        return neighbor_C

    @staticmethod
    def get_hbonds(X: torch.Tensor, E_idx: torch.Tensor, mask_neighbors: torch.Tensor, eps=1E-3):
        """Derive hydrogen bonds and contact map."""
        X_atoms = dict(zip(['N', 'CA', 'C', 'O'], torch.unbind(X, 2)))

        # Virtual hydrogen atoms
        X_atoms['C_prev'] = F.pad(X_atoms['C'][:, 1:, :], (0, 0, 0, 1), 'constant', 0)
        X_atoms['H'] = X_atoms['N'] + F.normalize(
            F.normalize(X_atoms['N'] - X_atoms['C_prev'], -1)
            + F.normalize(X_atoms['N'] - X_atoms['CA'], -1)
            , -1)

        def _distance(X_a: torch.Tensor, X_b: torch.Tensor):
            return torch.norm(X_a[:, None, :, :] - X_b[:, :, None, :], dim=-1)

        def _inv_distance(X_a: torch.Tensor, X_b: torch.Tensor):
            return 1. / (_distance(X_a, X_b) + eps)

        # DSSP vacuum electrostatics model
        U = (0.084 * 332) * (
                _inv_distance(X_atoms['O'], X_atoms['N'])
                + _inv_distance(X_atoms['C'], X_atoms['H'])
                - _inv_distance(X_atoms['O'], X_atoms['H'])
                - _inv_distance(X_atoms['C'], X_atoms['N'])
        )

        HB = (U < -0.5).type(torch.float32)
        neighbor_HB = mask_neighbors * gather_edges(HB.unsqueeze(-1), E_idx)
        # print(HB)
        # HB = F.sigmoid(U)
        # U_np = U.cpu().data.numpy()
        # # plt.matshow(np.mean(U_np < -0.5, axis=0))
        # plt.matshow(HB[0,:,:])
        # plt.colorbar()
        # plt.show()
        # D_CA = _distance(X_atoms['CA'], X_atoms['CA'])
        # D_CA = D_CA.cpu().data.numpy()
        # plt.matshow(D_CA[0,:,:] < contact_D)
        # # plt.colorbar()
        # plt.show()
        # exit(0)
        return neighbor_HB

    def get_coarse_orientation_feats(self, X: torch.Tensor, E_idx: torch.Tensor, eps=1e-6):
        """Derive pair features."""
        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Bond angle calculation
        cosA = -(u_1 * u_0).sum(-1)
        cosA = torch.clamp(cosA, -1 + eps, 1 - eps)
        A = torch.acos(cosA)
        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)
        # Backbone features
        AD_features = torch.stack((torch.cos(A), torch.sin(A) * torch.cos(D), torch.sin(A) * torch.sin(D)), 2)
        AD_features = F.pad(AD_features, (0, 0, 1, 2), 'constant', 0)

        # Build relative orientations
        o_1 = F.normalize(u_2 - u_1, dim=-1)
        O = torch.stack((o_1, n_2, torch.cross(o_1, n_2)), 2)
        O = O.view(list(O.shape[:2]) + [9])
        O = F.pad(O, (0, 0, 1, 2), 'constant', 0)

        # DEBUG: Viz [dense] pairwise orientations
        # O = O.view(list(O.shape[:2]) + [3,3])
        # dX = X.unsqueeze(2) - X.unsqueeze(1)
        # dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        # dU = dU / torch.norm(dU, dim=-1, keepdim=True)
        # dU = (dU + 1.) / 2.
        # plt.imshow(dU.data.numpy()[0])
        # plt.show()
        # print(dX.size(), O.size(), dU.size())
        # exit(0)

        O_neighbors = gather_nodes(O, E_idx)
        X_neighbors = gather_nodes(X, E_idx)

        # Re-view as rotation matrices
        O = O.view(list(O.shape[:2]) + [3, 3])  # O can map from a global ref. frame to a node's local ref. frame
        O_neighbors = O_neighbors.view(list(O_neighbors.shape[:3]) + [3, 3])

        # Rotate into local reference frames
        dX = X_neighbors - X.unsqueeze(-2)
        dU = torch.matmul(O.unsqueeze(2), dX.unsqueeze(-1)).squeeze(-1)
        dU = F.normalize(dU, dim=-1)
        # dU represents the relative direction to neighboring node j from node i's ref. frame
        R = torch.matmul(O.unsqueeze(2).transpose(-1, -2), O_neighbors)
        Q = self.convert_rotations_into_quaternions(R)
        # Q is a compact representation of the rotation matrices that map to neighboring node j from node i's ref. frame

        # Orientation features
        O_features = torch.cat((dU, Q), dim=-1)

        # DEBUG: Viz pairwise orientations
        # IMG = Q[:,:,:,:3]
        # # IMG = dU
        # dU_full = torch.zeros(X.shape[0], X.shape[1], X.shape[1], 3).scatter(
        #     2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), IMG
        # )
        # print(dU_full)
        # dU_full = (dU_full + 1.) / 2.
        # plt.imshow(dU_full.data.numpy()[0])
        # plt.show()
        # exit(0)
        # print(Q.sum(), dU.sum(), R.sum())
        return AD_features, O_features

    @staticmethod
    def get_dihedrals(X: torch.Tensor, eps=1e-7):
        """Calculate dihedral angles given atomic coordinates."""
        # First 3 coordinates are N, CA, C
        X = X[:, :, :3, :].reshape(X.shape[0], 3 * X.shape[1], 3)

        # Shifted slices of unit vectors
        dX = X[:, 1:, :] - X[:, :-1, :]
        U = F.normalize(dX, dim=-1)
        u_2 = U[:, :-2, :]
        u_1 = U[:, 1:-1, :]
        u_0 = U[:, 2:, :]
        # Backbone normals
        n_2 = F.normalize(torch.cross(u_2, u_1), dim=-1)
        n_1 = F.normalize(torch.cross(u_1, u_0), dim=-1)

        # Angle between normals
        cosD = (n_2 * n_1).sum(-1)
        cosD = torch.clamp(cosD, -1 + eps, 1 - eps)
        D = torch.sign((u_2 * n_1).sum(-1)) * torch.acos(cosD)

        # This scheme will remove phi[0], psi[-1], omega[-1]
        D = F.pad(D, (1, 2), 'constant', 0)
        D = D.view((D.size(0), int(D.size(1) / 3), 3))
        # phi, psi, omega = torch.unbind(D, -1)

        # print(cosD.cpu().data.numpy().flatten())
        # print(omega.sum().cpu().data.numpy().flatten())

        # Bond angle calculation
        # A = torch.acos(-(u_1 * u_0).sum(-1))

        # DEBUG: Ramachandran plot
        # x = phi.cpu().data.numpy().flatten()
        # y = psi.cpu().data.numpy().flatten()
        # plt.scatter(x * 180 / np.pi, y * 180 / np.pi, s=1, marker='.')
        # plt.xlabel('phi')
        # plt.ylabel('psi')
        # plt.axis('square')
        # plt.grid()
        # plt.axis([-180,180,-180,180])
        # plt.show()

        # Lift angle representations to the circle
        D_features = torch.cat((torch.cos(D), torch.sin(D)), 2)
        return D_features

    def forward(self, coords: torch.Tensor, pairwise_dists: torch.Tensor, edge_ids: torch.Tensor, mask: torch.Tensor):
        """Transform atom coordinates into geometric node (i.e., residue) and edge (i.e., residue-residue) features."""
        # Debug plot KNN
        # print(E_idx[:10, :10])
        # D_simple = mask_2D * torch.zeros(D.size()).scatter(-1, E_idx, torch.ones_like(D_neighbors))
        # print(D_simple)
        # D_simple = D.data.numpy()[0, :, :]
        # plt.imshow(D_simple, aspect='equal')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.savefig('D_knn.pdf')
        # exit(0)

        # Find masked neighbors
        masked_neighbors = self.get_masked_neighbors(edge_ids, mask)

        # Derive pairwise features
        ca_atom_coords = coords[:, :, 1, :]
        rbf_feats = self.compute_rbfs(pairwise_dists, self.num_rbf)
        ad_features, o_features = self.get_coarse_orientation_feats(ca_atom_coords, edge_ids)

        if self.features_type == 'coarse':
            # Coarse backbone features
            geo_node_feats = ad_features
            geo_edge_feats = torch.cat((rbf_feats, o_features), -1)
        elif self.features_type == 'hbonds':
            # Hydrogen bonds and contacts
            neighbor_hb = self.get_hbonds(coords, edge_ids, masked_neighbors)
            neighbor_c = self.get_contacts(pairwise_dists, masked_neighbors)
            # Dropout
            neighbor_c = self.dropout(neighbor_c)
            neighbor_hb = self.dropout(neighbor_hb)
            # Pack
            geo_node_feats = mask.unsqueeze(-1) * torch.ones_like(ad_features)
            neighbor_c = neighbor_c.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            neighbor_hb = neighbor_hb.expand(-1, -1, -1, int(self.num_positional_embeddings / 2))
            geo_edge_feats = torch.cat((neighbor_c, neighbor_hb), -1)
        # Default to assigning to nodes (circular) dihedral angles and to edges distance and orientation features
        elif self.features_type == 'full':
            # Full backbone angles
            geo_node_feats = self.get_dihedrals(coords)
            geo_edge_feats = torch.cat((rbf_feats, o_features), -1)
        elif self.features_type == 'dist':
            # Full backbone angles
            geo_node_feats = self.get_dihedrals(coords)
            geo_edge_feats = rbf_feats
        else:  # Otherwise, assume full backbone angles
            geo_node_feats = self.get_dihedrals(coords)
            geo_edge_feats = rbf_feats

        # DEBUG
        # U = (np.nan * torch.zeros(X.size(0),X.size(1),X.size(1),3)).scatter(2, E_idx.unsqueeze(-1).expand(-1,-1,-1,3), E[:,:,:,:3])
        # plt.imshow(U.data.numpy()[0,:,:,0])
        # plt.show()
        # exit(0)
        return geo_node_feats, geo_edge_feats
