
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
from joblib import dump, load
from tqdm.auto import tqdm

OVERWRITE_FILES = True


import sys
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors, Crippen, Lipinski
from rdkit.Chem import PandasTools
import pandas as pd
from sklearn.linear_model import LinearRegression
from collections import namedtuple


class ESOLCalculator:
    aromatic_query = Chem.MolFromSmarts("a")
    Descriptor = namedtuple("Descriptor", "mw logp rotors ap")

    @classmethod
    def calc_ap(cls, mol):
        """
        Calculate aromatic proportion #aromatic atoms/#atoms total
        :param mol: input molecule
        :return: aromatic proportion
        """
        matches = mol.GetSubstructMatches(cls.aromatic_query)
        return len(matches) / mol.GetNumAtoms()

    @classmethod
    def calc_esol_descriptors(cls, mol):
        """
        Calcuate mw,logp,rotors and aromatic proportion (ap)
        :param mol: input molecule
        :return: named tuple with descriptor values
        """
        mw = Descriptors.MolWt(mol)
        logp = Crippen.MolLogP(mol)
        rotors = Lipinski.NumRotatableBonds(mol)
        ap = cls.calc_ap(mol)
        return cls.Descriptor(mw=mw, logp=logp, rotors=rotors, ap=ap)

    @classmethod
    def calc_esol_orig(cls, mol):
        """
        Original parameters from the Delaney paper, just here for comparison
        :param mol: input molecule
        :return: predicted solubility
        """
        # just here as a reference don't use this!
        intercept = 0.16
        coef = {"logp": -0.63, "mw": -0.0062, "rotors": 0.066, "ap": -0.74}
        desc = cls.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol

    @classmethod
    def calc_esol(cls, mol):
        """
        Calculate ESOL based on descriptors in the Delaney paper, coefficients refit for the RDKit using the
        routine refit_esol below
        :param mol: input molecule
        :return: predicted solubility
        """
        intercept = 0.26121066137801696
        coef = {'mw': -0.0066138847738667125, 'logp': -0.7416739523408995, 'rotors': 0.003451545565957996, 'ap': -0.42624840441316975}
        desc = cls.calc_esol_descriptors(mol)
        esol = intercept + coef["logp"] * desc.logp + coef["mw"] * desc.mw + coef["rotors"] * desc.rotors \
               + coef["ap"] * desc.ap
        return esol


    
def calculate_properties(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        valid = 1 if mol else 0
        normalized_smiles = Chem.MolToSmiles(mol)
        fp = AllChem.GetMorganFingerprintAsBitVect(mol,3, nBits=2048)
        ecfp6 = np.zeros((1,2048), np.uint8)
        AllChem.DataStructs.ConvertToNumpyArray(fp, ecfp6)
        hbd = Chem.rdMolDescriptors.CalcNumHBD(mol)
        hba = Chem.rdMolDescriptors.CalcNumHBA(mol)
        nrb = Chem.rdMolDescriptors.CalcNumRotatableBonds(mol)
        logp = Chem.rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
        aqs = ESOLCalculator.calc_esol(mol)
        tpsa = Chem.rdMolDescriptors.CalcTPSA(mol)
        mw = Chem.rdMolDescriptors.CalcExactMolWt(mol)
        return ecfp6, hbd, hba, nrb, logp, aqs, tpsa, mw
    except:
        return np.zeros((1,2048), np.uint8), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    
import numpy as np
import scipy.sparse
import torch

def tanimotokernel(data_1, data_2):
    if isinstance(data_1, scipy.sparse.csr_matrix) and isinstance(data_2, scipy.sparse.csr_matrix):
        return _sparse_tanimotokernel(data_1, data_2)
    elif isinstance(data_1, scipy.sparse.csr_matrix) or isinstance(data_2, scipy.sparse.csr_matrix):
        # try to sparsify the input
        return _sparse_tanimotokernel(scipy.sparse.csr_matrix(data_1), scipy.sparse.csr_matrix(data_2)) 
    elif isinstance(data_1, torch.Tensor) and isinstance(data_2, torch.Tensor):
        return _torch_dense_tanimotokernel(data_1, data_2)
    else:  # both are dense try np
        return _dense_tanimotokernel(data_1, data_2)
    
    
    
    
def _dense_tanimotokernel(data_1, data_2):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)
    as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    https://www.sciencedirect.com/science/article/pii/S0893608005001693
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Ralaivola2005Graph.pdf
    """

    norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
    norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
    prod = data_1.dot(data_2.T)

    divisor = (norm_1 + norm_2.T - prod) + np.finfo(data_1.dtype).eps
    return prod / divisor



def _sparse_tanimotokernel(data_1, data_2):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)
    as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    https://www.sciencedirect.com/science/article/pii/S0893608005001693
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Ralaivola2005Graph.pdf
    """

    norm_1 = np.array(data_1.power(2).sum(axis=1).reshape(data_1.shape[0], 1))
    norm_2 = np.array(data_2.power(2).sum(axis=1).reshape(data_2.shape[0], 1))
    prod = data_1.dot(data_2.T).A

    divisor = (norm_1 + norm_2.T - prod) + np.finfo(data_1.dtype).eps
    result = prod / divisor
    return result

def _torch_dense_tanimotokernel(data_1, data_2):
    """
    Tanimoto kernel
        K(x, y) = <x, y> / (||x||^2 + ||y||^2 - <x, y>)
    as defined in:
    "Graph Kernels for Chemical Informatics"
    Liva Ralaivola, Sanjay J. Swamidass, Hiroto Saigo and Pierre Baldi
    Neural Networks
    https://www.sciencedirect.com/science/article/pii/S0893608005001693
    http://members.cbio.mines-paristech.fr/~jvert/svn/bibli/local/Ralaivola2005Graph.pdf
    """
    norm_1 = (data_1 ** 2).sum(axis=1).reshape(data_1.shape[0], 1)
    norm_2 = (data_2 ** 2).sum(axis=1).reshape(data_2.shape[0], 1)
    prod = data_1 @ data_2.T

    divisor = (norm_1 + norm_2.T - prod) + torch.finfo(data_1.dtype).eps
    prod /= divisor
    return prod

def read_npz(file):
    file = np.load(file)
    if "arr_0" in file:
        fp = file["arr_0"]
    else:
        fp = file["fps"]
    file.close()
    fp = np.unpackbits(fp, axis=1)
    return fp

import torch 

def set_default_device_cuda():
    """Sets the default device (cpu or cuda) used for all tensors."""
    if torch.cuda.is_available() == False:
        tensor = torch.FloatTensor
        torch.set_default_tensor_type(tensor)
        return False
    else:  # device_name == "cuda":
        tensor = torch.cuda.FloatTensor  # pylint: disable=E1101
        torch.set_default_tensor_type(tensor)
        return True

    
def read_epoch(epoch):
    sampled_df = pd.read_feather(f"sampled/{epoch}/processed.feather")
    sampled_ecfp = np.array(read_npz(f"sampled/{epoch}/processed_fp.npz"),dtype=np.uint8)
    return sampled_df, sampled_ecfp




def process(epoch):
    if set_default_device_cuda():
        print("Use GPU")
    else:
        print("Use CPU")

    if not (os.path.exists("output/processed.feather") and os.path.exists("output/processed_fp.npz")):
        all_data = pd.read_csv("data/pubchem_assay_compounds_processed.tsv", sep="\t")

        properites = all_data["rdkit_smiles"].map(calculate_properties)
        ecfp6, hbd, hba, nrb, logp, aqs, tpsa, mw = zip(*properites)

        ecfp6 = np.vstack(ecfp6)
        all_data["hbd"] = hbd
        all_data["hba"] = hba
        all_data["nrb"] = nrb
        all_data["logp"] = logp
        all_data["aqs"] = aqs
        all_data["tpsa"] = tpsa
        all_data["mw"] = mw
        all_data.to_feather("output/processed.feather")
        ecfp6 = np.packbits(ecfp6, axis=1)
        np.savez_compressed("output/processed_fp.npz", fps=ecfp6)

    ecfp6 = read_npz("output/processed_fp.npz")
    all_data = pd.read_feather("output/processed.feather")
    all_data["class_set"] = all_data[["class","set"]].apply(lambda x: ' '.join(x), axis=1)
    subsets = list(all_data.class_set.unique())
    distances = [0.1, 0.3, 0.4, 0.5]
    batch_size = 500
    
    if not os.path.exists(f"sampled/{epoch}/processed_with_nn.feather") or OVERWRITE_FILES:
        sampled_df, sampled_ecfp = read_epoch(epoch)
        sampled_ecfp = torch.tensor(sampled_ecfp, dtype=torch.float)
        for subset in subsets:
            subset_idx = all_data.query("class_set == @subset").index.to_list()
            subset_ecfp = torch.tensor(ecfp6[subset_idx], dtype=torch.float)
            results = torch.zeros((len(sampled_df),len(distances)), dtype=np.int)
            for i in tqdm(range(0,len(sampled_df),batch_size)):
                dists = 1 - tanimotokernel(sampled_ecfp[i:i+batch_size],subset_ecfp)
                for j, distance in enumerate(distances):
                    results[i:i+batch_size,j] = (dists <= distance).sum(axis=1)
            colnames = [f"{subset} {dist}" for dist in distances ]
            sampled_df[colnames] = results.cpu().numpy()


        sampled_df.to_feather(f"sampled/{epoch}/processed_with_nn.feather")





import sys

if __name__ == "__main__":    
    import sys
    i = int(sys.argv[1])
    process(i)
