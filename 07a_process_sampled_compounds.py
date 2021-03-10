import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np


import sys
from rdkit import Chem
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
        return valid, normalized_smiles, ecfp6, hbd, hba, nrb, logp, aqs, tpsa, mw
    except:
        return 0, np.nan, np.zeros((1,2048), np.uint8), np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan
    
    
def process(i):
    filename = f"sampled/{i}/sampled.smi"
    output = f"sampled/{i}/processed.feather"
    outputfp = f"sampled/{i}/processed_fp.npz"
    data = pd.read_csv(filename, sep="\t", header=None)
    data.columns = ['SMILES', "NLL"]
    data["epoch"] = i
    properites = data["SMILES"].map(calculate_properties)
    valid, normalized_smiles, ecfp6, hbd, hba, nrb, logp, aqs, tpsa, mw = zip(*properites)

    ecfp6 = np.vstack(ecfp6)
    ecfp6 = np.packbits(ecfp6, axis=1)
    data["valid"] = valid
    data["normalized_smiles"] = normalized_smiles
    data["hbd"] = hbd
    data["hba"] = hba
    data["nrb"] = nrb
    data["logp"] = logp
    data["aqs"] = aqs
    data["tpsa"] = tpsa
    data["mw"] = mw
    data.to_feather(output)
    np.savez_compressed(outputfp, fps=ecfp6)
    
def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    import rdkit.rdBase as rkrb
    import rdkit.RDLogger as rkl
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog('rdApp.error')
    
    
if __name__ == "__main__":
    import sys
    disable_rdkit_logging()
    i = int(sys.argv[1])
    process(i)
