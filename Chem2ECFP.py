import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem,DataStructs
import pandas as pd

if __name__ == "__main__":
    ecfp2 = np.array((1,1024))

    df = pd.read_csv("data/ABL1.1988.csv")
    for idx,line in df.iterrows():
        ids = line["ID"]
        smi = line["SMILES"]
        m = Chem.MolFromSmiles(smi,sanitize=True)
        ecfp = AllChem.GetMorganFingerprint(m,2,useCounts=True)
        arr = np.zeros((0,),dtype=int)
        DataStructs.ConvertToNumpyArray(ecfp,arr)
        print(arr)
        break

