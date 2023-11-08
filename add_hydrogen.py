import os
import torch
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm import tqdm

# Function to add hydrogens to ligand files
def add_hydrogens(ligand_path):
    # Check if the ligand file exists
    if not os.path.isfile(ligand_path):
        print(f"Ligand file not found: {ligand_path}")
        return
    suppl = Chem.SDMolSupplier(ligand_path)
    mols = [mol for mol in suppl if mol is not None]
    writer = Chem.SDWriter(ligand_path)
    for mol in mols:
        mol_with_h = Chem.AddHs(mol, addCoords=True)
        writer.write(mol_with_h)
    writer.close()

# Main processing function
def process_data(data_entries, data_root):
    # Process each ligand
    for entry in tqdm(data_entries, desc="Adding hydrogens", unit="ligand"):
        # Get the ligand file path from the entry tuple
        ligand_path = os.path.join(data_root, entry[1])
        # Add hydrogens to the ligand file
        add_hydrogens(ligand_path)

# Load the data
data = torch.load("split_by_name.pt")

# Directory with the data
data_root = 'data/crossdocked_pocket10'

# Process the training data
process_data(data['train'], data_root)