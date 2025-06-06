import pandas as pd, numpy as np
import scanpy as sc

def read_velmeshev_meta():
    # Read metadata on velmeshev cells
    meta = (pd.concat({
                'Ex': pd.read_csv("snRNAseq/data/velmeshev_meta/ex_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num'}, axis=1),
                'In': pd.read_csv("snRNAseq/data/velmeshev_meta/in_meta.tsv", sep='\t', low_memory=False).rename({'Age_(days)':'Age_Num', 'cellId':'Cell_ID'}, axis=1),
                'Macro': pd.read_csv("snRNAseq/data/velmeshev_meta/macro_meta.tsv", sep='\t'),
                'Micro': pd.read_csv("snRNAseq/data/velmeshev_meta/micro_meta.tsv", sep='\t').assign(Cell_Type = 'Microglia')
            })
            .reset_index(0, names='Cell_Class').set_index('Cell_ID')
            # Logic to assign 0 as birth
            .assign(Age_Years = lambda x: np.select(
                [
                    (x['Age'].str.contains('GW')) & (x['Age_Num'] > 268),
                    (~x['Age'].str.contains('GW')) & (x['Age_Num'] < 268)
                ],
                [-0.01,0],
                default = (x['Age_Num']-268)/365)
            )
            .assign(Cell_Class = lambda x: x['Cell_Class'].replace({'Macro':'Glia', 'Micro':'Glia'})) 
            .assign(Cell_Class = lambda x: pd.Categorical(x['Cell_Class'], ordered=True, categories=['Ex','In','Glia']))
            .assign(Cell_Type = lambda x: x['Cell_Type'].replace({'PV_MP':'PV', 'SST_RELN':'SST'}))
            .assign(Cell_Type = lambda x: pd.Categorical(x['Cell_Type'], ordered=True, categories=x['Cell_Type'].unique()))
            .assign(Cell_Lineage = lambda x: np.select(
                [
                    x['Cell_Class'] == 'Ex', 
                    x['Cell_Class'] == 'In',
                    x['Cell_Class'] == 'Glia'
                ],
                ['Excitatory', 'Inhibitory', x['Cell_Type']],
                default='Other'
            ))
            .assign(Cell_Lineage = lambda x: x['Cell_Lineage'].replace({'Fibrous_astrocytes':'Astrocytes', 'Protoplasmic_astrocytes':'Astrocytes'}))
            .assign(Age_Range2 = lambda x: pd.Categorical(np.where(
                    np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']), 
                    x['Age_Range'],
                    pd.cut((x['Age_Num']-273)/365, 
                            bins=[-np.inf,1,2,4,9,17,25,np.inf],
                            labels=['0-1','1-2','2-4','4-9','9-17','17-25','25+'])
                    ), 
                ordered=True, 
                categories=['2nd trimester', '3rd trimester']+['0-1','1-2','2-4','4-9','9-17','17-25','25+'])
            )
            # .assign(Age_Years = lambda x: (x['Age_Num']-273)/365)
            .assign(Age_log2 = lambda x: np.log2(1 + x['Age_Years'] ) )
            .assign(Age_log10 = lambda x: np.log10(1 + x['Age_Years'] ) )
            .assign(Age_Postnatal = lambda x: ~np.isin(x['Age_Range'], ['2nd trimester', '3rd trimester']))
            .assign(Age_Postinfant = lambda x: x['Age_Years'].fillna(0)>=2)
            .assign(Individual = lambda x: x['Individual'].astype('str'))
            .assign(Pseudotime_pct = lambda x: x.groupby('Cell_Class')['Pseudotime'].apply(lambda y: y*100/y.max()).reset_index(0, drop=True)) # 
        )

    return(meta)


def read_velmeshev(path = "snRNAseq/velmeshev2023/velmeshev100k.h5ad"):
    # Function to read Velmeshev 2023 data, joining metadata to get numeric age    
    meta = read_velmeshev_meta()
    adata = sc.read_h5ad(path)
    
    cells = np.intersect1d(adata.obs.index, meta.index)
    
    adata = adata[cells]
    meta = meta.loc[cells]

    adata.obs = (adata.obs
        .assign(dataset = meta['Dataset'])
        .assign(chemistry = meta['Chemistry'])
        .assign(lineage = meta['Cell_Lineage'])
        .assign(age_years = meta['Age_Years'])
        .assign(age_log2 = meta['Age_log2'])
        .assign(region = meta['Region'])
        .assign(region_broad = meta['Region_Broad'])
        .assign(tissue = lambda x: x['tissue'].replace({
            'ganglionic eminence': 'telencephalon',
            'medial ganglionic eminence': 'telencephalon',
            'lateral ganglionic eminence': 'telencephalon',
            'caudal ganglionic eminence': 'telencephalon',
            'cerebral cortex': 'neocortex',
        })
        )
    )

    return(adata)


def read_wang(path = "snRNAseq/wang2024/wang50k.h5ad"):
    # Function to read Wang 2024 data
    adata = sc.read_h5ad(path)

    adata.obs = (adata.obs
        .assign(age_years = lambda x: np.select(
                [
                    (x['Group'].str.contains('trimester')) & (x['Estimated_postconceptional_age_in_days'] > 268),
                    (~x['Group'].str.contains('trimester')) & (x['Estimated_postconceptional_age_in_days'] < 268)
                ],
                [-0.01,0],
                default = (x['Estimated_postconceptional_age_in_days']-268)/365)
            )
        .assign(age_log2 = lambda x: np.log2(1 + x['age_years']))
        .rename({'Region':'region_broad'}, axis=1)
        .assign(region = lambda x: x['tissue'].str.replace('Brodmann (1909) area ', 'BA'))
        .assign(tissue = lambda x: x['tissue'].replace({
            'Brodmann (1909) area 17': 'visual cortex',
            'Brodmann (1909) area 10': 'prefrontal cortex',
            'Brodmann (1909) area 9': 'prefrontal cortex',
            'forebrain': 'neocortex'
        }))
    )

    # Assign dataset as 'Wang'
    adata.obs['dataset'] = 'Wang'
    # Assign chemistry as 'multiome'
    adata.obs['chemistry'] = 'multiome'

    # Map Wang cell_type labels to broad classes using regex
    import re
    def map_cell_lineage_wang(cell_type):
        s = str(cell_type).lower()
        if re.search(r'glutamatergic|corticothalamic|intratelencephalic|extratelencephalic|near-projecting', s):
            return 'Excitatory'
        if re.search(r'gaba|interneuron', s):
            return 'Inhibitory'
        if re.search(r'oligodendrocyte', s):
            return 'Oligos'
        if re.search(r'oligodendrocyte precursor', s):
            return 'OPC'
        if re.search(r'astrocyte', s):
            return 'Astrocytes'
        if re.search(r'microglia', s):
            return 'Microglia'
        if re.search(r'progenitor', s) or re.search(r'radial glial', s):
            return 'Glial_progenitors'
        # if re.search(r'cajal-retzius', s):
        #     return 'Cajal-Retzius cell'
        # if re.search(r'vascular', s):
        #     return 'Vascular cell'
        return 'Other'

    adata.obs['lineage'] = adata.obs['cell_type'].apply(map_cell_lineage_wang)

    return(adata)


def save_simplified_h5ad(input_path, output_path):
    """
    Load an .h5ad file, create a simplified AnnData object with only X, obs_names, and var_names,
    and save it to output_path.
    """
    adata = sc.read_h5ad(input_path)
    # Use raw if available, else X
    X = adata.raw.X if adata.raw is not None else adata.X
    # Minimal obs/var DataFrames
    obs = adata.obs[[]].copy()
    var = adata.var[[]].copy()
    # Create new AnnData
    new_adata = ad.AnnData(X=X, obs=obs, var=var)
    new_adata.obs_names = adata.obs_names.copy()
    new_adata.var_names = adata.var_names.copy()
    # Ensure indices match
    if not new_adata.var.index.equals(new_adata.var_names):
        new_adata.var.index = new_adata.var_names
    if not new_adata.obs.index.equals(new_adata.obs_names):
        new_adata.obs.index = new_adata.obs_names
    new_adata.write_h5ad(output_path)