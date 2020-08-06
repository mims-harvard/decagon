from collections import defaultdict
import networkx as nx
import pandas as pd
from typing import Tuple


def load_combo_se(combo_path: str = 'bio-decagon-combo.csv') -> \
        Tuple[nx.Graph, dict, dict, dict]:
    """

    Parameters
    ----------
    combo_path: path to file with table drug-drug-side

    Returns
    -------
        1. drug-drug network
        2. dictionary from combination ID to pair of stitch IDs
        3. dictionary from combination ID to set of polypharmacy side effects
        4. dictionary from side effects to their names
    """
    print(f'Reading: {combo_path}')
    combo_df = pd.read_csv(combo_path, delimiter=',', header=0)
    combo_id = combo_df['STITCH 1'] + '_' + combo_df['STITCH 2']

    # from side effects to their names
    se2name = dict(zip(combo_df['Polypharmacy Side Effect'],
                       combo_df['Side Effect Name']))

    # from combination ID to pair of stitch IDs
    combo2stitch = dict(zip(combo_id,
                            combo_df[['STITCH 1', 'STITCH 2']].values))

    # from combination ID to set of polypharmacy side effects
    combo2se = defaultdict(set)
    for combo_id, se in zip(combo_id, combo_df['Polypharmacy Side Effect']):
        combo2se[combo_id].add(se)

    n_interactions = sum([len(v) for v in combo2se.values()])
    print(f'Drug combinations: {len(combo2stitch)}' +
          f'Side effects: {len(se2name)}')
    print(f'Drug-drug interactions: {n_interactions}')

    # drug-drug net
    net = nx.Graph(list(combo2stitch.values()))
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(list(nx.selfloop_edges(net)))
    print(f'Edges: {len(net.edges)}')
    print(f'Nodes: {len(net.nodes)}')
    return net, combo2stitch, combo2se, se2name


def load_ppi(ppi_path: str = 'bio-decagon-ppi.csv') -> Tuple[nx.Graph, dict]:
    """

    Parameters
    ----------
    ppi_path: Path to file with ppi

    Returns
    -------
        1. protein-protein network
        2. dictionary that maps each gene ID to a number

    """
    print(f'Reading: {ppi_path}')
    ppi_df = pd.read_csv(ppi_path, delimiter=',', header=0)
    edges = ppi_df[['Gene 1', 'Gene 2']].values

    # protein-protein net
    net = nx.Graph(list(edges))
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(list(nx.selfloop_edges(net)))
    print(f'Edges: {len(net.edges)}')
    print(f'Nodes: {len(net.nodes)}')

    # from gene ID to a number
    node2idx = {node: i for i, node in enumerate(net.nodes())}
    return net, node2idx


def load_mono_se(mono_path:str='bio-decagon-mono.csv') -> Tuple[dict, dict]:
    """

    Parameters
    ----------
    mono_path: path to file with side effects of drugs (individual)

    Returns
    -------
        1. dictionary from Stitch ID to set of individual side effects
        2. dictionary from side effects to their names

    """
    print(f'Reading: {mono_path}')
    mono_df = pd.read_csv(mono_path)

    # from side effects to their names
    se2name = dict(mono_df[['Individual Side Effect',
                            'Side Effect Name']].values)

    # from Stitch ID to set of individual side effects
    stitch2se = defaultdict(set)
    for stitch, se in mono_df[['STITCH', 'Individual Side Effect']].values:
        stitch2se[stitch].add(se)

    return stitch2se, se2name


def load_targets(targets_path: str = 'bio-decagon-targets.csv') -> \
        Tuple[nx.Graph, dict]:
    """

    Parameters
    ----------
    target_path: path to drug-target associations

    Returns
    -------
        1. drug-target network
        2. dictionary from Stitch ID to set of drug targets

    """
    print(f'Reading: {targets_path}')
    target_df = pd.read_csv(targets_path)

    # from Stitch ID to set of drug targets
    stitch2proteins = defaultdict(set)
    for stitch, gene in target_df[['STITCH', 'Gene']].values:
        stitch2proteins[stitch].add(gene)

    #drug-target network
    net = nx.Graph(list(target_df[['STITCH', 'Gene']].values))
    net.remove_nodes_from(nx.isolates(net))
    net.remove_edges_from(list(nx.selfloop_edges(net)))
    print(f'Edges: {len(net.edges)}')
    print(f'Nodes: {len(net.nodes)}')
    return net, stitch2proteins


def load_categories(se_categories_path:str=
                    'bio-decagon-effectcategories.csv') -> Tuple[dict, dict]:
    """

    Parameters
    ----------
    se_categories_path: path to data with side effects and corresponded disease
                        classes

    Returns
    -------
        1. dictionary from side effect to disease class of that side effect
        2. dictionary from side effects to their names

    """
    print(f'Reading: {se_categories_path}')
    se_cat_df = pd.read_csv(se_categories_path)
    se2name = dict(se_cat_df[['Side Effect', 'Side Effect Name']].values)
    se2class = dict(se_cat_df[['Side Effect', 'Disease Class']].values)
    return se2class, se2name
