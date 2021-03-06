import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

def load_data(path):
    df = pd.read_csv(path)
    return df

def load_graph(path):
    g = nx.read_edgelist(path, nodetype=int)
    return g


# ---------------------------------
# overlay dataset on a given graph
# ---------------------------------
def build_network(df, g0, g):
    # Network Definition

    x = df['case_id'].values

    date_onset_symptoms = df['symptom_onset'].values
    date_confirmation = df['confirm_date'].values
    date_death = df['death'].values

    node_data = {}
    node_dict = {}

    k = 0
    n = len(x)
    for i in range(n):
        n1 = x[i].lower()

        if n1 not in node_dict:
            node_dict[n1] = k
            k += 1

        if pd.isna(date_onset_symptoms[i]):
            t_I = date_confirmation[i]
        else:
            t_I = date_onset_symptoms[i]

        if pd.isna(date_death[i]):
            t_R = 'Inf'
        else:
            t_R = date_death[i]

        node_data[node_dict[n1]] = [t_I, t_R]

    for e in g0.edges:
        a = str(e[0]).strip('\"')
        b = str(e[1]).strip('\"')
        x = node_dict[a]
        y = node_dict[b]
        g.add_edge(x,y)


    return g, node_dict, node_data


# -----------------------------
# get a sequence of snapshots (SIR)
# ------------------------------
def gen_SIR_iteration(node_data, g, date_format='%Y-%m-%d'):
    t_I = sorted(np.unique([datetime.strptime(node_data[k][0], date_format) for k in node_data]))
    t_R = sorted(np.unique([datetime.strptime(node_data[k][1], date_format) for k in node_data if \
                            node_data[k][1] != 'Inf']))


    t_start = min(t_I[0], t_R[0])
    t_end = max(t_I[-1], t_R[-1])

    time_dict = {}
    for i in range((t_end - t_start).days):
        t = t_start + timedelta(i)
        time_dict[t] = i

    time_dict[t_end] = (i+1)


    iteration = [{'iteration':i, 'status': {}} for i in range(len(time_dict))]

    # initialize iteration (note: 0 step should have all the nodes)
    iteration[0]['status'] = {k: 0 for k in g.nodes}

    for k in node_data:
        t1 = datetime.strptime(node_data[k][0], date_format)
        i = time_dict[t1]
        iteration[i]['status'][k] = 1

        t2 = node_data[k][1]
        if t2 != 'Inf':
            t2 = datetime.strptime(t2, date_format)
            j = time_dict[t2]
            iteration[j]['status'][k] = 2

    return iteration


# -----------------------
# get possible sources
# -----------------------
def get_src(df, node_dict):
    case_id = df['case_id'].values
    case = df['Infection_source'].values
    src = []
    for i, idx in enumerate(case_id):
        if pd.isna(case[i]):
            continue
        if 'Wuhan' in case[i] or 'Hubei' in case[i]:
            src.append(node_dict[idx.lower()])

    return src

if __name__ == '__main__':
    df = load_data('../../Tianjin135cases_revissed.csv')

    g0 = nx.read_edgelist('../../tianjin_caseid.edgelist')

    N = 1000

    n_graphs = 5

    for gid in range(n_graphs):
        g = load_graph('../../highSchool/data/graph/highSchool_config_%s-%s.edgelist' % (N, gid))

        g, node_dict, node_data = build_network(df, g0, g)

        print('graph %s' % (gid))
        print('# nodes: {}'.format(len(g.nodes)))
        print('# edges: {}'.format(len(g.edges)))
        print('is graph connected? {}'.format(nx.is_connected(g)))

        iteration = gen_SIR_iteration(node_data, g, date_format='%d/%m/%Y')

        #print('Iteration: ')
        #print(iteration)

        src = get_src(df, node_dict)

        print('# of srcs {}'.format(len(src)))

        nx.write_edgelist(g, '../data/graph/tianjin_config_N%s_g%s.edgelist' % (N, gid), data=False)

        with open('../data/SIR/tianjin_node_dict_config_N%s_g%s.pickle' % (N, gid), 'wb') as f:
            pickle.dump(node_dict, f)

        with open('../data/SIR/SIR_tianjin_config_N%s_g%s.pickle' % (N, gid), 'wb') as f:
            pickle.dump(([iteration], [src]), f)


