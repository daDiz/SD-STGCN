import networkx as nx
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle

def load_data(path):
    df = pd.read_csv(path)
    return df

# ---------------------------------
# overlay dataset on a graph
# ---------------------------------
def build_network(df, g):

    x = df['CaseID'].values
    y = df['Related cases'].values
    z = df['Related cases'].isna().values

    date_onset_symptoms = df['date_onset_symptoms'].values
    date_confirmation = df['date_confirmation'].values
    outcome = df['outcome'].values
    date_discharge = df['date_discharge'].values

    node_data = {}

    node_dict = {}
    k = 0
    n = len(x)
    for i in range(n):
        n1 = int(x[i])
        if n1 not in node_dict:
            node_dict[n1] = k
            k += 1

        if pd.isna(date_onset_symptoms[i]):
            t_I = date_confirmation[i]
        else:
            t_I = date_onset_symptoms[i]

        if pd.isna(outcome[i]):
            t_R = 'Inf'
        else:
            t_R = date_discharge[i]

        node_data[node_dict[n1]] = [t_I, t_R]

        # add edges provided in the original dataset
        if not z[i] and len(y[i]) > 0:
            n2_list = map(int, y[i].split(','))

            for n2 in n2_list:
                if n2 not in node_dict:
                    node_dict[n2] = k
                    k += 1
                    g.add_edge(node_dict[n1], node_dict[n2])

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
    case_id = df['CaseID'].values
    case = df['Case'].values
    src = []
    for i, idx in enumerate(case_id):
        #print(case[i].split(','))
        if case[i].split(', ')[-1] == 'Wuhan':
            src.append(node_dict[idx])

    return src

if __name__ == '__main__':
    df = load_data('../../COVID-19_Singapore_data_revised.csv')

    g = nx.read_edgelist('../../highSchool/data/graph/highSchool.edgelist', nodetype=int)

    g, node_dict, node_data = build_network(df, g)

    print('# nodes: {}'.format(len(g.nodes)))
    print('# edges: {}'.format(len(g.edges)))
    print('is graph connected? {}'.format(nx.is_connected(g)))
    print('density: {}'.format(nx.density(g)))

    iteration = gen_SIR_iteration(node_data, g)

    src = get_src(df, node_dict)

    print('# of srcs {}'.format(len(src)))

    nx.write_edgelist(g, '../data/graph/singapore_highSchool.edgelist', data=False)

    with open('../data/SIR/singapore_node_dict_highSchool.pickle', 'wb') as f:
        pickle.dump(node_dict, f)

    with open('../data/SIR/SIR_singapore_highSchool.pickle', 'wb') as f:
        pickle.dump(([iteration], [src]), f)


