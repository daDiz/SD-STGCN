# ----------------------------------
# utilities for EoN simulations
# -----------------------------------


# --------------------------------------------------------
# convert EoN simulation to NDLib iteration by unit time
# Note: t in the same unit time step is merged into one snapshot
# ---------------------------------------------------------
def sim2Iter_unitTime(sim, G, status_dict={'S':0, 'I':1, 'R':2}):
    '''
    args:
    sim - EoN full data
    G - networkx graph
    status_dict - status to integer

    return:
    NDLib iteration

    Note:
    iteration is a list of dict
    iteration[0] is the initial state of all nodes
    iteration[t] t > 0 only contains states that are changed
    '''
    t = sim.t()
    tmax = int(max(t))
    tmin = int(min(t))

    if tmin != 0:
        raise Exception('simulation does not start from t = 0')


    iteration = {k: {'status': {}} for k in range(tmin, tmax+1)}

    for node in G.nodes:
        ts_list, status_list = sim.node_history(node)
        for time_step, status in zip(ts_list, status_list):
            integer_ts = int(time_step) # integer time step
            if integer_ts not in iteration:
                raise Exception('time step %s not found' % integer_ts)

            status_code = status_dict[status]
            if node in iteration[integer_ts]['status']:
                if status_code > iteration[integer_ts]['status'][node]:
                    iteration[integer_ts]['status'][node] = status_code
            else:
                iteration[integer_ts]['status'][node] = status_code

    return iteration

# ---------------------------------------------
# convert EoN simulation to NDLib iteration
# --------------------------------------------
def sim2Iter(sim, G, status_dict={'S':0, 'I':1, 'R':2}):
    '''
    args:
    sim - EoN full data
    G - networkx graph
    status_dict - status to integer

    return:
    NDLib iteration

    Note:
    iteration is a list of dict
    iteration[0] is the initial state of all nodes
    iteration[t] t > 0 only contains states that are changed
    '''
    t = sim.t()
    iteration = []
    for i in range(len(t)):
        cur = t[i]
        frame = sim.get_statuses(G, cur)
        if i == 0:
            status = {k: status_dict[frame[k]] for k in frame}
        else:
            prev = t[i-1]
            frame = sim.get_statuses(G, cur)
            frame_prev = sim.get_statuses(G, prev)

            status = {}
            for k in frame:
                if frame[k] != frame_prev[k]:
                    status[k] = status_dict[frame[k]]

        iteration.append({'t': t[i], 'status': status})

    return iteration

