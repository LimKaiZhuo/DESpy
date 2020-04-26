import numpy as np
import pandas as pd
import pickle

from own_packages.consumer_class import Consumer


def read_consumer_excel(consumer_excel_file, solar_excel_file, sigma, k, save_path=None):
    solar_df = pd.read_excel(solar_excel_file, index_col=0)
    xp = solar_df.values

    xls = pd.ExcelFile(consumer_excel_file).sheet_names
    all_sheets_df = pd.read_excel(consumer_excel_file, index_col=0, sheet_name=None)

    sheet_names = []
    i = 0
    while i < len(xls):
        count_per_class = 6
        sheet_names.append(xls[i:i + count_per_class])
        i += count_per_class

    c_idx = 1
    consumer_class = []
    oo_store = []
    ns_store = []

    for subclass, (s_a, s_b, s_c, s_d, ns, oo) in enumerate(sheet_names):
        ns_store.append(all_sheets_df[ns].values)
        oo_store.append(all_sheets_df[oo].values)
    xt_no_ecs = np.sum(sum(oo_store), axis=0).reshape(-1, 1) + np.sum(sum(ns_store), axis=0).reshape(-1, 1)

    for subclass, (s_a, s_b, s_c, s_d, ns, oo) in enumerate(sheet_names):
        a_df = all_sheets_df[s_a]
        b_df = all_sheets_df[s_b]
        c_df = all_sheets_df[s_c]
        d_df = all_sheets_df[s_d]
        ns_df = all_sheets_df[ns]

        subclass_pop = len(a_df.columns.values)
        a = a_df.values
        b = b_df.values
        c = c_df.values
        d = d_df.values
        ns = ns_df.values
        for idx in range(subclass_pop):
            consumer_class.append(Consumer(ns=ns[idx, :],
                                           a=a[:, idx], b=b[:, idx], c=c[:, idx], d=d[:, idx], k=k,
                                           xp=xp, sigma=sigma, xt_no_ecs=xt_no_ecs,
                                           group=subclass, c_idx=c_idx))
            c_idx += 1

    xt = np.zeros((24, 1))
    for consumer in consumer_class:
        xt += consumer.xn_init_first
        consumer.oo_store = oo_store

    xt = xt * sigma + (1 - sigma) * xt_no_ecs
    max_x = max(xt)

    w1 = 2218
    w2 = 33.69477939605713
    for consumer in consumer_class:
        consumer.coef0 = 129.592 / w1
        consumer.coef1 = 9.99347 * (10 ** -6)
        consumer.coef2 = 1.77324 * w2*w1 * (10 ** -14)


    total_cost = np.ones((1, 24)) @ (consumer_class[0].coef0 + consumer_class[0].coef1 * xt +
                                     consumer_class[0].coef2 * (xt ** 2)) * consumer_class[0].k
    for consumer in consumer_class:
        consumer.tcost_init = total_cost.item()
        consumer.par_init = (max_x / np.average(xt)).item()
        consumer.bn_store[-1] = (
                np.sum(consumer.xn_init_first.reshape(-1)) / np.sum(xt.reshape(-1)) * total_cost).item()

    if save_path:
        with open(save_path, "wb") as f:
            pickle.dump(consumer_class, f)

    return consumer_class
