import numpy as np
import matplotlib.pyplot as plt
import gurobipy
import random


def base_opt(lm, consumer_class, esd_assignment, full_iter=2, efficiency=None,
             plot_mode=False, plot_dir=None, plot_subname=None,
             full_return=False, verbose=0):
    total_customers = len(consumer_class)

    # Assigning ESD to customers based on esd_assignment
    for j, (consumer, esd) in enumerate(zip(consumer_class, esd_assignment)):
        consumer.clear_history()
        if esd == 1:
            consumer.activate_esd_constraints()
        elif esd == 2:
            consumer.activate_sp_constraints()
            # print('{} consumer, total x = {}'.format(j + 1, lm.sum()))
            lm[:, j] += - consumer.xp.reshape(-1)
            # print('{} consumer, total x = {}'.format(j + 1, lm.sum()))
        else:
            consumer.deactivate_esd_constraints()

    # Start base algorithm
    totalcost_store = []
    par_store = []
    if efficiency:
        for i in range(full_iter):
            for j in range(total_customers):
                lmsum = np.sum(lm, axis=1).reshape((-1, 1)) - lm[:, j].reshape((-1, 1))
                # print('{} consumer, total x = {}'.format(j + 1, lm[:,j].sum()))
                newecv, par, tcost = consumer_class[j].optimize_continuous_with_efficiency(lm=lmsum, verbose=verbose,
                                                                                           efficiency=efficiency)
                # print('{} consumer, total x = {}'.format(j+1, lm.sum()))
                lm[:, j] = newecv.reshape(-1)
                # print('{} consumer, total x = {}'.format(j + 1, lm[:, j].sum()))
                totalcost_store.append(tcost)
                par_store.append(par)
    else:
        for i in range(full_iter):
            for j in range(total_customers):
                lmsum = np.sum(lm, axis=1).reshape((-1, 1)) - lm[:, j].reshape((-1, 1))
                # print('{} consumer, total x = {}'.format(j + 1, lm[:,j].sum()))
                newecv, par, tcost = consumer_class[j].optimize_continuous(lm=lmsum, verbose=verbose)
                # print('{} consumer, total x = {}'.format(j+1, lm.sum()))
                lm[:, j] = newecv.reshape(-1)
                # print('{} consumer, total x = {}'.format(j + 1, lm[:, j].sum()))
                totalcost_store.append(tcost)
                par_store.append(par)

    if plot_mode:
        plt.figure()
        for i in range(total_customers):
            plt.plot(consumer_class[i].bn_store)
        plt.ylabel('Bill to Individual Customers')
        plt.xlabel('Full Iterations')
        plt.title('Bills vs Full Iterations')
        plt.savefig('{}/bills_{}.png'.format(plot_dir, plot_subname), bbox_inches='tight')
        plt.close()
        plt.plot(par_store)
        plt.xlabel('Iterations')
        plt.ylabel('PAR')
        plt.title('PAR vs Iterations')
        plt.savefig('{}/PAR_{}.png'.format(plot_dir, plot_subname), bbox_inches='tight')
        plt.close()
        plt.plot(totalcost_store)
        plt.xlabel('Iterations')
        plt.ylabel('Total Generation Cost')
        plt.title('Total Generation Cost vs Iterations')
        plt.savefig('{}/Total_Cost_{}.png'.format(plot_dir, plot_subname), bbox_inches='tight')
        plt.close()

    if full_return:
        return totalcost_store, consumer_class, par_store
    else:
        return totalcost_store[-1], consumer_class, par_store[-1]
