import numpy as np
import cvxpy as cp


class Consumer:
    def __init__(self, ns, a, b, c, d, group, c_idx, xp, sigma, xt_no_ecs, k=1):
        self.c_idx = c_idx
        self.k = k
        self.ns = np.reshape(ns, (-1, 1))
        self.app_count = len(a)

        self.a = []
        ones = [1] * 24
        zeros = [0] * 24
        for a_idx in a:
            single_a = [0] * 24
            single_a[a_idx:] = ones[a_idx:]
            self.a.append(single_a)
        self.a = np.array(self.a).T

        self.b = []
        for b_idx in b:
            single_b = [1] * 24
            if b_idx == 23:
                pass
            else:
                # For example, if b=23, it means appliance must end by end of 23th hour NOT at the start of 23th hour
                b_idx += 1
                single_b[b_idx:] = zeros[b_idx:]
            self.b.append(single_b)
        self.b = np.array(self.b).T

        self.c = np.reshape(c, (-1, 1))
        d = np.reshape(d, (-1, 1))
        self.e = d / self.c
        self.e[self.e == np.inf] = 0
        self.e = np.nan_to_num(self.e)
        self.d = self.e * self.c
        self.group = group
        self.esd = False
        self.sp = False
        self.xp = xp  # constant solar power generation hourly vector

        # Fractional Sign ups
        self.sigma = sigma
        self.xt_no_ecs = xt_no_ecs

        # Tracking
        self.t_store = []
        self.xn_store = []
        self.bn_store = []
        self.qss_store = []
        self.xnb_store = []

        # Init xn without optimization for first bill calculation
        t = np.zeros((24, self.app_count))
        for idx, (single_a, single_c) in enumerate(zip(self.a.T.tolist(), self.c.flatten().tolist())):
            # each single_a is a 24 len list for one appliance hourly information whether it is on or off
            start_idx = single_a.index(1)
            t[:, idx] = [0] * start_idx + [1] * single_c + [0] * (24 - start_idx - single_c)
        self.xn_init_first = self.ns + t @ self.e

        # Init xn assuming lm = 0
        self.scale = 1000  # Initial scaling. Will be changed later in the preprocess function
        self.coef0 = 129.592
        self.coef1 = 9.99347 * (10 ** -6) * self.scale
        self.coef2 = 1.77324 * (self.scale ** 2) * (10 ** -14)
        self.xn_init, _, _ = self.optimize_continuous(lm=np.zeros((24, 1)), verbose=0)

    def activate_esd_constraints(self):
        self.esd = True
        self.sp = False

    def activate_sp_constraints(self):
        # SP has its own type of ESD built in that need not be the same as the stand-alone ESD
        self.esd = True
        self.sp = True

    def deactivate_esd_constraints(self):
        self.esd = False
        self.sp = False

    def clear_history(self):
        # Tracking
        self.t_store = [self.t_store[0]]
        self.xn_store = [self.xn_store[0]]
        self.bn_store = [self.bn_store[0]]
        self.qss_store = []
        self.xnb_store = []

    def optimize_continuous(self, lm, iteration=-1, verbose=1):
        # ECV variable to optimize
        t = cp.Variable(shape=(24, self.app_count))

        # ECV constraints
        ecv_1 = t - self.a <= 0
        ecv_2 = t - self.b <= 0
        ecv_3 = t >= 0
        ecv_4 = t <= 1
        constraints = [ecv_1, ecv_2, ecv_3, ecv_4]
        for app in range(self.app_count):
            post_mul = np.zeros((self.app_count, 1))
            post_mul[app] = 1

            constraints.append(np.ones((1, 24)) @ (t @ post_mul * self.e[app, 0]) - self.d[app, 0] == 0)

        # ESD constraints
        if self.esd and not self.sp:
            # ESD variable to optimiz
            qss = cp.Variable()
            xnb = cp.Variable(shape=(24, 1))

            # EDS constraints
            m_ns = 13500
            m_c = 5000
            m_d = -5000
            ltril_24 = np.tril(np.ones((24, 24)), k=0)
            l_24 = np.ones((24, 1))
            l_24[-1] = 0
            esd_1 = ltril_24 @ xnb + qss * l_24 >= 0
            esd_2 = ltril_24 @ xnb + qss * l_24 <= m_ns
            esd_3 = qss >= 0
            esd_4 = qss <= m_ns
            esd_5 = xnb >= m_d
            esd_6 = xnb <= m_c
            esd_7 = self.ns + t @ self.e + xnb >= 0
            constraints.extend([esd_1, esd_2, esd_3, esd_4, esd_5, esd_6, esd_7])

            # Total energy vector equation
            # ns = nonshiftable hourly vector
            # t = matrix indicating appliance on/off
            # xnb = battery hourly vector
            # xn = hourly vector for one particular consumer
            # lm = hourly vector for all other consumer
            # x = total hourly vector for entire grid
            xn = self.ns + t @ self.e + xnb
            x = xn + lm
        elif self.sp:
            # SP constraints (Which is ESD constraints, but SP can be a different model of battery with different params
            # ESD variable to optimize
            qss = cp.Variable()
            xnb = cp.Variable(shape=(24, 1))

            # EDS constraints
            m_ns = 13500
            m_c = 5000
            m_d = -5000
            ltril_24 = np.tril(np.ones((24, 24)), k=0)
            l_24 = np.ones((24, 1))
            l_24[-1] = 0
            esd_1 = ltril_24 @ xnb + qss * l_24 >= 0
            esd_2 = ltril_24 @ xnb + qss * l_24 <= m_ns
            esd_3 = qss >= 0
            esd_4 = qss <= m_ns
            esd_5 = xnb >= m_d
            esd_6 = xnb <= m_c
            esd_7 = self.ns + t @ self.e + xnb - self.xp >= 0
            constraints.extend([esd_1, esd_2, esd_3, esd_4, esd_5, esd_6, esd_7])

            # Total energy vector equation
            # xp = solar power generation hourly vector if self.sp = true
            xn = self.ns + t @ self.e + xnb - self.xp
            x = xn + lm
        else:
            # Total energy vector equation. For household with no ESD and no SP
            xn = self.ns + t @ self.e
            x = xn + lm

        # Obj Function
        xt = self.sigma * x + (1 - self.sigma) * self.xt_no_ecs
        objective = cp.Minimize(
            np.ones((1, 24)) @ (self.coef0 + self.coef1 * xt + self.coef2 * (xt ** 2)))
        prob = cp.Problem(objective, constraints)
        # Solving
        prob.solve(solver='GUROBI')
        total_cost = prob.value * self.k
        t_opt = np.abs(t.value)
        if self.sp:
            qss_opt = qss.value
            xnb_opt = xnb.value
            xn = self.ns + t_opt @ self.e + xnb_opt - self.xp
            self.qss_store.append(qss_opt)
            self.xnb_store.append(xnb_opt)
        elif self.esd:
            qss_opt = qss.value
            xnb_opt = xnb.value
            xn = self.ns + t_opt @ self.e + xnb_opt
            self.qss_store.append(qss_opt)
            self.xnb_store.append(xnb_opt)
        else:
            xn = self.ns + t_opt @ self.e

        x = self.sigma * (xn + lm) + (1-self.sigma)*self.xt_no_ecs
        par = np.max(x.reshape(-1)) / np.average(x.reshape(-1))
        bn = np.sum(xn.reshape(-1)) / np.sum(x.reshape(-1)) * total_cost
        self.xn_store.append(xn)
        self.t_store.append(t_opt)
        self.bn_store.append(bn.item())

        # Printing statements
        if verbose > 0:
            print(
                "Solving for: Consumer {}. ESD Assigned: {}."
                " Iteration: {}. Prob is DCP: {}".format(self.c_idx, self.esd, iteration, prob.is_dcp()))
            print("Total Cost: ", total_cost)
            print('Current Bill to Consumer{}: {}'.format(self.c_idx, bn))
            if verbose == 2:
                print('Optimal xn:\n{}'.format(xn))
                print('Optimal t: \n{}'.format(t_opt))
                if self.sp:
                    print('SP ESD vector: \n{}'.format(xnb_opt))
                    print('SP Battery start state qss = {}'.format(qss_opt))
                elif self.esd:
                    print('ESD vector: \n{}'.format(xnb_opt))
                    print('Battery start state qss = {}'.format(qss_opt))

                else:
                    print('Consumer does not have ESD nor SP assigned')

        return xn, par, total_cost

    def optimize_continuous_with_efficiency(self, lm, efficiency, iteration=-1, verbose=1):
        # ECV variable to optimize
        t = cp.Variable(shape=(24, self.app_count))

        # ECV constraints
        ecv_1 = t - self.a <= 0
        ecv_2 = t - self.b <= 0
        ecv_3 = t >= 0
        ecv_4 = t <= 1
        constraints = [ecv_1, ecv_2, ecv_3, ecv_4]
        for app in range(self.app_count):
            post_mul = np.zeros((self.app_count, 1))
            post_mul[app] = 1

            constraints.append(np.ones((1, 24)) @ (t @ post_mul * self.e[app, 0]) - self.d[app, 0] == 0)

        # ESD constraints
        if self.esd and not self.sp:
            # ESD variable to optimiz
            qss = cp.Variable()
            xnb_charge = cp.Variable(shape=(24, 1))
            xnb_discharge = cp.Variable(shape=(24, 1))

            # EDS constraints
            m_ns = 13500
            m_c = 5000
            m_d = -5000
            ltril_24 = np.tril(np.ones((24, 24)), k=0)
            l_24 = np.ones((24, 1))
            l_24[-1] = 0
            esd_1 = ltril_24 @ xnb_charge + ltril_24 @ xnb_discharge + qss * l_24 >= 0
            esd_2 = ltril_24 @ xnb_charge + ltril_24 @ xnb_discharge + qss * l_24 <= m_ns
            esd_3 = qss >= 0
            esd_4 = qss <= m_ns
            esd_5 = xnb_discharge >= m_d
            esd_6 = 0 >= xnb_discharge
            esd_7 = xnb_charge <= m_c
            esd_8 =0 <= xnb_charge
            esd_9 = self.ns + t @ self.e + xnb_charge/efficiency + xnb_discharge*efficiency >= 0

            constraints.extend([esd_1, esd_2, esd_3, esd_4, esd_5, esd_6, esd_7, esd_8, esd_9])

            # Total energy vector equation
            # ns = nonshiftable hourly vector
            # t = matrix indicating appliance on/off
            # xnb = battery hourly vector
            # xn = hourly vector for one particular consumer
            # lm = hourly vector for all other consumer
            # x = total hourly vector for entire grid
            xn = self.ns + t @ self.e + xnb_charge/efficiency + xnb_discharge*efficiency
            x = xn + lm
        elif self.sp:
            # SP constraints (Which is ESD constraints, but SP can be a different model of battery with different params
            # ESD variable to optimize
            qss = cp.Variable()
            xnb_charge = cp.Variable(shape=(24, 1))
            xnb_discharge = cp.Variable(shape=(24, 1))

            # EDS constraints
            m_ns = 13500
            m_c = 5000
            m_d = -5000
            ltril_24 = np.tril(np.ones((24, 24)), k=0)
            l_24 = np.ones((24, 1))
            l_24[-1] = 0
            esd_1 = ltril_24 @ xnb_charge + ltril_24 @ xnb_discharge + qss * l_24 >= 0
            esd_2 = ltril_24 @ xnb_charge + ltril_24 @ xnb_discharge + qss * l_24 <= m_ns
            esd_3 = qss >= 0
            esd_4 = qss <= m_ns
            esd_5 = xnb_discharge >= m_d
            esd_6 = 0 >= xnb_discharge
            esd_7 = xnb_charge <= m_c
            esd_8 =0 <= xnb_charge
            esd_9 = self.ns + t @ self.e + xnb_charge/efficiency + xnb_discharge*efficiency >= 0

            constraints.extend([esd_1, esd_2, esd_3, esd_4, esd_5, esd_6, esd_7, esd_8, esd_9])

            # Total energy vector equation
            # xp = solar power generation hourly vector if self.sp = true
            xn = self.ns + t @ self.e + xnb_charge/efficiency + xnb_discharge*efficiency - self.xp
            x = xn + lm
        else:
            # Total energy vector equation. For household with no ESD and no SP
            xn = self.ns + t @ self.e
            x = xn + lm

        # Obj Function
        xt = self.sigma * x + (1 - self.sigma) * self.xt_no_ecs
        objective = cp.Minimize(
            np.ones((1, 24)) @ (self.coef0 + self.coef1 * xt + self.coef2 * (xt ** 2)))
        prob = cp.Problem(objective, constraints)
        # Solving
        prob.solve(solver='GUROBI')
        total_cost = prob.value * self.k
        t_opt = np.abs(t.value)
        if self.sp:
            qss_opt = qss.value
            xnb_charge_opt = xnb_charge.value
            xnb_discharge_opt = xnb_discharge.value
            xn = self.ns + t_opt @ self.e + xnb_charge_opt/efficiency + xnb_discharge_opt*efficiency - self.xp
            self.qss_store.append(qss_opt)
            self.xnb_store.append(xnb_charge_opt+xnb_discharge_opt)
        elif self.esd:
            qss_opt = qss.value
            xnb_charge_opt = xnb_charge.value
            xnb_discharge_opt = xnb_discharge.value
            xn = self.ns + t_opt @ self.e + xnb_charge_opt/efficiency + xnb_discharge_opt*efficiency
            self.qss_store.append(qss_opt)
            self.xnb_store.append(xnb_charge_opt+xnb_discharge_opt)
        else:
            xn = self.ns + t_opt @ self.e

        x = self.sigma * (xn + lm) + (1-self.sigma)*self.xt_no_ecs
        par = np.max(x.reshape(-1)) / np.average(x.reshape(-1))
        bn = np.sum(xn.reshape(-1)) / np.sum(x.reshape(-1)) * total_cost
        self.xn_store.append(xn)
        self.t_store.append(t_opt)
        self.bn_store.append(bn.item())

        # Printing statements
        if verbose > 0:
            print(
                "Solving for: Consumer {}. ESD Assigned: {}."
                " Iteration: {}. Prob is DCP: {}".format(self.c_idx, self.esd, iteration, prob.is_dcp()))
            print("Total Cost: ", total_cost)
            print('Current Bill to Consumer{}: {}'.format(self.c_idx, bn))
            if verbose == 2:
                print('Optimal xn:\n{}'.format(xn))
                print('Optimal t: \n{}'.format(t_opt))
                if self.sp:
                    print('SP ESD vector: \n{}'.format(xnb_charge_opt+xnb_discharge_opt))
                    print('SP Battery start state qss = {}'.format(qss_opt))
                elif self.esd:
                    print('ESD vector: \n{}'.format(xnb_charge_opt+xnb_discharge_opt))
                    print('Battery start state qss = {}'.format(qss_opt))

                else:
                    print('Consumer does not have ESD nor SP assigned')

        return xn, par, total_cost
