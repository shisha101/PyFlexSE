class msd_model:
    def __init__(self, init_cond):

        # IC
        x0_t0 = init_cond[0]
        x1_t0 = init_cond[1]
        self.initial_cond = [x0_t0, x1_t0]

        # parameters
        mass = 1.0  # mass of cart
        sc = 1.0    # spring constant
        df = 0.5   # damping factor
        f = 5.0     # input force
        self.params = [mass, sc, df, f]

    def model_ode(self, x, t, args):
        x0 = x[0]
        x1 = x[1]
        mass, sc, df, f = self.params
        x0_dot = x1
        x1_dot = -sc*x0**2 -df*x1 + f
        X_dot = [x0_dot,x1_dot]
        return X_dot


    def print_params(self):
        print self.params

    def print_ic(self):
        print self.initial_cond
