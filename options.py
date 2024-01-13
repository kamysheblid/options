import sympy
from sympy import exp, oo, log, exp, sqrt, pi
from sympy.abc import z,q,r,T,S,K,sigma,delta, gamma,rho,theta
vega = sympy.symbols('vega')

put = 'put'
call = 'call'

N = lambda x: (1/(2*pi)**0.5 * exp(-0.5*z**2)).integrate((z, -oo, x))
d1 = (log(S/K) + (r - q + sigma**2 / 2) * (T/365)) / (sigma*(T/365)**0.5)
d2 = (log(S/K) + (r - q - sigma**2 / 2) * (T/365)) / (sigma*(T/365)**0.5)

calleq = S*exp(-q*T/365)*N(d1) - K*exp(-r*T/365)*N(d2)
puteq = K*exp(-r*T/365)*N(-d2) - S*exp(-q*T/365)*N(-d1)
callfn = sympy.lambdify((S,K,T,sigma,r,q), calleq)
putfn = sympy.lambdify((S,K,T,sigma,r,q), puteq)

deltaeq = sympy.diff(calleq, S)
gammaeq = sympy.diff(deltaeq, S)
thetaeq = sympy.diff(calleq, T)
vegaeq = sympy.diff(calleq, sigma)
rhoeq = sympy.diff(calleq, r)
deltafn = sympy.lambdify((S,K,T,sigma,r,q), deltaeq)
gammafn = sympy.lambdify((S,K,T,sigma,r,q), gammaeq)
thetafn = sympy.lambdify((S,K,T,sigma,r,q), thetaeq)
vegafn = sympy.lambdify((S,K,T,sigma,r,q), vegaeq)
rhofn = sympy.lambdify((S,K,T,sigma,r,q), rhoeq)

class BlackSholes:
    """Option Class. Give it PRICE STRIKE TIME VOL RATE DIVIDEND and optionally OPTION_TYPE."""
    def __init__(self, option_type = None):
        if option_type and isinstance(option_type, str) and option_type.lower() == put:
            self.option_type = 'put'
            self.optioneq = puteq
            self.optionfn = putfn
        else:
            self.option_type = call
            self.optioneq = calleq
            self.optionfn = callfn

        self.deltaeq = sympy.diff(self.optioneq, S)
        self.gammaeq = sympy.diff(self.deltaeq, S)
        self.thetaeq = sympy.diff(self.optioneq, T)
        self.vegaeq = sympy.diff(self.optioneq, sigma)
        self.rhoeq = sympy.diff(self.optioneq, r)

        self.deltafn = sympy.lambdify((S,K,T,sigma,r,q), self.deltaeq)
        self.gammafn = sympy.lambdify((S,K,T,sigma,r,q), self.gammaeq)
        self.thetafn = sympy.lambdify((S,K,T,sigma,r,q), self.thetaeq)
        self.vegafn = sympy.lambdify((S,K,T,sigma,r,q), self.vegaeq)
        self.rhofn = sympy.lambdify((S,K,T,sigma,r,q), self.rhoeq)
        return
    # def __eq__(self, other_option):
    #     if other_option and self and self.price == other_option.price and self.strike == other_option.strike and self.time == other_option.time and self.vol == other_option.vol and self.rate == other_option.rate and self.dividend == other_option.dividend and self.option_fn == other_option.option_fn:
    #         return True
    #     return False
    def greeks(self, *args):
        return {delta: self.deltafn(*args),
                gamma: self.gammafn(*args),
                theta: self.thetafn(*args),
                vega: self.vegafn(*args),
                rho: self.rhofn(*args)}

    def __repr__(self):
        #<__main__.Option object at 0x7f7dd5ab9050>
        # greeks_str = f"delta={self.delta:.2f} gamma={self.gamma:.6f} theta={self.theta:.2f} vega={self.vega:.2f} rho={self.rho:.2f}"
        # return f"<Option object Price={self.price} Strike={self.strike} Time={self.time} Vol={self.vol} Rate={self.rate} Dividend={self.dividend} Type={self.option_type} Option Price={self.option_price:.2f} Greeks={greeks_str}>"
        # return f"<Option object Price={self.price} Strike={self.strike} Time={self.time} Vol={self.vol} Rate={self.rate} Dividend={self.dividend} Type={self.option_type} Option Price={self.option_price:.2f}>"
        return f"<Option object Type={self.option_type}>"

class Put(BlackSholes):
    def __init__(self):
        super(Put, self).__init__('put')
        return

class Call(BlackSholes):
    def __init__(self):
        super(Call, self).__init__('call')
        return

class Option(BlackSholes):
    """Documentation for Option
    """
    def __init__(self, args):
        super(Option, self).__init__()
        self.args = args
