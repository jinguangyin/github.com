# -*- coding: utf-8 -*-
"""
Created on Fri Aug 10 22:35:01 2018

@author: 35002
"""

import random as rd
import numpy as np
import matplotlib.pyplot as plt

class Env1(object):
    def __init__(self):
        self.s=['lt','a','b','c','d','e','rt']
        
    def step(self,s):
        s_n=None
        r=None
        terminal=False
        n=rd.randint(1,10)
        if (s=='a')and(n<6):
            s_n='lt'
            r=0
            terminal=True
        if (s=='a')and(n>5):
            s_n='b'
            r=0
        if (s=='b')and(n<6):
            s_n='a'
            r=0
        if (s=='b')and(n>5):
            s_n='c'
            r=0
        if (s=='c')and(n<6):
            s_n='b'
            r=0
        if (s=='c')and(n>5):
            s_n='d'
            r=0
        if (s=='d')and(n<6):
            s_n='c'
            r=0
        if (s=='d')and(n>5):
            s_n='e'
            r=0
        if (s=='e')and(n<6):
            s_n='d'
            r=0
        if (s=='e')and(n>5):
            s_n='rt'
            r=0
            terminal=True

        return s_n,r,terminal
    
class Env2(object):
    def __init__(self):
        self.s=['a','b','c','d','e']
        
    def step(self,s,a):
        s_n=None
        r=None
        terminal=False
        if (s=='a')and(a==0):
            s_n='a'
            r=-1
            terminal=True
        if (s=='a')and(a==1):
            s_n='b'
            r=0
        if (s=='b')and(a==0):
            s_n='a'
            r=0
        if (s=='b')and(a==1):
            s_n='c'
            r=0
        if (s=='c')and(a==0):
            s_n='b'
            r=0
        if (s=='c')and(a==1):
            s_n='d'
            r=0
        if (s=='d')and(a==0):
            s_n='c'
            r=0
        if (s=='d')and(a==1):
            s_n='e'
            r=0
        if (s=='e')and(a==0):
            s_n='d'
            r=0
        if (s=='e')and(a==1):
            s_n='e'
            r=1
            terminal=True

        return s_n,r,terminal
    
def tdn():
    env=Env1()
    v=[0,0,0,0,0,0,1]
    #tran=[]
    n=3
    for k in range(10):
        s0='c'
        tra=[]
        g=0
        t=0
        T = float('inf')
        while True:
            t=t+1
            s_n,r,terminal=env.step(s0)
            tra.append([s_n,r])
            if (s_n=='rt')or(s_n=='lt'):
                T=t
            ut=t-n
            if ut>0:
                g=0
                for i in range(ut + 1, min(T, ut + n) ):
                    g=g+tra[i][1]
                if ut + n <= T:
                    g=g+v[env.s.index(tra[ut+n][0])]
                su=tra[ut][0]
                if (su!='rt')or(su!='lt'):
                    v[env.s.index(tra[ut][0])]=v[env.s.index(tra[ut][0])]+0.1*(g-v[env.s.index(tra[ut][0])])
                if ut==T-1:
                    break
            s0=s_n
            
if __name__ == "__main__":
    env=Env2()
    q=np.zeros((2,5))
    #q[1,4]=1
    #q[0,0]=-1
    al=[0,1]
    e=np.zeros((2,5))
    for k in range(100):
         e=np.zeros((2,5))
         s0='c'
         a0=rd.choice(al)
         while True:
             s_n,r,terminal=env.step(s0,a0)
             rn=rd.randint(1,10)
             if rn==1:
                 a_n=rd.choice(al)
             else:
                 a_n=np.argmax(q[:,env.s.index(s_n)])
             d=r+q[a_n,env.s.index(s_n)]-q[a0,env.s.index(s0)]
             e[a0,env.s.index(s0)]=e[a0,env.s.index(s0)]+1
             for i in range(2):
                 for j in range(5):
                     q[i,j]=q[i,j]+0.1*d*e[i,j]
                     e[i,j]=0.5*e[i,j]
             s0=s_n
             a0=a_n
             if terminal:
                 break

            
def tdlam(la, alpha):
    env=Env1()
    v=[0,0,0,0,0,0,1]
    e=[0,0,0,0,0,0,0]
    for k in range(10):
        e=[0,0,0,0,0,0,0]
        s0='c'
        while True:
            s_n,r,terminal=env.step(s0)
            d=r+v[env.s.index(s_n)]-v[env.s.index(s0)]
            e[env.s.index(s0)]=e[env.s.index(s0)]+1
            for i in range(7):
                v[i]=v[i]+alpha*d*e[i]
                e[i]=la*e[i]
            s0=s_n
            if terminal:
                break
    return v

realStateValues=np.array([0,1/6,2/6,3/6,4/6,5/6,1])
def figure7_2():
    # truncate value for better display
    truncateValue = 0.55

    # all possible steps
    las =  np.arange(0, 1.1, 0.1)

    # all possible alphas
    alphas = np.arange(0, 1.1, 0.1)

    # each run has 10 episodes
    episodes = 10

    # perform 100 independent runs
    runs = 100

    # track the errors for each (step, alpha) combination
    errors = np.zeros((len(las), len(alphas)))
    for run in range(0, runs):
        for laInd, la in zip(range(len(las)), las):
            for alphaInd, alpha in zip(range(len(alphas)), alphas):
                print('run:', run, 'lambda:', la, 'alpha:', alpha)
                for ep in range(0, episodes):
                    v=np.array(tdlam(la, alpha))
                    # calculate the RMS error
                    errors[laInd, alphaInd] += np.sqrt(np.sum(np.power(v - realStateValues, 2)) / 7)
    # take average
    errors /= episodes * runs
    # truncate the error
    errors[errors > truncateValue] = truncateValue
    plt.figure()
    for i in range(0, len(las)):
        plt.plot(alphas, errors[i, :], label='lambda = ' + str(las[i]))
    plt.xlabel('alpha')
    plt.ylabel('RMS error')
    plt.legend()

#figure7_2()
#plt.show()