from mg_euler import *
import matplotlib.pyplot as plt
import scipy.stats as sct
import numpy as np
# Parameters
#H = 0.99 #3/4
T = 1
NumberMonteCarlo = 200 #1000 #200
NNumberTimeSteps = 8#10
NumberTimeSteps = 2**NNumberTimeSteps
NPrecisionHaar = NNumberTimeSteps+1
PrecisionHaar = 2**NPrecisionHaar
SemigroupParameter = 0.01
X0 =0.
a=-2
b=2
H = 0.85 #85

scheme = RandomizedEulerMaruyama(T, NumberTimeSteps, X0, NumberMonteCarlo, SemigroupParameter, a, b, H, PrecisionHaar, NPrecisionHaar)
scheme = RandomizedEulerMaruyama(T, NumberTimeSteps, X0, NumberMonteCarlo, SemigroupParameter, a, b, H, PrecisionHaar, 4)




'''
scheme.plotDerivativeApprox(NPrecisionHaar-1,1e-2)
scheme.plotDerivativeApprox(2,1e-2)
scheme.plotDerivativeApprox(NPrecisionHaar-1,1e-5)
scheme.plotDerivativeApprox(2,1e-5)
scheme.plotDerivativeApprox(NPrecisionHaar-1,1e-7)
scheme.plotDerivativeApprox(2,1e-7)
'''




'''
X, trueX, res = scheme.solve(BM)
results = [res]
print(results)
'''



'''
results=[]
for prec in range(NPrecisionHaar-2,-1,-1):
    scheme.updateDerivativeCoefficients(prec)
    X, trueX, res = scheme.solve()
    results.append(res)
    print(results)
results.reverse()
print(results)
#results= [0.1]
plt.figure()
plt.plot(np.arange(0,NPrecisionHaar-1),np.log(np.array(results)),'.')
plt.grid('both')
plt.xlabel(r'N')
plt.ylabel(r'$\log(\sup_t \mathbb{E}|X_t^{N_0,m_0} - X_t^{N,m_0}|)$')
plt.savefig(str(NPrecisionHaar-1)+'N'+str(NNumberTimeSteps)+'.png')
plt.show()
'''



print(H)
np.random.seed(2)
results=[]
ylist = []
increments = np.sqrt(T/(2**NNumberTimeSteps) ) * np.random.normal(size= (2**(NNumberTimeSteps-1),NumberMonteCarlo))
print(np.cumsum(increments,0))
BM = np.concatenate([np.zeros((1,NumberMonteCarlo)),np.cumsum(increments,0)])
#BM =np.cumsum(increments,0)
for prec in range(4,NNumberTimeSteps):
    scheme.updateNumberTimeStep(2**prec)
    #print(scheme.N(2**prec))
    #scheme.updateDerivativeCoefficients(prec) # -1 ? scheme.N(2**
    X, _, _ = scheme.solve(BM[::2**(NNumberTimeSteps-1-prec),:])
    Y = X[:,0] - BM[::2**(NNumberTimeSteps-1-prec),0]
    
    results.append(X)
    ylist.append(Y)
    #print(results)
    #np.savetxt(str(prec)+'.txt',X,delimiter=';')
#results.reverse()
#print(results)

plt.figure()
plt.plot(np.linspace(0,T,2**(NNumberTimeSteps-1)+1),results[-1][:,0],'-')
plt.plot(np.linspace(0,T,2**4+1),results[-1-NNumberTimeSteps+1+4][:,0],'-')
plt.grid('both')
#plt.xlabel(r'$\log_2(m)$')
#plt.ylabel(r'$\log_2(\sup_t \mathbb{E}|X_t^{N,m} - X_t^{N,2*m}|)$')
#plt.savefig(str(H)+'Res.png', bbox_inches = "tight")
#plt.savefig(str(H)+'Res.pdf', bbox_inches = "tight")
plt.show()

plt.figure()
plt.plot(np.linspace(0,T,2**(NNumberTimeSteps-1)+1),ylist[-1],'-')
plt.plot(np.linspace(0,T,2**4+1),ylist[-1-NNumberTimeSteps+1+4],'-')
plt.grid('both')
#plt.xlabel(r'$\log_2(m)$')
#plt.ylabel(r'$\log_2(\sup_t \mathbb{E}|X_t^{N,m} - X_t^{N,2*m}|)$')
#plt.savefig(str(H)+'X-W.png', bbox_inches = "tight")
#plt.savefig(str(H)+'X-W.pdf', bbox_inches = "tight")
plt.show()

logError = []
for prec in range(4,NNumberTimeSteps-1):
    nextRes = results[prec+1-4]
    logError.append(np.log(np.max(np.mean(np.abs(np.repeat(results[prec-4][1:,:],2, axis=0) - nextRes[1:,:]),axis = -1)))/np.log(2))
print(logError)
slope1, intercept1, r_value, p_value, std_err = sct.linregress(np.arange(4,NNumberTimeSteps-1),logError)
print(slope1)

logErrorTotal = []
for prec in range(4,NNumberTimeSteps-1):
    end = results[-1]
    #logErrorTotal.append(np.log(np.max(np.mean(np.abs(results[prec] - end[::2**(NNumberTimeSteps-1-prec),:]),axis = -1)))/np.log(2))
    logErrorTotal.append(np.log(np.max(np.mean(np.abs(np.repeat(results[prec-4][1:,:],2**(NNumberTimeSteps-1-prec), axis=0) - end[1:,:]),axis = -1)))/np.log(2))
print(logErrorTotal)
slope2, intercept2, r_value, p_value, std_err = sct.linregress(np.arange(4,NNumberTimeSteps-1),logErrorTotal)
print(slope2)
#results= [0.1]

plt.figure()
plt.plot(np.arange(4,NNumberTimeSteps-1),logError,'.')
plt.plot(np.arange(4,NNumberTimeSteps-1),slope1*np.arange(4,NNumberTimeSteps-1)+intercept1)
plt.grid('both')
plt.xlabel(r'$\log_2(m)$')
plt.ylabel(r'$\log_2(\sup_t \mathbb{E}|X_t^{N,m} - X_t^{N,2*m}|)$')
#plt.savefig(str(NPrecisionHaar-1)+'N'+str(H)+'H'+str(NNumberTimeSteps)+'Nt.png', bbox_inches = "tight")
#plt.savefig(str(NPrecisionHaar-1)+'N'+str(H)+'H'+str(NNumberTimeSteps)+'Nt.pdf', bbox_inches = "tight")
plt.show()

plt.figure()
plt.plot(np.arange(4,NNumberTimeSteps-1),logErrorTotal,'.')
plt.plot(np.arange(4,NNumberTimeSteps-1),slope2*np.arange(4,NNumberTimeSteps-1)+intercept2)
plt.grid('both')
plt.xlabel(r'$\log_2(m)$')
plt.ylabel(r'$\log_2(\sup_t \mathbb{E}|X_t^{N,m} - X_t^{N,m_0}|)$')
#plt.savefig('total'+str(NPrecisionHaar-1)+str(H)+'H'+'N'+str(NNumberTimeSteps)+'Nt.png', bbox_inches = "tight")
#plt.savefig('total'+str(NPrecisionHaar-1)+str(H)+'H'+'N'+str(NNumberTimeSteps)+'Nt.pdf', bbox_inches = "tight")
plt.show()

print(H)

print(scheme.N(2**4),scheme.N(2**6),scheme.N(2**8),scheme.N(2**10),scheme.N(2**12))
