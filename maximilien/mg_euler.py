import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as sci 

class RandomizedEulerMaruyama:

    def __init__(self, T, NumberTimeSteps, X0, NumberMonteCarlo, SemigroupParameter, a, b, H, PrecisionHaar, NPrecisionHaar):
        # Parameters assigned by the user
        self.T = T
        self.NumberTimeSteps = NumberTimeSteps
        self.X0 = X0
        self.NumberMonteCarlo = NumberMonteCarlo
        self.SemigroupParameter = SemigroupParameter
        self.a = a
        self.b = b
        self.H = H
        self.PrecisionHaar = PrecisionHaar
        # Paramenters computed or assigned by the class
        self.NPrecisionHaar = NPrecisionHaar-1 # NPrecisionHaar is given but for the object the value is computed -1
        self.time = np.linspace(0, self.T, self.NumberTimeSteps)
        self.step = self.time[1]
        self.drift = 0.
        self.truedrift = 0.
        self.beta = -self.H + 1
        self.time = np.linspace(0, self.T, self.NumberTimeSteps+1)
        self.FBM = self.generateFBM() # np.linspace(0,self.T,self.PrecisionHaar+1)**2 
        self.coeff0 = self.FBM[-1] - self.FBM[0]
        self.currentPrecision = NPrecisionHaar-1    
        self.updateTrueDerivativeCoefficients(self.FBM,NPrecisionHaar)
        self.updateDerivativeCoefficients(self.currentPrecision)
        #self.beta = self.H - 1
        print(self.coefficients)
        print(self.TrueCoefficients)
        print(self.semigroup(self.NumberTimeSteps))

    def evaluateTrueDrift(self,x,SemigroupParameter):
        return self.Haar(self.TrueCoefficients, x, SemigroupParameter)

    def evaluateDrift(self,x,SemigroupParameter):
        return self.Haar(self.coefficients, x, SemigroupParameter)

    def updateNumberTimeStep(self, new):
        self.NumberTimeSteps = new
        print('Semigroup : ', self.semigroup(self.NumberTimeSteps))

    def solve(self, BM):
        X = [np.tile(self.X0, (self.NumberMonteCarlo))]        
        trueX = [np.tile(self.X0, (self.NumberMonteCarlo))]
        for iteration in range(self.NumberTimeSteps):
            #noise = np.random.normal(size= self.NumberMonteCarlo)
            X.append(X[-1] + self.evaluateDrift((X[-1]-self.a)/(self.b-self.a), self.semigroup(self.NumberTimeSteps)) * self.step +  BM[iteration+1] - BM[iteration])
            #trueX.append(self.evaluateTrueDrift((trueX[-1]-self.a)/(self.b-self.a), self.semigroup(self.NumberTimeSteps)) * self.step + np.sqrt(self.step) * noise)
        X = np.stack(X)
        print(X.shape)
        #trueX = np.stack(trueX)
        '''
        plt.plot(self.time,X[:,0])
        plt.plot(self.time,trueX[:,0])
        plt.grid('both')
        plt.show()
        '''
        return X, trueX, np.max(np.mean(np.abs(X-trueX), axis = -1))
        

    def computeDerivativeCoefficients(self, f, Precisionf):
        j = np.arange(0,Precisionf)
        m = np.arange(0,2**(Precisionf-1))
        jm , mm = np.meshgrid(j, m, sparse=False, indexing='ij')
        
        f = np.concatenate([f, np.zeros(2**(2*Precisionf+1))]) #2
        coefficients = -2**jm * (f[(mm+1)*2**(Precisionf-jm)] - 2*f[(2*mm+1)*2**(Precisionf-1-jm)] + f[mm*2**(Precisionf-jm)])
        selectCoefficients = np.zeros((Precisionf,2**(Precisionf-1)), dtype=np.float32)
        #print(coefficients)
        for i in range(Precisionf):
            for j in range(2**(i)):
                selectCoefficients[i,j] = 1.
        #print(selectCoefficients)
        return coefficients * selectCoefficients

    def truncateCoeffficientsMatrix(self,Precision):
        self.coefficients = self.TrueCoefficients[:Precision+1,:2**(Precision)]
    
    def updateDerivativeCoefficients(self, Precision):
        self.truncateCoeffficientsMatrix(Precision)
        self.currentPrecision = Precision
        #self.coeff0 = f[-1] - f[0]
    
    def updateTrueDerivativeCoefficients(self, f, Precisionf):
        self.TrueCoefficients = self.computeDerivativeCoefficients(f, Precisionf)
        #self.coeff0 = f[-1] - f[0]

    def generateFBM(self):
        rand = np.random.normal(size=(self.PrecisionHaar,1)) #-1
        x = np.linspace(1/self.PrecisionHaar,1.,self.PrecisionHaar) #-1
        xv, yv = np.meshgrid(x, x, sparse=False, indexing='ij')
        correlationMatrix = 1/2 * (np.exp(2*self.H*np.log(np.abs(xv))) +np.exp(2*self.H*np.log(np.abs(yv))) - np.exp(2*self.H*np.log(np.abs(xv - yv))))

        FBM = np.matmul(np.linalg.cholesky(correlationMatrix), rand).reshape(self.PrecisionHaar,)#-1
        FBM = np.concatenate([np.zeros(1, dtype= np.float32),FBM])
        #plt.plot(np.linspace(self.a,self.b,precision), FBM)
        #plt.show()
        #print(correlationMatrix)
        return FBM

    def semigroup(self,Ntimesteps):
        return np.exp(-0.5  / (3/4  - self.beta*(1/2-self.beta-self.beta))* np.log(Ntimesteps))/10
    
    def N(self,Ntimesteps):
        return int(1 + np.ceil(-(-0.5  / (3/4  - self.beta*(1/2-self.beta-self.beta))* np.log(Ntimesteps))))

    def Haar(self, coefficients, x,SemigroupParameter):
        N = coefficients.shape[0]
        j = np.arange(0,N)
        m = np.arange(0,2**(N-1))
        jm , mm = np.meshgrid(j, m, sparse=False, indexing='ij')
        #values = np.zeros(x.shape)
        values = []
        #for i in range(x.shape[0]):
        #    for j in range(x.shape[1]):

        for elem in x:
            norm = sci.norm(loc = elem, scale = np.sqrt(SemigroupParameter)) 
            cdf = 2*norm.cdf((2*mm+1)/(2**(jm+1))) - norm.cdf((mm+1)/(2**(jm))) - norm.cdf((mm)/(2**(jm)))
            #if elem == 0. or elem == 1.:
                #print(2*norm.cdf((2*mm+1)/(2**(jm+1))), norm.cdf((mm+1)/(2**(jm))), norm.cdf((mm)/(2**(jm))))
                #print(cdf)
            #values[i,j] = self.coeff0 + np.sum(coefficients * cdf) 
            values.append(self.coeff0 + np.sum(coefficients * cdf)  )     #
        return np.array(values)

    def plotDerivativeApprox(self, Precision, SemigroupParameter):
        self.updateDerivativeCoefficients(Precision)
        x = np.linspace(0., 1.0, 2**(self.NPrecisionHaar+1)+1)
        data1 = self.FBM
        data2 = self.evaluateDrift(x, SemigroupParameter)

        fig, ax1 = plt.subplots()

        color = 'tab:red'
        plt.grid('both')
        ax1.set_xlabel(r'$x$')
        #ax1.set_ylabel('exp', color=color)
        line1, =ax1.plot(x, data1, color=color, label='Sample path of '+r'$B^H(x)$')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_ylim([-1.1*np.max(np.abs(data1)), 1.1*np.max(np.abs(data1))])
        
        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

        color = 'tab:blue'
        #ax2.set_ylabel('sin', color=color)  # we already handled the x-label with ax1
        line2, = ax2.plot(x, data2, '--',color=color, label='Derivative approximation')
        ax2.tick_params(axis='y', labelcolor=color)
        ax2.set_ylim([-1.1*np.max(np.abs(data2)), 1.1*np.max(np.abs(data2))])

        plt.title("N=%f, $\eta$=%f"%(Precision, SemigroupParameter))
        plt.legend((line1, line2), ('Sample path of '+r'$B^H(x)$','Derivative approximation'))

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        #plt.savefig('Derivative'+str(self.currentPrecision)+str(SemigroupParameter)+'.png')

