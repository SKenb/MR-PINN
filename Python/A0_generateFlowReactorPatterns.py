import numpy as np
import pickle

from PyFlow.FlowReactor import FlowReactor

def main():
    reactor = FlowReactor(length=5, diameter=.8*1e-3)
    reactor.setReactionNetworkCallback(lambda C, speciesIdx, T: -0.25 * C[0, :])
    print(reactor)


    timeVec = np.linspace(0, 20, 100)
    Cin = np.zeros((1, len(timeVec)))
    flowRate = 1
    temperature = 150

    Cin[0, 0:] = 0.75 
    Cin[0, 10:30] = 1.0 
    Cin[0, 50:60] = 0.5 
    Cin[0, 60:] = 0 

    tout, Cout, Cspatial = reactor.simulate(Cin=Cin, timeVec=timeVec, flowRate=flowRate, temperature=temperature, isVolumetricFlowRate=False)
    reactor.plot(Cin=Cin, CinTimeVec=timeVec, showPlot=True)

    # Save reactor outlet concentration and time vector to file
    with open("./Data/flowReactorData.pkl", "wb") as f:
        pickle.dump({
            "timeInVec": timeVec,
            "timeOutVec": tout,
            "Cout": Cout,
            "Cin": Cin,
            "Cspatial": Cspatial,
            "flowRate": flowRate,
            "temperature": temperature
        }, f)

# python -m examples.A0_SimpleExample 
if __name__ == "__main__":
    main()