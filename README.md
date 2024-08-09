# Traveling Salesman Problem

### Problem Description

Briefly, for the traveling salesman problem, you randomly place $N$ cites in a square box. The problem is the following:

Given $N$ cities and the costs (distances) of traveling from any city to any other city, what is the cheapest (shortest) round-trip route that visits each city exactly once and then returns to the starting city?

The "energy" (objective function) $E$ is simply the total distance traveled. You begin with some initial guess for a particular route. You then will swap two city routes. This will change the energy. If the energy change is negative, you always accept the move. If it's positive, you accept the move with a Boltzmann factor probability, i.e., $exp(-E/T)$, where $T$ is a fictitious temperature. You go through all of the cities this way. You need to begin with large $T$ (meaning you will accept most moves independent of the energy change) and then you gradually lower the temperature after each equilibration at a particular temperature. This is the "cooling schedule." For simplicity, one might use the following cooling schedule:

------------------
    ------------------              
                     |----------------------
                                           |---------------------

-----------------

where the vertical axis is $T$ and the horizontal axis is time. The time interval before dropping in temperature is enough to "equilibrate" at that particular $T$. As $T$ gets smaller, you will accept less and less
uphill moves, until you achieve the "ground state" (the lowest energy). A criterion for determining when you are in the ground state is when the energy does not change after some number of iterations (which you can determine from experience). You should keep track of the lowest energy solutions for different initial
conditions and then determine the frequency of the lowest energy configurations. Begin with 5 cities to debug your program because in such an instance you can enumerate all possible solutions and determine the true ground state exactly. Once you are satisfied the program is working, try 25, 50 and 100
cities. Determine the fraction times you achieve what you believe to be the global minimum starting from 100 different initial conditions for 5, 25, 50 and 100 cities. You should also plot the distribution of final energies as a function of N. Finally, plot your solutions of the minimal path for 5, 25, 50 and 100 cities.
