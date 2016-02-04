from Propagator import * 
from Propagator_TD import * # Propagator for Time-Dependent Hamiltonians. 
from TensorNumerics import * 
from SpectralAnalysis import * 
from TightBindingTCL2 import *

output = sys.stdout

usage = "usage: Just run." 
if len(sys.argv) < 0:
       print usage
	   
#
# Main Routine
#
class ToyManyBody: 
	""" Main MB Class """
	def __init__(self): 
		ToProp = None
		if (Params.Parallel == True): 
			print 'Parallel not implemented! Defaulting to serial.'
		ToProp = TightBindingTCL()
		TheDynamics = Propagator_TD(ToProp)  
		TheDynamics.Propagate()

ToyManyBody = ToyManyBody()
