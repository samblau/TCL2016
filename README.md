# TCL2016

	Introduction

Redfield / TCL2 Exciton Dynamics Code written by Samuel M. Blau, based on a template written by John Parkhill, between 2012 and 2016 in the research group of Professor Alan Aspuru-Guzik at Harvard University. Please acknowledge Blau, Parkhill, and Aspuru-Guzik in any publications which include or reference any data produced or processed by this package. Please also feel free to contact Sam with any questions or issues at samblau1@gmail.com. 

This file will serve as a manual for the code, providing general information for interested users as well as specifying the input parameters that are used in the TCL.input file. TCL.input is the crucial file for controlling the function of the code, although the code is run with the command "python Main.py" in the TCL2016 directory. 

The heart of the code can be found in TightBindingTCL2.py, while the details of the propagation can be found in Propagator.py and Propagator_TD.py. Main.py acts as the front end for execution, while TensorNumerics.py sets up a data structure which holds crucial information that can be referenced from any of the other files using a Params._____ call. SpectralAnalysis.py handles many of the plotting routines, while LooseAdditions.py and NumGrad.py hold a few other tidbits that are still in use by the code.

Note that this framework was originally written by John Parkhill and Thomas Markovich for their paper titled "A correlated-polaron electronic propagator: Open electronic dynamics beyond the Born-Oppenheimer approximation" published in 2012. I've made sure to keep TightBindingTCL2.py nice and clean, but many of the files mentioned above remain messy and often contain chunks of code that are no longer in use. I hope to improve this in time, as well as to add additional comments throughout the code and to continue to add information to this manual. For now let's move onto the user-adjustable parameters found in TCL.input.




	TCL.input parameters

name: Specifies the name of the system which will be used in the "Figures" and "Output" directories that are generated when the code is run. Any string is a valid input.

sites: Specifies the number of physical sites contained in the system. Input should be an integer greater than 1. 

markov: Specifies if the Markovian approximation will be used (True or true) or will not (False or false). In other words, True = Redfield, False = TCL2. Some of the following parameters take identical inputs and will be denoted as having a T/F input.

secular: Specifies if the secular approximation will be used with a T/F input. The secular approximation prevents transfer between populations and coherences. This can be crucial for preventing exciton populations from falling bellow zero depending on the specifics of your Hamiltonian and spectral density. 

save_dens: Specifies if the trajectories of density matrices will be stored and printed with a T/F input.

duration: Specifies the length of propagation in picoseconds. Any positive integer or float is a valid input, though a recommended default value is 1 picosecond.

step: Specifies the length of the time-step of propagation in picoseconds. A recommended default value is 0.00015 picoseconds.

sd_range: Specifies the maximum frequency, in wavenumber, for the plotting of the spectral densities. Any positive integer or float is a valid input. 

coh_type: Specifies whether we are simulating a coherence between physical sites (0) or between excitonic states (1). Excitonic states are the eigenvectors of the site-basis Hamiltonian. 0 and 1 are the only valid inputs.

pop: Specifies the physical site on which the exciton is initially localized for population dynamics. Accepted inputs are integers between 1 and the number of physical sites. 

coh: Takes two integer inputs that specify the indices of the two sites between which a coherence is propagated. These can be physical sites or excitonic states depending on the coh_type input. 

temp: Specifies the temperature of the simulation in Kelvin.

dipole: Specifies a vector of average dipole amplitudes squared in units of debye^2. This should contain entries equal to the number of physical sites plus one, and the first entry should always be 0.0. Note that the dipole input used in all the example TCL.input files is essentially just the vector used for FMO. This explains why FMO is the only example with an absorbance spectrum that actually looks realistic. I hope to remedy this in the near future. 

SD_dir: Specifies the directory that contains the spectral density inputs which will be discussed in more detail below. 

ham: Specifies an element of the excitonic Hamiltonian. Entries should be of the following form: siteA(integer) siteB(integer) Hvalue(float). Setting the [A,B] element also automatically sets the [B,A] element. 




	Examples provided and more information on spectral densities

Sample TCL.input files have been provided for FMO (TCL.input), PE545 (TCL.input_545), and PC645 (TCL.input_645). Spectral densities accompanying these input files can be found in FMOSDs, 545SDs, and 645SDs_coker and 645SDs_sam respectively. Each of these directories contains csv files for each site contained in the system, and the files are numbered 1.csv, 2.csv, etc. Each csv file contains rows equal to the number of Drude-Lorentz peaks, and each row specifies the SD parameters as follows: gamma (cm^-1), lambda (cm^-1), Omega (cm^-1). The SDs contained in FMOSDs were fitted by Christoph Kreisbeck to the low-frequency region of the spectral densities reported by Shim et. al in the paper titled "Atomistic Study of the Long-Lived Quantum Coherences in the Fenna-Matthews-Olson Complex" published in 2012. The SDs contained in 545SDs are very basic and contain one peak each with gamma = lambda = 50 cm^-1. The SDs contained in 645SDs_coker are nearly identical to those used by Huo and Coker in the 2011 paper titled "Theoretical Study of Coherent Excitation Energy Transfer in Cryptophyte Phycocyanin 645 at Physiological Temperature". The SDs contained in 645SDs_sam are preliminary fits to the low-frequency region of some of my PC645 atomistic spectral densities recently obtained from QM/MM / TDDFT trajectories. 

A brief aside on the form of the Drude-Lorentz spectral density: In this code we use the form J[w] = 2*l*g*w*(1/(g^2+(w-O)^2) + 1/(g^2+(w+O)^2)) where g = gamma, l = lambda, and O = Omega. Note that this is actually J[w]/hbar, which is the convention throughout the field. Note also that in some cases you may see a similar equation which also contains a factor of beta*(1/sqrt(6*pi)) out front. This has been explicitly rolled into our lambda, simplifying our functional form and allowing lambda to have units of [energy] instead of [energy^2].




	Outputs generated

When you run the code two directories are generated that contain figures and outputs that the user may want. These directories are named using the "name" input as well as the exact time that the code was run. This should prevent any results from ever being overwritten, and should ideally allow the user to recreate any set of results at a later time. Many of the outputs are pretty self-explanatory, such as the populations plot, but a few require a bit more explanation. The SD#.png plots are clearly of the spectral densities of each site. Note, however, that each plot is titled with its RE, reorganization energy, and TRE, thermal reorganization energy. The difference between them is that the TRE is only integrated up to a frequency equal to 1/beta, aka it only covers the thermally accessible range. NormOfState.png tracks the total exciton population throughout propagation. If this is changing in any range beyond machine precision then you have a big problem and should contact Sam immediately. Markovian.png should always be zero if you are using the Markovian approximation. If you are not then this plot should gradually increase over the propagation after starting at zero. This keeps track of each peak in each site spectral density reaching a Markovian limit, at which point the calculations done for that peak at every step can be turned off and the Markovian values can be referenced. This is one of the crucial speedups for the TCL2 code. Users should expect to see a large increase in this plot at around 0.25 ps as many of the peaks reach their Markovian limit. This in turn causes the expected time remaining in the calculation to drop precipitously at this point since a large percentage of the time-consuming calculations performed at each step are no longer necessary. 

I don't really want to get into a big discussion of the Spec____.png plots at this point. They are basically just the various parts of the absorbance spectrum obtained from propagating the dipole and then that absorbance spectrum with various Gaussians rolled over it. Almost all of the code that generates these plots was written by John Parkhill and I've left it in simply because it hasn't yet broken. At some point I hope to dramatically improve the spectroscopic aspect of the code but at present it is definitely a weakness.

One last note for this section - while TCL2 is less approximate than Redfield since it incorporates non-Markovian effects, don't expect to see massively different results between the two theories. Non-Markovian effects are limited to short times, after which the Gammas calculated by Redfield and TCL2 will be identical. With TCL2 you may see additional oscillations and can see some differences in rapid transfer rates, but if you obvserve more substantial changes please refer them to Sam for additional investigation. 




	Speedup strategies

Besides turning off various SD peaks as they reach their Markovian limit, we employ three other main strategies to improve the performance of our TCL2 code. First let's discuss the use of scipy.weave, which can be seen in TightBindingTCL2.py lines 181-205, 425-465, 417-457, and 586-623. Python is notoriously bad at efficiently performing deeply nested loops. Unfortunately, TCL2 requires us to construct four-index Gamma tensors by contracting over two variables, yielding six-fold nested loops at the heart of our algorithm. Thankfully, Scipy's weave functionality allows us to briefly jump into C just for these sections by writing snippets of C code directly into our python file. These snippets are compiled at run-time and allow us to obtain speedups of roughly 300x with minimal writing of unwieldy C code.

The next strategy that we use involves precomputation of all the terms that are necessary for the construction of our Gamma tensors which are not time-dependent. These can be seen in TightBindingTCL2.py lines 228 - 284. Since these terms are all time-independent we can simply evaluate them all once at the beginning of the calculation and then simply reference them during propagation. These make the code significantly harder to read but make a huge performance difference. 

Finally, the last strategy that we use to significantly speed up our code is to link against a fortran implementation of the hypergeometric 2F1 special function. This function is at the heart of our Gamma construction and it just so happens that someone has written an almost 2000-line fortran implementation that is far faster than anything found in numpy or scipy while also being far more numerically stable. It also allows for complex inputs, which is absolutely necessary for this code and is missing from the best numpy/scipy implementations. The actual fortran code can be found in the hyp_2f1 directory, and the files that are responsible for the linking can be found in the base directory and are named hyp_2f1_module.mod and hyp_2f1.so. 




	Initial density matrix clarification

I want to end by providing a bit of clarification about how the code functions and what our initial states are. The slow step in the TCL2 code is the construction of the Gamma tensors at each step in the propagation. Thus, in order to be efficient we simultaneously propagate three density matrices when the code is executed - one for a population, one for a coherence, and one for the dipole moment / absorbance spectrum. If you have a five site system and your TCL.input contains the line "pop 2" then your initial population density matrix looks like this:

[[0,0,0,0,0],
 [0,1,0,0,0],
 [0,0,0,0,0],
 [0,0,0,0,0],
 [0,0,0,0,0]]

aka you have an exciton localized on physical site 2 which will then be propagated forward in time. Note that the propagation actually takes place in the exciton basis, which is why in TightBindingTCL2.py line 119 you see that this density matrix undergoes a change of basis. Moving onto the coherence initial density matrix, let's say you once again have a five site system and you want to observe the coherence between physical sites 3 and 4. We know from the parameters section that you would include the following two lines in your TCL.input file:

coh_type 0
coh 3 4

This would then give you the following initial coherence "density matrix":

[[0,0,0,0,0],
 [0,0,0,0,0],
 [0,0,0,1,0],
 [0,0,1,0,0],
 [0,0,0,0,0]]

which would then be converted into the exciton basis as seen on line 122 in TightBindingTCL2.py. You'll notice that I put density matrix in quotes above. That's because technically this is not a density matrix since its trace is not equal to one. This is how I was taught, and I've never fully understood it. I intend to improve my understanding in the near future and update this document with what I learn. My understanding of the initial density matrix for the calculation of the absorbance spectrum is even shakier, so for now I'm just going to skip that entirely, but I hope to remedy this in the near future as well.




Thanks so much for using the code!

Sincerely,

Sam Blau

