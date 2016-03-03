import numpy, scipy
#from scipy import special
from numpy import array
from numpy import linalg
# standard crap
import __builtin__ 
import os, sys, time, math, re, random, cmath
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt, isnan
from LooseAdditions import * 
from NumGrad import * 
from TensorNumerics import *

EvPerAu = 27.2113
	   
def	GeneralizedFFT(Input,b=1.0,a=0.0): 
	dim = len(Input)
	t1 = numpy.fromfunction(lambda X,Y: 2.0*pi*complex(0.0,1.0)*b*(X-1.0)*(Y-1.0)/dim,shape=[dim,dim])
	t1 = numpy.exp(t1)
	return (pow(dim,-(1.0-a)/2.0)*numpy.tensordot(Input,t1,axes=([0],[0])))

# Pass a numpy ndarray 
# get a list of positions which "Stand out."
def LocalMaxima(AList, Tol = 0.2 ,NMax=10): 
	tore = []
	Z = abs(AList)
	tmean = numpy.mean(abs(Z))
	tstdev = numpy.std(abs(Z))
	a = numpy.diff(Z)
	FDiffDer = numpy.insert(a,0,a[0])
	from scipy import interpolate, optimize
	dZ = scipy.interpolate.interp1d(numpy.arange(len(FDiffDer)), FDiffDer, kind='linear')
	print "Finding LocalMaxima of List ... "
	print "Mean: ", tmean, " stdev: ", tstdev

	# find the 10 most important maxima
	N0 = 0
	for N in range(NMax): 
		for Nz in range(N0,len(Z)): 
			OuterBreak = False
			if (abs(Z[Nz]) > tmean+Tol*tstdev):
				# find interval where the derivative changes sign
				for Nz2 in range(Nz,len(Z)-1): 
					if ( numpy.sign(dZ(Nz))*numpy.sign(dZ(Nz2)) < 0 ): 
						Xm = scipy.optimize.brentq(dZ,Nz,Nz2)
						if (Z[int(Xm)] > tmean+Tol*tstdev): 
							tore.append(Xm)
						# N0 begins when it goes back under tolerance. 
						for Nz3 in range(int(Xm),len(Z)): 
							if (abs(Z[Nz3]) < tmean+Tol*tstdev): 
								N0 = int(Xm)
								break 
						N0 = min(N0,len(Z))
						OuterBreak = True
				if (not OuterBreak): 
					OuterBreak = True
			if (OuterBreak): 
				break 
	return list(set(tuple(map(lambda X: round(X,3),tore))))

def SpectralAnalysis(Arg_Data,Arg_SampleInterval,DesiredZoom = 1.0/30.0,Title = "Spec",DesiredMaximum=None,Smoothing = False): #add logic here to take into account the density of states from imaginary time propagation
	Data = Arg_Data.copy()	
	# By default, remove all the zero frequency information. 
	Arg_Data -= Arg_Data.mean()
	DataPts = len(Data)
	SampleInterval = Arg_SampleInterval
	if (pow(DataPts,2.0) > 1*1024*1024): 
		if Params.Parallel == False:
			print "Spectral analysis of very large dataset, downsampling"
		#KeepEvery = int(sqrt(pow(DataPts,2.0)/pow(3000,2.)))
		KeepEvery = max(int(sqrt(pow(DataPts,2.0)/pow(6000,2.))),1)
		if Params.Parallel == False:
			print " by factor of ", KeepEvery
		SampleInterval = Arg_SampleInterval*KeepEvery
		Data = Arg_Data[0::KeepEvery].copy()
		DataPts = len(Data) 

	AuPerWavenumber = 4.5563e-6

	Zoom = DesiredZoom
	if (DesiredMaximum != None): 
		Zoom = SampleInterval*DesiredMaximum*(1.0/pi)
		if Params.Parallel == False:
			print "Assigning Zoom", Zoom		

	Freqs = pi*(2.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)		
#	Freqs = pi*(1.0/SampleInterval)*Zoom*(numpy.arange(DataPts/2.0))/(DataPts)
	CplxStrengths = GeneralizedFFT(Data,-1.0*Zoom)
	if Params.Parallel == False:
		print "Generalized FFT result: ", numpy.sum(CplxStrengths*CplxStrengths.conj())
	
	MakeSimplePlot(CplxStrengths.real[0:3000],tit=Title+"RealStrengths")
	MakeSimplePlot(CplxStrengths.imag[0:3000],tit=Title+"ImStrengths")
	CplxStrengths = CplxStrengths[:len(Freqs)] # I bet the problem is the negative frequencies... 
	import scipy.special
	# Damp out the low frequency information. 
#	DampLow= numpy.vectorize(lambda X: scipy.special.erf(X/int(0.07*len(CplxStrengths))))
#	Damping = DampLow(numpy.arange(len(CplxStrengths)))
#	CplxStrengths = CplxStrengths*Damping
	
	numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/FFTStrengths',CplxStrengths,fmt='%.18e')	
	# Strengths = CplxStrengths.real
	Strengths = CplxStrengths.imag

	import matplotlib
	import matplotlib.pyplot as plt
	import matplotlib.font_manager as fnt
	matplotlib.rcParams['legend.fancybox'] = True	
	PlotFont = {'fontname':'Helvetica','fontsize':18,'weight':'bold'}
	LegendFont = fnt.FontProperties(family='Helvetica',size='17',weight='bold')

	

	fig = plt.figure()
	ax = fig.add_subplot(111)	
	Freqs = Params.globalscale + Freqs/AuPerWavenumber
	numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/Freqs',Freqs,fmt='%.18e')
	Strengths = Strengths#/AuPerWavenumber
	l1 = plt.plot( Freqs , Strengths,'k')

	plt.setp(l1,linewidth=2, color='r')
	plt.xlabel('Frequency(wavenumber)',fontsize = Params.LabelFontSize)
	plt.ylabel('Strength',fontsize = Params.LabelFontSize)
	plt.xlim(Params.globalscale,Params.upperlimit)
	plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+Title+"TransformedSignal")
	plt.clf()	

	if (Smoothing): 
		for DR in [10,20,30,40,50,60,70,80,90,100]: 
			DeRung = Smooth(numpy.abs(Strengths),(float(DR)/10000.0))
			numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/Derung'+str(DR),DeRung,fmt='%.18e')	
			fig = plt.figure()
			ax = fig.add_subplot(111)		
			l1= plt.plot( Freqs , abs(DeRung) )
			plt.xlim(Params.globalscale,Params.upperlimit)
			plt.ylim(0.0,abs(DeRung).max()*1.15)
			plt.setp(l1,linewidth=2, color='r')
			plt.xlabel('Frequency(cm^-1)',fontsize = Params.LabelFontSize)
			plt.ylabel('Strength',fontsize = Params.LabelFontSize)
			plt.legend(["2-TCL"],loc=2,prop={'size':Params.LegendFontSize})
			plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+Title+"Zoomed"+"_Derung"+str(DR)+"cent")
			plt.clf()	
	return 

def Smooth(ToSmooth,GaussianWidth=0.05): 
	BlurWidth = int(GaussianWidth*len(ToSmooth)) 
	if Params.Parallel == False:
		print BlurWidth
	if(BlurWidth*numpy.size(ToSmooth)/2 < 1.4 or BlurWidth == 2):
		return ToSmooth
	else:
		Cnv = numpy.blackman(BlurWidth)/(numpy.hanning(BlurWidth).sum())
		return numpy.convolve(ToSmooth,Cnv,mode='same')

