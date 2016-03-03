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
from TensorNumerics import * 
from SpectralAnalysis import * 
from NumGrad import * 
from Propagator import *

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
	   
#
# This does the exterior work of propagation for a time dependent Hamiltonian matrix. 
# d(Rho)/dt = H(t)Rho 
#

class Propagator_TD(Propagator): 
	def __init__(self,a_propagatable, proc_index=None):
		self.ToExp = None
		self.Propm = None
		self.TimesEvaluated = []


		self.proc_index = proc_index

		Propagator.__init__(self,a_propagatable)
				
		Params.DoCisDecomposition = False	
		self.Field = None 
		Params.DoBCT = Params.DoEntropies = False
		Params.DoBCT = False
		Params.DoEntropies = False
		Params.DoBCT = False
		# if Params.Parallel == False:
		# 	print "Ready For propagation... "
		# 	print "Initial Vector: ", self.VNow
		# 	print "Initial Norm: ", self.VNow.InnerProduct(self.VNow)
		return 
		
	def Propagate(self, aPipe=None): 

		start = time.time()

		Step=0

		popVNow = []

		if self.Field != None : 
			self.VNow.MultiplyScalar(0.000001)
				
		# Checks to make sure every process starts. 
		if (Params.AllDirections): 
			aPipe.send(None)

		if Params.Parallel == True:
			aPipe.send(None)

		import matplotlib.pyplot as plt

		if Params.save_dens:
			# f = open('./Output'+Params.SystemName+ Params.start_time+'/pop_mat','w')
			# f.write(str(self.VNow["pop"]) + '\n' + '\n')
			# f.close()
			# f = open('./Output'+Params.SystemName+ Params.start_time+'/coh_mat','w')
			# f.write(str(self.VNow["coh"]) + '\n' + '\n')
			# f.close()
			# f = open('./Output'+Params.SystemName+ Params.start_time+'/abs_mat','w')
			# f.write(str(self.VNow["abs"]) + '\n' + '\n')
			# f.close()
			mypop = []
			mypop.append(self.VNow["pop"])
			mycoh = []
			mycoh.append(self.VNow["coh"])
			myabs = []
			myabs.append(self.VNow["abs"])



			
		while (self.TNow < Params.TMax): 
			# returns an estimate of the energy. 
			if Params.Parallel == False:
				TimingStart = time.time()

			# Because of the variable timestep you can exhaust self.Dipoles. 
			# So resize if you come close to filling it. 
			if (len(self.Dipoles)-Step < 2000):
				if (len(self.Dipoles.shape)>1): 
					self.Dipoles.resize((len(self.Dipoles)+2000,3))
					self.Norms.resize(len(self.Norms)+2000)		
					self.Coherences.resize(len(self.Coherences)+2000)	
					self.Populations.resize(len(self.Norms)+2000,Params.dim-1)
					self.MarkovianRec.resize(len(self.Norms)+2000,Params.dim-1)
				else: 
					self.Dipoles.resize(len(self.Dipoles)+2000)
					self.Norms.resize(len(self.Norms)+2000)				
					self.Coherences.resize(len(self.Coherences)+2000)
					self.MarkovianRec.resize(len(self.Norms)+2000,Params.dim-1)	
			
			#self.RK45Step(self.VNow,self.TNow)
			self.RungeKuttaStep(self.VNow,self.TNow)

			if Params.save_dens:
				mypop.append(self.VNow["pop"])
				mycoh.append(self.VNow["coh"])
				myabs.append(self.VNow["abs"])
			# 	f = open('./Output'+Params.SystemName+ Params.start_time+'/pop_mat','a')
			# 	f.write(str(self.VNow["pop"]) + '\n' + '\n')
			# 	f.close()
			# 	f = open('./Output'+Params.SystemName+ Params.start_time+'/coh_mat','a')
			# 	f.write(str(self.VNow["coh"]) + '\n' + '\n')
			# 	f.close()
			# 	f = open('./Output'+Params.SystemName+ Params.start_time+'/abs_mat','a')
			# 	f.write(str(self.VNow["abs"]) + '\n' + '\n')
			# 	f.close()

			self.MarkovianRec[Step] = copy.deepcopy(Params.MarkovianRecord)
			# if Step%200 == 0:
			# 	xind = np.arange(Step)
			# 	fig = plt.figure()
			# 	plt.plot(xind,self.MarkovianRec[0:Step,0])
			# 	plt.hold()
			# 	for ii in range(1,7):
			# 		plt.plot(xind,self.MarkovianRec[0:Step,ii])
			# 	plt.legend(['1','2','3','4','5','6','7'])
			# 	plt.show()
			# 	plt.clf()

			# if Params.pop:
			# 	popVNow = np.dot(np.dot(Params.evecs,self.VNow["coh"]),Params.invevecs)
			popVNow = np.dot(np.dot(Params.evecs,self.VNow["pop"]),Params.invevecs)

			self.TimesEvaluated.append(self.TNow*Params.time_constant)
			# if (self.Field == None): 
			# 	# Only collect dipoles for fourier transform if the field is off. 
			# 	self.Dipoles[Step] = numpy.real(self.ShortTimePropagator.DipoleMoment(self.VNow,self.TNow))		
			
			self.Dipoles[Step] = numpy.real(self.ShortTimePropagator.DipoleMoment(self.VNow,self.TNow))
			if Params.coh_type == 1:
				self.Coherences[Step] = numpy.real(self.VNow["coh"][Params.coherences[0],Params.coherences[1]])
			if Params.coh_type == 0:
				self.Coherences[Step] = numpy.real(np.dot(np.dot(Params.evecs,self.VNow["coh"]),Params.invevecs)[Params.coherences[0],Params.coherences[1]])
			self.Norms[Step] = np.sum(np.diag(np.real(self.VNow["coh"]))) + np.sum(np.diag(np.real(self.VNow["pop"])))

			for ii in range(Params.dim-1):
				self.Populations[Step,ii] = numpy.real(popVNow[ii+1,ii+1])
				# else:
				# 	# self.Populations[Step,ii] = numpy.real(self.VNow["coh"][ii+1,ii+1])
				# 	self.Populations[Step,ii] = self.VNow["coh"][ii+1,ii+1]


			if Params.Parallel == False:
				TimingEnd = time.time()

			if Params.Parallel == False:
				#print abs(self.Populations[Step,:])
				# print self.Norms[Step]
				myprint = "T: " + str(self.TNow*Params.time_constant) + " ps  Mu: " + str(round(self.Dipoles[Step],4))
				for ii in range(Params.dim-1):
					myprint += "  |P" + str(ii+1) + "| " + str(round(self.Populations[Step,ii],4))
				myprint += "  Total " + str(round(np.sum(self.Populations[Step,:]),4)) + "  WallMinToGo: " + str(round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0))
				print myprint
				# print "T: ", self.TNow, " Mu : ", round(self.Dipoles[Step],4), " |P1| ", round(self.Populations[Step,0],4), " |P2| ", round(self.Populations[Step,1],4), " |P3| ", round(self.Populations[Step,2],4), " |P4| ", round(self.Populations[Step,3],4), " |P5| ", round(self.Populations[Step,4],4), " |P6| ", round(self.Populations[Step,5],4), " |P7| ", round(self.Populations[Step,6],4), " Total ", round(np.sum(self.Populations[Step,:]),4), " WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)
				#print "T: ", self.TNow, " Mu : ", round(self.Dipoles[Step],4), " |St|: ", self.Norms[Step], " WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)
				# else:
					#print "T: ", self.TNow, " Mu : ", round(self.Dipoles[Step],4), " |St| ", round(self.Norms[Step],4), " WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)
			# else:
			# 	print "WallMinToGo: ", round(((Params.TMax-self.TNow)/Params.TStep)*(TimingEnd-TimingStart)/60.0)

			
			
			Step += 1
		
		print "Propagation " + str(self.proc_index) + " Complete... Collecting Data. "

		if Params.save_dens:
			f = open('./Output'+Params.SystemName+ Params.start_time+'/pop_mat','w')
			f.write(str(mypop))
			f.close()
			f = open('./Output'+Params.SystemName+ Params.start_time+'/coh_mat','w')
			f.write(str(mycoh))
			f.close()
			f = open('./Output'+Params.SystemName+ Params.start_time+'/abs_mat','w')
			f.write(str(myabs))
			f.close()

		if Params.equil == True:
			#print self.VNow["pop"]
			#os.mkdir('equil_mats')
			f = open('./equil_mats/pop_mat_real','w')
			for ii in range(8):
				for jj in range(8):
					f.write(str(numpy.real(self.VNow["pop"][ii][jj])) + '\n')
			f.close()

			f = open('./equil_mats/pop_mat_imag','w')
			for ii in range(8):
				for jj in range(8):
					f.write(str(numpy.imag(self.VNow["pop"][ii][jj])) + '\n')
			f.close()

			f = open('./equil_mats/coh_mat_real','w')
			for ii in range(8):
				for jj in range(8):
					f.write(str(numpy.real(self.VNow["coh"][ii][jj])) + '\n')
			f.close()

			f = open('./equil_mats/coh_mat_imag','w')
			for ii in range(8):
				for jj in range(8):
					f.write(str(numpy.imag(self.VNow["coh"][ii][jj])) + '\n')
			f.close()

			f = open('./equil_mats/abs_mat_real','w')
			for ii in range(8):
				for jj in range(8):
					f.write(str(numpy.real(self.VNow["abs"][ii][jj])) + '\n')
			f.close()

			f = open('./equil_mats/abs_mat_imag','w')
			for ii in range(8):
				for jj in range(8):
					f.write(str(numpy.imag(self.VNow["abs"][ii][jj])) + '\n')
			f.close()

		end = time.time()

		print "It took " + str((end-start)/60.0) + " minutes"

		f = open('./Output'+Params.SystemName+ Params.start_time+'/RunParams.txt','a')
		f.write("It took " + str((end-start)/60.0) + " minutes" + '\n')
		f.close()

		import matplotlib
		import matplotlib.pyplot as plt
		import matplotlib.pylab as lab


		colors = ['blue','green','red','magenta','saddlebrown','orange','darkturquoise','yellowgreen','dodgerblue','darkolivegreen']
		styles = ['solid','dashed','dashdot','dotted','dashed','dashdot','dotted','dashed','dashdot','dotted']
		# print self.Populations[:,3]
		self.Populations.resize(len(self.TimesEvaluated),Params.dim-1)
		fig = plt.figure()
		#plt.plot(self.TimesEvaluated,self.Populations[:,3])
		fig.hold()
		if Params.vib_mult == 1:
			for ii in range(Params.dim-1):
				plt.plot(self.TimesEvaluated,self.Populations[:,ii],c=colors[ii],linewidth=2)
		else:
			for ii in range(Params.orig_dim-1):
				for jj in range(Params.vib_mult):
					plt.plot(self.TimesEvaluated,self.Populations[:,ii*Params.vib_mult+jj],c=colors[ii],ls=styles[jj],linewidth=2)
		#plt.ylim(0,1)
		myleg = []
		for ii in range(Params.dim-1):
			myleg = np.concatenate((myleg,[str(ii+1)]))
		plt.legend(myleg)
		# plt.legend(['DBVc','DBVd','MBVa','MBVb','PCBc158','PCBd158','PCBc82','PCBd82']) # MAKE PRETTY PLOTS HERE

		plt.xlabel('Time(ps)',fontsize = Params.LabelFontSize)
		plt.ylabel('Site Populations',fontsize = Params.LabelFontSize)
		plt.xlim(0,(Params.TMax+1)*Params.time_constant)
		plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+'Populations')
		plt.clf()

		if Params.vib_mult != 1:
			orig_pops = np.zeros(shape=(len(self.TimesEvaluated),Params.orig_dim-1))
			for ii in range(len(self.TimesEvaluated)):
				for jj in range(Params.orig_dim-1):
					for kk in range(Params.vib_mult):
						orig_pops[ii][jj] += self.Populations[ii][jj*Params.vib_mult+kk]

			fig = plt.figure()
			#plt.plot(self.TimesEvaluated,self.Populations[:,3])
			fig.hold()
			for ii in range(0,Params.orig_dim-1):
				plt.plot(self.TimesEvaluated,orig_pops[:,ii],c=colors[ii],linewidth=2)
			#plt.ylim(0,1)
			myleg = []
			for ii in range(Params.orig_dim-1):
				myleg = np.concatenate((myleg,[str(ii+1)]))
			plt.legend(myleg)
			# plt.legend(['DBVc','DBVd','MBVa','MBVb','PCBc158','PCBd158','PCBc82','PCBd82']) # MAKE PRETTY PLOTS HERE

			plt.xlabel('Time(ps)',fontsize = Params.LabelFontSize)
			plt.ylabel('Site Populations',fontsize = Params.LabelFontSize)
			plt.xlim(0,(Params.TMax+1)*Params.time_constant)
			plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+'Populations_orig')
			plt.clf()

		self.MarkovianRec.resize(len(self.TimesEvaluated),Params.dim-1)
		fig = plt.figure()
		#plt.plot(self.TimesEvaluated,self.Populations[:,3])
		fig.hold()
		for ii in range(0,Params.dim-1):
			plt.plot(self.TimesEvaluated,self.MarkovianRec[:,ii],linewidth=2)
		#plt.ylim(0,1)
		# plt.legend(['1','2','3','4','5','6','7'])
		plt.legend(myleg)
		# plt.legend(['DBVc','DBVd','MBVa','MBVb','PCBc158','PCBd158','PCBc82','PCBd82'])

		plt.xlabel('Time(ps)',fontsize = Params.LabelFontSize)
		plt.ylabel('Markovianity',fontsize = Params.LabelFontSize)
		plt.xlim(0,(Params.TMax+1)*Params.time_constant)
		plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+'Markovian')
		plt.clf()


			#plt.show()

		# Return dipole to generate isotropic dipole spectrum... 
		if Params.AllDirections: 
			ToInterpolate = None
			if (self.ShortTimePropagator.Polarization == "x"):
				ToInterpolate = self.Dipoles[:len(self.TimesEvaluated),0]
			elif (self.ShortTimePropagator.Polarization == "y"):
				ToInterpolate = self.Dipoles[:len(self.TimesEvaluated),1]
			elif (self.ShortTimePropagator.Polarization == "z"):
				ToInterpolate = self.Dipoles[:len(self.TimesEvaluated),2]						
			print "Interpolating... "
			from scipy import interpolate
			from scipy.interpolate import InterpolatedUnivariateSpline 
			interpf = InterpolatedUnivariateSpline(self.TimesEvaluated, ToInterpolate, k=1)
			print "Resampling at intervals of: ",Params.TMax/3000.," Atomic units"			
			Tstep = Params.TMax/3000.
			print "Resolution up to: ", (1/Tstep)*EvPerAu
			Times = numpy.arange(0.0,Params.TMax,Tstep)
			Mu = interpf(Times)
			print "Saving Resampled Dipoles... "
			numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/Dipole'+self.ShortTimePropagator.Polarization,Mu,fmt='%.18e')	
			print "Saving Last State Vector"
			self.VNow.Save('./Output'+Params.SystemName+ Params.start_time+'/LastState'+self.ShortTimePropagator.Polarization)
			print "Sending Array of shape: ", Mu.shape
			aPipe.send((Tstep,Mu))
			print "sent... (Killing This Child Process)"
			return 

		self.Norms[-1] = self.Norms[-2]
		if (Params.DoEntropies): 
			self.TimeDependentEntropy[-1] = self.TimeDependentEntropy[-2]		
		
		numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/TimesEvaluated',self.TimesEvaluated,fmt='%.18e')	
		numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/Dipoles',self.Dipoles,fmt='%.18e')	
		numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/Norms',self.Norms,fmt='%.18e')	
		numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/Coherences',self.Coherences,fmt='%.18e')	
		numpy.savetxt('./Output'+Params.SystemName+ Params.start_time+'/Populations',self.Populations,fmt='%.18e')	

	
		if (Params.Plotting): 

			# HACK
			self.Norms.resize(len(self.TimesEvaluated))
			self.Coherences.resize(len(self.TimesEvaluated))
			if Params.DirectionSpecific:
				self.Dipoles.resize(len(self.TimesEvaluated),3)
			else:
				self.Dipoles.resize(len(self.TimesEvaluated))

			# END HACK

			#import matplotlib.pyplot as plt
			import matplotlib.font_manager as fnt
			# Make plot styles visible. 
			PlotFont = {'fontname':'Helvetica','fontsize':18,'weight':'bold'}
			LegendFont = fnt.FontProperties(family='Helvetica',size='17',weight='bold')	
			l1 = plt.plot(self.TimesEvaluated,self.Norms,'k--')
			plt.setp(l1,linewidth=2, color='r')
			plt.xlabel('Time(ps)',fontsize = Params.LabelFontSize)
			plt.ylabel('|State|',fontsize = Params.LabelFontSize)
			plt.xlim(0,(Params.TMax+1)*Params.time_constant)
			plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+'NormOfState')
			plt.clf()

			l1 = plt.plot(self.TimesEvaluated,self.Coherences,'k--')
			plt.setp(l1,linewidth=2, color='r')
			plt.xlabel('Time(ps)',fontsize = Params.LabelFontSize)
			plt.ylabel('Coherence',fontsize = Params.LabelFontSize)
			plt.xlim(0,(Params.TMax+1)*Params.time_constant)
			plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+'Coherences')
			plt.clf()

			if (Params.DirectionSpecific):
				lx,ly,lz = plt.plot(self.TimesEvaluated,self.Dipoles[:,0],'k',self.TimesEvaluated,self.Dipoles[:,1],'k--',self.TimesEvaluated,self.Dipoles[:,2],'k.')
				plt.setp(lx,linewidth=2, color='r')
				plt.setp(ly,linewidth=2, color='g')
				plt.setp(lz,linewidth=2, color='b')
				plt.legend(['x','y','z'],loc=2)
				plt.xlabel('Time (ps)',fontsize = Params.LabelFontSize)
				plt.ylabel('Mu (au)',fontsize = Params.LabelFontSize)
				plt.xlim(0,(Params.TMax+1)*Params.time_constant)
				plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+'Dipole')
				plt.clf()
				Nrm = lambda X: (X[0]*X[0]+X[1]*X[1]+X[2]*X[2])
				DipoleStrength = map(Nrm,self.Dipoles)				
				SpectralAnalysis(numpy.array(DipoleStrength), Params.TStep, DesiredMaximum = 26.0/EvPerAu,Smoothing = True)
			else : 
				l1 = plt.plot(self.TimesEvaluated,self.Dipoles,'k--')
				plt.setp(l1,linewidth=2, color='r')
				plt.xlabel('Time(ps)',fontsize = Params.LabelFontSize)
				plt.ylabel('|Mu|',fontsize = Params.LabelFontSize)
				plt.xlim(0,(Params.TMax+1)*Params.time_constant)
				plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/"+'Dipole')
				plt.clf()
				print "Proc " + str(self.proc_index) + " building spectra"
				SpectralAnalysis(self.Dipoles, Params.TStep, DesiredMaximum = 4.0/EvPerAu,Smoothing = True)

		if Params.Parallel == True:
			aPipe.send(None)

		return 

	def AdiabaticRamp(self,Time): 
		if Time < 25: 
			return 0.0
		elif Time >= 25 and Time < 125: 
			return 1.0*((Time - 25.)/100.0)
		elif Time >=125 and Time < 225: 
			return 1.0*(100.0-(Time - 125.))/100.0
		else: 
			return 0.0

	def ExponentialStepDebug(self,VNow,TNow): 	
		self.TNow += Params.TStep	
		Tmp = self.ShortTimePropagator.NonPertMtrx + self.ShortTimePropagator.MarkovMatrix
		self.ToExp = Tmp.reshape((Params.nocc*Params.nvirt,Params.nocc*Params.nvirt))
	#	print "ToExp:",self.ToExp
		import scipy.linalg.matfuncs
		self.Propm = scipy.linalg.matfuncs.expm(self.ToExp*self.TNow)
		Tmp2 = self.Propm.reshape(Params.nocc,Params.nvirt,Params.nocc,Params.nvirt)
		self.VNow["r1_ph"] = numpy.tensordot(Tmp2,self.ShortTimePropagator.V0["r1_ph"],axes=([2,3],[0,1]))
		#tmp = self.Propm.reshape((Params.nocc,Params.nvirt,Params.nocc,Params.nvirt))
		#self.VNow["r1_ph"] = numpy.tensordot(tmp,self.VNow["r1_ph"],axes=([2,3],[0,1]))
		#self.TNow += Params.TStep	
		return 

	def ExponentialStep(self,VNow,TNow): 	
	#	import scipy.linalg.matfuncs
	#	self.Propm = scipy.linalg.matfuncs.expm(self.ToExp*self.TNow)
		V1 = numpy.tensordot(self.Propm,self.VNow["r1_ph"],axes=([2,3],[0,1]))
		V2 = numpy.tensordot(self.Propm,self.VNow["r1_ph"]+0.5*Params.TStep*V1,axes=([2,3],[0,1]))		
		V3 = numpy.tensordot(self.Propm,self.VNow["r1_ph"]+0.5*Params.TStep*V2,axes=([2,3],[0,1]))		
		V4 = numpy.tensordot(self.Propm,self.VNow["r1_ph"]+Params.TStep*V3,axes=([2,3],[0,1]))
		VNow["r1_ph"] += (1.0/6.0)*Params.TStep*V1
		VNow["r1_ph"] += (2.0/6.0)*Params.TStep*V2
		VNow["r1_ph"] += (2.0/6.0)*Params.TStep*V3
		VNow["r1_ph"] += (1.0/6.0)*Params.TStep*V4
		self.TNow += Params.TStep	
		return 

	def checkherm(self, somearray): # Used to check if a square array is Hermitian
		realdiff = 0.0
		imagdiff = 0.0
		for ii in range(len(somearray)):
			for jj in range(len(somearray)):
				realdiff += abs(np.real(somearray[ii,jj]) - np.real(somearray[jj,ii]))
				imagdiff += abs(np.imag(somearray[ii,jj]) + np.imag(somearray[jj,ii]))
		return [realdiff, imagdiff]

	def RungeKuttaStep(self,VNow,TNow): 
		if (self.Field != None) : 
			V1 = self.ShortTimePropagator.Step(VNow,TNow,Field=self.Field(self.TNow))
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V1),TNow+0.5*Params.TStep,Field=self.Field(self.TNow+0.5*Params.TStep))
			V3 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V2),TNow+0.5*Params.TStep,Field=self.Field(self.TNow+0.5*Params.TStep))
			V4 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,Params.TStep,V3),TNow+Params.TStep,Field=self.Field(self.TNow+Params.TStep))
		else: 
			V1 = self.ShortTimePropagator.Step(VNow,TNow)
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V1),TNow+0.5*Params.TStep)
			V3 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V2),TNow+0.5*Params.TStep)
			V4 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,Params.TStep,V3),TNow+Params.TStep)
		VNow.Add(V1,(1.0/6.0)*Params.TStep)
		VNow.Add(V2,(2.0/6.0)*Params.TStep)
		VNow.Add(V3,(2.0/6.0)*Params.TStep)
		VNow.Add(V4,(1.0/6.0)*Params.TStep)
		self.TNow += Params.TStep	
		return 

	def RK45Step(self,VNow,TNow):
		print "TimeStep", Params.TStep
		a = [ 0.0, 0.2, 0.3, 0.6, 1.0, 0.875 ]
 		b = [[],
      		  [0.2],
      		  [3.0/40.0, 9.0/40.0],
       		  [0.3, -0.9, 1.2],
      		  [-11.0/54.0, 2.5, -70.0/27.0, 35.0/27.0],
       		  [1631.0/55296.0, 175.0/512.0, 575.0/13824.0, 44275.0/110592.0, 253.0/4096.0]]
 		c  = [37.0/378.0, 0.0, 250.0/621.0, 125.0/594.0, 0.0, 512.0/1771.0]
  		dc = [c[0]-2825.0/27648.0, c[1]-0.0, c[2]-18575.0/48384.0, c[3]-13525.0/55296.0, c[4]-277.00/14336.0, c[5]-0.25]

  		if(self.TNow + Params.TStep > Params.TMax):
  			Params.TStep = Params.TMax - self.TNow 
		if (self.Field != None) : 
			print "To Be Done"
		else: 
			V1 = self.ShortTimePropagator.Step(VNow,TNow) #This is fine
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,b[1][0]*Params.TStep,V1),TNow+a[1]*Params.TStep) #This is fine
			V3 = self.ShortTimePropagator.Step(VNow.TwoCombo(1.0, V1, Params.TStep*b[2][0], V2, Params.TStep*b[2][1]),TNow+a[2]*Params.TStep)
			V4 = self.ShortTimePropagator.Step(VNow.ThreeCombo(1.0, V1, Params.TStep*b[3][0], V2, Params.TStep*b[3][1], V3, Params.TStep*b[3][2]),TNow+a[3]*Params.TStep)
			V5 = self.ShortTimePropagator.Step(VNow.FourCombo(1.0, V1, Params.TStep*b[4][0], V2, Params.TStep*b[4][1], V3, Params.TStep*b[4][2], V4, Params.TStep*b[4][3]),TNow+a[4]*Params.TStep)
			V6 = self.ShortTimePropagator.Step(VNow.FiveCombo(1.0, V1, Params.TStep*b[5][0], V2, Params.TStep*b[5][1], V3, Params.TStep*b[5][2], V4, Params.TStep*b[5][3], V5, Params.TStep*b[5][4]),TNow+a[4]*Params.TStep)

		E = V1.FiveCombo(dc[0], V2, dc[1], V3, dc[2], V4, dc[3], V5, dc[4], V6, dc[5])
		tore = complex(0.0,0.0)
		for T in E.iterkeys(): 
			tore += numpy.sum(E[T].conj()*E[T])
		Error = numpy.sqrt(tore.real)
		#print "Error = " + str(Error)
		#Emax = Params.tol*VNow.InnerProduct(VNow).real
		tore = complex(0.0,0.0)
		for T in E.iterkeys(): 
			tore += numpy.sum(VNow["r"].conj()*VNow["r"])
		Emax = Params.tol*numpy.sqrt(tore.real)
		#print V1.InnerProduct(V1), V2.InnerProduct(V2), V3.InnerProduct(V3), V4.InnerProduct(V4), V5.InnerProduct(V5), V6.InnerProduct(V6) 
		#print VNow.InnerProduct(VNow)

		#print "E ", Error, "   Inner Product", VNow.InnerProduct(VNow).real

		#VNow.Add(V1,c[0]*Params.TStep)
		#VNow.Add(V2,c[1]*Params.TStep)
		#VNow.Add(V3,c[2]*Params.TStep)
		#VNow.Add(V4,c[3]*Params.TStep)
		#VNow.Add(V5,c[4]*Params.TStep)
		#VNow.Add(V6,c[5]*Params.TStep)
		
		if Error < Emax or Emax == 0.0:
			#print 'Taking route ONE'
			self.TNow += Params.TStep
			VNow.Add(V1,c[0]*Params.TStep)
			VNow.Add(V2,c[1]*Params.TStep)
			VNow.Add(V3,c[2]*Params.TStep)
			VNow.Add(V4,c[3]*Params.TStep)
			VNow.Add(V5,c[4]*Params.TStep)
			VNow.Add(V6,c[5]*Params.TStep)
		else:
			#print 'Taking route TWO'
			Params.TStep = Params.Safety*Params.TStep*(Emax/Error)**0.2
			V1 = self.ShortTimePropagator.Step(VNow,TNow) #This is fine
			V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,b[1][0]*Params.TStep,V1),TNow+a[1]*Params.TStep) #This is fine
			V3 = self.ShortTimePropagator.Step(VNow.TwoCombo(1.0, V1, Params.TStep*b[2][0], V2, Params.TStep*b[2][1]),TNow+a[2]*Params.TStep)
			V4 = self.ShortTimePropagator.Step(VNow.ThreeCombo(1.0, V1, Params.TStep*b[3][0], V2, Params.TStep*b[3][1], V3, Params.TStep*b[3][2]),TNow+a[3]*Params.TStep)
			V5 = self.ShortTimePropagator.Step(VNow.FourCombo(1.0, V1, Params.TStep*b[4][0], V2, Params.TStep*b[4][1], V3, Params.TStep*b[4][2], V4, Params.TStep*b[4][3]),TNow+a[4]*Params.TStep)
			V6 = self.ShortTimePropagator.Step(VNow.FiveCombo(1.0, V1, Params.TStep*b[5][0], V2, Params.TStep*b[5][1], V3, Params.TStep*b[5][2], V4, Params.TStep*b[5][3], V5, Params.TStep*b[5][4]),TNow+a[4]*Params.TStep)
			self.TNow += Params.TStep
			VNow.Add(V1,c[0]*Params.TStep)
			VNow.Add(V2,c[1]*Params.TStep)
			VNow.Add(V3,c[2]*Params.TStep)
			VNow.Add(V4,c[3]*Params.TStep)
			VNow.Add(V5,c[4]*Params.TStep)
			VNow.Add(V6,c[5]*Params.TStep)
			return

		if (Emax == 0):
			Params.TStep = Params.TStep*2.0
			return
		Params.TStep = Params.Safety*Params.TStep*(Emax/Error)**0.2
		return 
		
	def ImRungeKuttaStep(self,VNow,TNow): 	
		V1 = self.ShortTimePropagator.Step(VNow,TNow)
		V2 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V1),TNow+0.5*Params.TStep)
		V3 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,0.5*Params.TStep,V2),TNow+0.5*Params.TStep)
		V4 = self.ShortTimePropagator.Step(VNow.LinearCombination(1.0,Params.TStep,V3),TNow+Params.TStep)
		VNow.Add(V1,(1.0/6.0)*(1.0j)*Params.TStep)
		VNow.Add(V2,(2.0/6.0)*(1.0j)*Params.TStep)
		VNow.Add(V3,(2.0/6.0)*(1.0j)*Params.TStep)
		VNow.Add(V4,(1.0/6.0)*(1.0j)*Params.TStep)	
		self.TNow += Params.TStep		
		return 
				