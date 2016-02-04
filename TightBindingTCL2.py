import numpy as np
import scipy
from numpy import array
from numpy import linalg
from scipy import interpolate
from scipy import linalg
from scipy import integrate
from scipy import special 
from scipy.optimize import curve_fit
import scipy.weave as weave
import __builtin__ 
import os, sys, time, math, re, random, cmath
import pickle
from time import gmtime, strftime
from types import * 
from itertools import izip
from heapq import nlargest
from multiprocessing import Process, Queue, Pipe
from math import pow, exp, cos, sin, log, pi, sqrt, isnan
from scipy.fftpack import ifft
from TensorNumerics import * 
from LooseAdditions import * 
from NumGrad import * 
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.pylab as lab
import hyp_2F1

def DL(w, g, l, O):
	prefact = 2*l*g*w
	rest = (1/(g**2+(w-O)**2) + 1/(g**2+(w+O)**2))
	return prefact*rest

def thermal_reorg(g, l, O, beta):
	return 2*l*(np.arctan((1-beta*O)/(beta*g)) + np.arctan((1+beta*O)/(beta*g)))/np.pi

class TightBindingTCL: 
	def __init__(self):


		tstep = Params.TStep
		Temp = Params.Temperature

		self.silence = True # If true, don't print a whole bunch of stuff. 
		# If false, print a whole bunch of stuff. 
		if Params.Parallel == True:
			self.silence = True

		# these things must be defined by the algebra. 
		self.VectorShape = None 
		self.ResidualShape = None

		mustr = []
		SD_dir = ''
		AuPerWavenumber = 4.5563e-6
		Hsys = np.zeros(shape=(Params.dim,Params.dim)) 
		H_elements_set = 0
		myinput = open('TCL.input')
		for line in myinput:
			split_line = line.split()
			if split_line[0] == 'dipole':
				for ii in range(1,len(split_line)):
					mustr += [float(split_line[ii])]
			if split_line[0] == 'SD_dir':
				SD_dir = split_line[1]
				if SD_dir[-1:] != '/':
					SD_dir = SD_dir + '/'
			if split_line[0] == 'ham':
				if Hsys[int(split_line[1]),int(split_line[2])] != 0:
					print 'ERROR: Element ' + split_line[1] + ',' + split_line[2] +' already set!' 
					print huh
				if Hsys[int(split_line[2]),int(split_line[1])] != 0:
					print 'ERROR: Element ' + split_line[2] + ',' + split_line[1] +' already set!' 
					print huh
				if int(split_line[1]) == int(split_line[2]):
					Hsys[int(split_line[1]),int(split_line[1])] = (float(split_line[3])-Params.globalscale)*AuPerWavenumber
				else:
					Hsys[int(split_line[1]),int(split_line[2])] = float(split_line[3])*AuPerWavenumber
					Hsys[int(split_line[2]),int(split_line[1])] = float(split_line[3])*AuPerWavenumber
				H_elements_set += 1
		if H_elements_set != ((Params.dim-1)*(Params.dim-1)-(Params.dim-1))/2 + (Params.dim-1):
			print 'ERROR: Number of Hsys elements set not correct!'
			print huh
	
		if len(mustr) != Params.dim:
			print 'ERROR: Dipole not set correctly! Check input file.'
			print huh

		self.field = 0.01

		EvPerAu = 27.2113
		Kb = 8.61734315e-5/EvPerAu
		self.beta = (1.0/(Kb*Temp)) # Define beta, units of 1/Hartree
	
		self.VNow = StateVector()
		self.VNow["abs"] = np.zeros(shape=(Params.dim,Params.dim),dtype = complex)
		self.VNow["coh"] = np.zeros(shape=(Params.dim,Params.dim),dtype = complex)
		self.VNow["pop"] = np.zeros(shape=(Params.dim,Params.dim),dtype = complex)
		self.V0 = self.VNow.clone()

		self.Mu = np.zeros(shape=(Params.dim,Params.dim))
		for ii in range(1,len(mustr)):
			self.V0["abs"][ii,0] = mustr[ii]*self.field
			self.Mu[0,ii] = mustr[ii]*self.field

		import csv
		self.peak_params = []
		self.prev_first_terms = []
		self.PV_rec = []

		for ii in range(Params.dim-1):
			reorg_so_far = 0.0
			self.peak_params.append([])
			self.prev_first_terms.append([])
			self.PV_rec.append([])
			reader = csv.reader(open(SD_dir + str(ii+1) + '.csv','rb'))
			for row in reader:
				g = float(row[0])*AuPerWavenumber
				l = float(row[1])*AuPerWavenumber
				O = float(row[2])*AuPerWavenumber
				self.peak_params[ii].append(g)
				self.peak_params[ii].append(l)
				self.peak_params[ii].append(O)
				reorg_so_far += 2*l
			if Params.const_reorg and Params.const_reorg_type == 1:
				self.peak_params[ii].append(60*AuPerWavenumber)
				self.peak_params[ii].append((AuPerWavenumber*Params.const_reorg_vals[ii]-reorg_so_far)/2)
				self.peak_params[ii].append(1600*AuPerWavenumber)
			elif Params.const_reorg and Params.const_reorg_type == 2:
				Hsys[ii+1,ii+1] += AuPerWavenumber*Params.const_reorg_vals[ii]-reorg_so_far
		print 'Number of peaks:'
		for ii in range(Params.dim-1):
			print len(self.peak_params[ii])/3


		self.evecs = np.linalg.eigh(Hsys)[1]
		self.invevecs = np.linalg.inv(self.evecs)

		Params.evecs = self.evecs
		Params.invevecs = self.invevecs

		self.Hsys = np.dot(np.dot(self.invevecs,Hsys),self.evecs)
		Hsysdiag = np.diag(self.Hsys)

		self.Mu = np.dot(np.dot(self.invevecs,self.Mu),self.evecs)

		self.V0["abs"] = np.dot(np.dot(self.invevecs,self.V0["abs"]),self.evecs)
		self.V0["pop"][Params.populations[0],Params.populations[1]] = 1.0
		self.V0["pop"] = np.dot(np.dot(self.invevecs,self.V0["pop"]),self.evecs)
		self.V0["coh"][Params.coherences[0],Params.coherences[1]] = 1.0
		if Params.coh_type == 0:
			self.V0["coh"] = np.dot(np.dot(self.invevecs,self.V0["coh"]),self.evecs)
		
		self.eps = np.zeros(shape=(Params.dim-1,Params.dim-1))
		for ii in range(1,Params.dim):
			for jj in range(1,Params.dim):
				self.eps[ii-1,jj-1] = Hsysdiag[ii] - Hsysdiag[jj]
		self.eps_squared = np.power(self.eps,2)
		self.exp_itD = np.exp(1j*0.0*self.eps)

		if self.silence == False:
			print "Hsys: "
			print self.Hsys

		# # # # # # # # # # # # # # # # SD construction workspace # # # # # # # # # # # # # #
		
		# wgrid = np.linspace(0,4000,4000)
		# # GoodMkdir("./Figures"+Params.SystemName+ Params.start_time)
		# # GoodMkdir("./Output"+Params.SystemName+ Params.start_time)	
		# bilins = ['DBT','DBO','MBT','MBO','PCH','PCO','PCF','PCT']
		# thermal_reorgs = [199.9,209.8,421.2,419.5,240.1,234.4,157.8,143.6]
		# reorgs = [748.15,672.34,1443.58,1103.51,974.00,1026.11,675.69,626.17]
		# # jgrid_total = np.zeros(1768)
		# jgrid_total = []
		# GoodMkdir("./Figures_testing")
		# for ii in range(Params.dim-1):
		# 	myreorg = 0.0
		# 	mythermalreorg = 0.0

		# 	wgrid1 = []
		# 	jgrid1 = []
		# 	myreal = open('SDs_QMMM_final/final_' + bilins[ii] + '_diab_0.05.csv')
		# 	for line in myreal:
		# 		split_line = line.split(',')
		# 		if float(split_line[0]) > Params.SD_bot and float(split_line[0]) < Params.SD_range:
		# 			wgrid1 = np.concatenate((wgrid1,[float(split_line[0])]))
		# 			jgrid1 = np.concatenate((jgrid1,[float(split_line[1])]))
		# 	if jgrid_total == []:
		# 		jgrid_total = np.zeros(len(jgrid1))
		# 	jgrid_total += jgrid1/8

		# 	jgrid = DL(wgrid1,1.0,0.0,0.0)
		# 	for jj in range(len(self.peak_params[ii])/3):
		# 		g = self.peak_params[ii][jj*3]/AuPerWavenumber
		# 		l = self.peak_params[ii][jj*3+1]/AuPerWavenumber
		# 		O = self.peak_params[ii][jj*3+2]/AuPerWavenumber

		# 		# plt.plot(wgrid1,DL(wgrid1,g,l,O))				

		# 		jgrid += DL(wgrid1,g,l,O)
		# 		myreorg += 2*l
		# 		mythermalreorg += thermal_reorg(g,l,O,self.beta*AuPerWavenumber)
		# 	# temp=plt.plot(wgrid[0:Params.SD_range],jgrid[0:Params.SD_range])
		# 	temp=plt.plot(wgrid1,jgrid)
		# 	plt.setp(temp,linewidth=2)


		# 	plt.plot(wgrid1,jgrid1)
		# 	mydiff = 0.0
		# 	for jj in range(len(jgrid)):
		# 		mydiff += abs(jgrid[jj] - jgrid1[jj])
		# 	print mydiff
		# 	plt.title('RE = ' + str(myreorg) + ', TRE = ' + str(mythermalreorg) + ' - both (cm^-1)')
		# 	# plt.title('mydiff = ' + str(mydiff))
		# 	plt.xlabel('w (cm^-1)',fontsize = Params.LabelFontSize)
		# 	plt.ylabel('J[w]/hbar (cm^-1)',fontsize = Params.LabelFontSize)
		# 	# if ii == 5:
		# 	# 	print bilins[ii]
		# 	# 	plt.show()
		# 	# 	print huh
		# 	plt.savefig("./Figures_testing/SD"+ str(ii+1))
		# 	plt.clf()
		# print huh
		# # jgrid = DL(wgrid1,1.0,0.0,0.0)
		# # myreorg = 0.0
		# # mythermalreorg = 0.0
		# # reader = csv.reader(open(SD_dir + 'avg.csv','rb'))
		# # for row in reader:
		# # 	g = float(row[0])
		# # 	l = float(row[1])
		# # 	O = float(row[2])
		# # 	jgrid += DL(wgrid1,g,l,O)
		# # 	myreorg += 2*l
		# # 	mythermalreorg += thermal_reorg(g,l,O,self.beta*AuPerWavenumber)
		# # temp=plt.plot(wgrid1,jgrid)
		# # plt.plot(wgrid1,jgrid_total)
		# # mydiff = 0.0
		# # for jj in range(len(jgrid)):
		# # 	mydiff += abs(jgrid[jj] - jgrid_total[jj])
		# # print mydiff
		# # plt.setp(temp,linewidth=2)
		# # plt.title('Average PC645 Spectral Density')
		# # # plt.title('RE = ' + str(np.average(reorgs)) + ', TRE = ' + str(np.average(thermal_reorgs)) + ' - both (cm^-1)')
		# # plt.title('RE = ' + str(myreorg) + ', TRE = ' + str(mythermalreorg) + ' - both (cm^-1)')
		# # plt.xlabel('w (cm^-1)',fontsize = Params.LabelFontSize)
		# # plt.ylabel('J[w] (cm^-1)',fontsize = Params.LabelFontSize)
		# # # plt.savefig("./Figures_testing/SDavg")	
		# # plt.show()
		# # print huh
		# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 


		wgrid = np.linspace(0,4000,4000)
		GoodMkdir("./Figures"+Params.SystemName+ Params.start_time)
		GoodMkdir("./Output"+Params.SystemName+ Params.start_time)	
		for ii in range(Params.dim-1):
			myreorg = 0.0
			mythermalreorg = 0.0
			jgrid = DL(wgrid,1.0,0.0,0.0)
			for jj in range(len(self.peak_params[ii])/3):
				g = self.peak_params[ii][jj*3]/AuPerWavenumber
				l = self.peak_params[ii][jj*3+1]/AuPerWavenumber
				O = self.peak_params[ii][jj*3+2]/AuPerWavenumber
				jgrid += DL(wgrid,g,l,O)
				myreorg += 2*l
				mythermalreorg += thermal_reorg(g,l,O,self.beta*AuPerWavenumber)
				# print 'sd_peak:' + str(ii)+ ':' + str(jj) + '=' + str(l) + ':' + str(5308.83662006/g) + ':' + str(O)
			temp=plt.plot(wgrid[0:Params.SD_range],jgrid[0:Params.SD_range])
			plt.setp(temp,linewidth=2)
			plt.title('RE = ' + str(myreorg) + ', TRE = ' + str(mythermalreorg) + ' - both (cm^-1)')
			plt.xlabel('w (cm^-1)',fontsize = Params.LabelFontSize)
			plt.ylabel('J[w]/hbar (cm^-1)',fontsize = Params.LabelFontSize)
			plt.savefig("./Figures"+Params.SystemName+ Params.start_time+"/SD" + str(ii+1))
			plt.clf()


		if self.silence == False:
			print "Building U..."

		U = np.zeros(shape=(Params.dim,Params.dim,Params.dim))
		evecs = self.evecs
		dim = Params.dim
		code =	"""
				#line 406 "TightBindingTCL2.py"
				using namespace std;
				complex<double> imi(0.0,1.0);
				for (int m = 1 ; m<dim; ++m)
				{
					for (int a = 1; a<dim; ++a)
					{
						for (int b = 1; b<dim; ++b)
						{
							U(m,a,b) = evecs(m,a) * evecs(m,b);
						}
					}
				}
				"""
		weave.inline(code,
					['U','evecs','dim'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler, extra_compile_args = Params.Flags,
					verbose = 1)

		self.U = U

		# Keep track of how much time has passed in the ds integration
		self.timesofar = 0.0

		# These will be our four gamma tensors.
		self.G1 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		self.G2 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		self.G3 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		self.G4 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)

		for ii in range(len(self.prev_first_terms)):
			for kk in range(len(self.peak_params[ii])/3):
				self.prev_first_terms[ii] = np.concatenate((self.prev_first_terms[ii],[1]))
				self.PV_rec[ii] = np.concatenate((self.PV_rec[ii],[0]))
			self.prev_first_terms[ii] = np.ones(shape=(len(self.prev_first_terms[ii]),Params.dim-1,Params.dim-1),dtype=complex)

		self.prev_second_terms = copy.deepcopy(self.prev_first_terms)

		# Precompute a whole bunch of stuff:

		self.g_2 = []
		self.O_2 = []
		self.O_4 = []
		self.complex_1 = []
		self.first_exp = []
		self.third_exp = []
		self.sixth_exp = []
		self.complex_2 = copy.deepcopy(self.prev_first_terms)
		self.DO_2 = copy.deepcopy(self.prev_first_terms)
		self.g2D2 = copy.deepcopy(self.prev_first_terms)
		self.first_chain = copy.deepcopy(self.prev_first_terms)
		self.second_chain = copy.deepcopy(self.prev_first_terms)
		self.third_chain = copy.deepcopy(self.prev_first_terms)
		self.fourth_chain = copy.deepcopy(self.prev_first_terms)
		self.fifth_chain = copy.deepcopy(self.prev_first_terms)
		self.sixth_chain = copy.deepcopy(self.prev_first_terms)
		self.i4gd = copy.deepcopy(self.prev_first_terms)
		self.full_long_first = copy.deepcopy(self.prev_first_terms)
		self.first_chunk = copy.deepcopy(self.prev_first_terms)
		self.second_chunk = copy.deepcopy(self.prev_first_terms)

		for ii in range(Params.dim-1):
			self.g_2.append([])
			self.O_2.append([])
			self.O_4.append([])
			self.complex_1.append([])
			self.first_exp.append([])
			self.third_exp.append([])
			self.sixth_exp.append([])
			for jj in range(len(self.peak_params[ii])/3):
				g = self.peak_params[ii][jj*3]
				l = self.peak_params[ii][jj*3+1]
				O = self.peak_params[ii][jj*3+2]
				self.g_2[ii] = np.concatenate((self.g_2[ii],[np.power(g,2)]))
				self.O_2[ii] = np.concatenate((self.O_2[ii],[np.power(O,2)]))
				self.O_4[ii] = np.concatenate((self.O_4[ii],[np.power(O,4)]))
				self.complex_1[ii] = np.concatenate((self.complex_1[ii],[np.power(g + 1j*O,2)]))
				self.first_exp[ii] = np.concatenate((self.first_exp[ii],[np.exp(self.beta*O)]))
				self.third_exp[ii] = np.concatenate((self.third_exp[ii],[np.exp(-(1j*self.beta*g))]))
				self.sixth_exp[ii] = np.concatenate((self.sixth_exp[ii],[np.exp(self.beta*(-1j*g + O))]))
				self.complex_2[ii][jj] = np.power(g - 1j*self.eps,2)
				self.DO_2[ii][jj] = np.power(self.eps + O,2)
				self.g2D2[ii][jj] = np.power(np.power(g,2) + self.eps_squared,2)
				self.first_chain[ii][jj] = (g - 1j*(self.eps - O))*(g + 1j*(self.eps - O))*(g - 1j*O)*(g + 1j*(self.eps + O))
				self.second_chain[ii][jj] = (g - 1j*(self.eps - O))*(g - 1j*O)*(g - 1j*(self.eps + O))*(g + 1j*(self.eps + O))
				self.third_chain[ii][jj] = (g + 1j*(self.eps - O))*(g + 1j*O)*(g - 1j*(self.eps + O))*(g + 1j*(self.eps + O))
				self.fourth_chain[ii][jj] = (g - 1j*(self.eps - O))*(g + 1j*(self.eps - O))*(g + 1j*O)*(g - 1j*(self.eps + O))
				self.fifth_chain[ii][jj] = np.pi*(self.g2D2[ii][jj] + 2*(g - self.eps)*(g + self.eps)*self.O_2[ii][jj] + self.O_4[ii][jj])
				self.i4gd[ii][jj] = 4*1j*g*self.eps*(self.g_2[ii][jj] + self.eps_squared + self.O_2[ii][jj])
				self.sixth_chain[ii][jj] = (-self.eps_squared - self.complex_1[ii][jj])*(g - 1j*O)
				self.full_long_first[ii][jj] = 2*l*((self.first_exp[ii][jj]*(g - 1j*O))/((-self.third_exp[ii][jj] + self.first_exp[ii][jj])*(-1j*g + self.eps - O)) - (g + 1j*O)/((-1 + self.sixth_exp[ii][jj])*(-1j*g + self.eps + O)))
				self.first_chunk[ii][jj] = ((-self.third_exp[ii][jj] + self.first_exp[ii][jj])*(-1j*g + self.eps - O))
				self.second_chunk[ii][jj] = (-1 + self.sixth_exp[ii][jj])*(-1j*g + self.eps + O)

		self.useful_exp = np.exp(-2.0 * np.pi * 50000.0 / self.beta)
		self.other_exp = 1/self.useful_exp
		self.exp_itD = np.exp(1j*50000.0*self.eps)
		# These three have to be set so that the long time terms are computed correctly below

		self.long_time_first_term = copy.deepcopy(self.prev_first_terms)
		self.long_time_second_term = copy.deepcopy(self.prev_second_terms)
		for ii in range(Params.dim-1):
			for jj in range(len(self.peak_params[ii])/3):
				g = self.peak_params[ii][jj*3]
				l = self.peak_params[ii][jj*3+1]
				O = self.peak_params[ii][jj*3+2]
				self.long_time_first_term[ii][jj] = self.first_term(50000.0, g, l, O, self.eps,ii,jj)
				self.long_time_second_term[ii][jj] = self.second_term(50000.0, g, l, O, self.eps,ii,jj)

		# Then we set them back to the value for t = 0
		self.useful_exp = 1.0
		self.other_exp = 1.0
		self.exp_itD = np.exp(1j*0.0*self.eps)

		self.markovian = Params.MarkovApprox

		self.secular_mat = np.ones(shape=(Params.dim,Params.dim,Params.dim,Params.dim))
		if Params.SecularApproximation == 1:
			for a in range(Params.dim):
				for b in range(Params.dim):
					for c in range(Params.dim):
						for d in range(Params.dim):
							if (a == b and c != d) or (c == d and a != b):
								self.secular_mat[a,b,c,d] = 0

		if self.markovian == True:
			self.build_markovian_gammas()

		GoodMkdir("./Figures"+Params.SystemName+ Params.start_time)
		GoodMkdir("./Output"+Params.SystemName+ Params.start_time)	
		os.system('cp TCL.input Output' + Params.SystemName + Params.start_time+ '/')
		os.system('cp -r ' + SD_dir + '/* Output'+ Params.SystemName + Params.start_time+ '/')
		np.savetxt('./Output'+Params.SystemName+ Params.start_time+'/RunParams.txt',[0],fmt='%.18e')
		f = open('./Output'+Params.SystemName+ Params.start_time+'/RunParams.txt','w')
		mywrite = 'Num peaks = '
		for ii in range(Params.dim-1):
			mywrite += str(len(self.peak_params[ii])/3) + ' '
		mywrite += '\n'
		f.write(mywrite)
		f.write('Markov Apprximation = ' + str(self.markovian) + '\n')
		f.write('Tmax = ' + str(Params.TMax) + '\n')
		f.write('TStep = ' + str(Params.TStep) + '\n')
		f.write('Pop = ' + str(Params.populations) + '\n')
		f.write('Coh = ' + str(Params.coherences) + '\n')
		f.close()

		self.InitAlgebra()
		if self.silence == False:
			print "Algebra Initalization Complete... "
		self.Mu0 = None
		return 

	def HurwitzLerchPhi(self, z, s, a):
		return hyp_2F1.hyp_2f1(1.0, a, 1.0+a, z)/a

	def HarmonicNumber(self, argument):
		return special.psi(argument + 1.0) - special.psi(1.0)


	def Energy(self,AStateVector): 
		return AStateVector["r1_ph"][0][1]
	
	def InitAlgebra(self): 	
		self.VectorShape = StateVector()
		self.VectorShape["abs"] = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		self.VectorShape["coh"] = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		self.VectorShape["pop"] = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		return 

	def DipoleMoment(self,AVector,Time=0.0): 
		return np.sum(np.diag(np.dot(self.Mu,AVector["abs"])))

	def InitalizeNumerics(self): 
		self.V0.MultiplyScalar(1.0/np.sqrt(self.V0.InnerProduct(self.V0)))
		self.Mu0 = self.DipoleMoment(self.V0)
		return 
		
	def First(self): 
		return self.V0

	def first_term(self, t, g, l, O, D, ii, jj):
		second_exp = np.exp(t*(g + 1j*(D - O)))
		fourth_exp = np.exp(t*(g + 1j*(D + O)))
		fifth_exp = np.exp(2*1j*t*O)
		if np.average(np.real(fourth_exp)) > 100000000.0:
			temp = self.full_long_first[ii][jj]
		else:
			temp = (2*l*((self.first_exp[ii][jj]*(-1 + second_exp)*(g - 1j*O))/self.first_chunk[ii][jj] - ((-1 + fourth_exp)*(g + 1j*O))/(fifth_exp*self.second_chunk[ii][jj])))/second_exp
		return temp

	def second_term(self, t, g, l, O, D, ii, jj):
		first_HN = self.HarmonicNumber((-0.5*1j*self.beta*D)/np.pi)
		temp0 = np.zeros(4,dtype=complex)
		temp0[0] = (-(self.beta*(g - 1j*O))/(2.*np.pi))
		temp0[1] = ( (self.beta*(g - 1j*O))/(2.*np.pi))
		temp0[2] = (-(self.beta*(g + 1j*O))/(2.*np.pi))
		temp0[3] = ( (self.beta*(g + 1j*O))/(2.*np.pi))
		HNs = self.HarmonicNumber(temp0)
		second_HN = HNs[0]
		third_HN = HNs[1] 
		fourth_HN = HNs[2]
		fifth_HN = HNs[3] 
		temp1 = np.zeros(4,dtype=complex)
		temp1[0] = ((2*np.pi + self.beta*g - 1j*self.beta*O)/(2.*np.pi))
		temp1[1] = ((2*np.pi - self.beta*g + 1j*self.beta*O)/(2.*np.pi))
		temp1[2] = ((2*np.pi - self.beta*g - 1j*self.beta*O)/(2.*np.pi))
		temp1[3] = ((2*np.pi + self.beta*g + 1j*self.beta*O)/(2.*np.pi))
		HLPs = self.HurwitzLerchPhi(self.useful_exp,1,temp1)
		second_HLP = HLPs[0]
		third_HLP = HLPs[1]
		fourth_HLP = HLPs[2]
		fifth_HLP = HLPs[3]
		first_HLP =  self.HurwitzLerchPhi(self.useful_exp,1,1 - (0.5*1j*self.beta*D)/np.pi)
		return (l*(self.other_exp*self.i4gd[ii][jj]*first_HN - self.other_exp*self.first_chain[ii][jj]*second_HN + self.other_exp*self.second_chain[ii][jj]*third_HN - self.other_exp*self.third_chain[ii][jj]*fourth_HN + self.other_exp*self.fourth_chain[ii][jj]*fifth_HN + self.exp_itD*(self.i4gd[ii][jj]*first_HLP + self.second_chain[ii][jj]*second_HLP + (g + 1j*(D - O))*(self.sixth_chain[ii][jj]*third_HLP + (g + 1j*O)*((-self.g_2[ii][jj] - self.DO_2[ii][jj])*fourth_HLP + (self.complex_2[ii][jj] + self.O_2[ii][jj])*fifth_HLP)))))/(self.other_exp*self.fifth_chain[ii][jj])

	def markov_term(self, g, l, O, D):
		return (l*((4*1j*g*D*(np.power(g,2) + self.eps_squared + np.power(O,2))*self.HarmonicNumber((-0.5*1j*self.beta*D)/np.pi))/(np.power(np.power(g,2) + self.eps_squared,2) + 2*(g - D)*(g + D)*np.power(O,2) + np.power(O,4)) - ((g - 1j*O)*self.HarmonicNumber(-(self.beta*(g - 1j*O))/(2.*np.pi)))/(g - 1j*(D + O)) + ((g - 1j*O)*self.HarmonicNumber((self.beta*(g - 1j*O))/(2.*np.pi)))/(g + 1j*(D - O)) - ((g + 1j*O)*self.HarmonicNumber(-(self.beta*(g + 1j*O))/(2.*np.pi)))/(g - 1j*D + 1j*O) + ((g + 1j*O)*self.HarmonicNumber((self.beta*(g + 1j*O))/(2.*np.pi)))/(g + 1j*(D + O))))/np.pi - (4*1j*np.exp(self.beta*O)*l*(np.exp(1j*self.beta*g)*(np.power(g,2) + 1j*g*D + np.power(O,2)) - (np.power(g,2) + 1j*g*D + np.power(O,2))*np.cosh(self.beta*O) - D*O*np.sinh(self.beta*O)))/((-1 + np.exp(self.beta*(-1j*g + O)))*(-1 + np.exp(self.beta*(1j*g + O)))*(np.power(g + 1j*D,2) + np.power(O,2)))

	def build_markovian_gammas(self):
		mytotals = np.zeros(shape=(Params.dim-1, Params.dim-1, Params.dim-1),dtype=complex)
		for ii in range(Params.dim-1):
			this_total = np.zeros(shape=(Params.dim-1, Params.dim-1),dtype=complex)
			for kk in range(len(self.peak_params[ii])/3):
				g = self.peak_params[ii][kk*3]
				l = self.peak_params[ii][kk*3+1]
				O = self.peak_params[ii][kk*3+2]
				this_total += self.markov_term(g, l, O, self.eps)
			mytotals[ii] = this_total

		G1 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		G2 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		G3 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		G4 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		U = self.U
		secular_mat = self.secular_mat
		dim = Params.dim

		code =	"""
				#line 408 "TightBindingTCL2.py"
				using namespace std;
				complex<double> imi(0.0,1.0);
				for (int a = 1 ; a<dim; ++a)
				{
					for (int b = 1; b<dim; ++b)
					{
						for (int c = 1; c<dim; ++c)
						{
							for (int d = 1; d<dim; ++d)
							{
								for (int m = 1; m<dim; ++m)
								{
									if (b == c and secular_mat(a,c,c,d) == 1)
									{
										G1(a,c,c,d) += mytotals(m-1,d-1,c-1) * U(m,a,c) * U(m,c,d);
									}

									if (a == d and secular_mat(c,d,d,b) == 1)
									{
										G4(c,d,d,b) += mytotals(m-1,c-1,d-1) * U(m,c,d) * U(m,d,b);
									}
									if (secular_mat(a,c,d,b) == 1)
									{
										G2(a,c,d,b) += mytotals(m-1,c-1,a-1) * U(m,a,c) * U(m,d,b);
										G3(a,c,d,b) += mytotals(m-1,d-1,b-1) * U(m,a,c) * U(m,d,b);
									}
								}
							}
						}
					}
				}
				"""
		weave.inline(code,
					['G1','G2','G3','G4','mytotals','U','secular_mat','dim'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler, extra_compile_args = Params.Flags,
					verbose = 1)
		self.G1 = G1
		self.G2 = G2
		self.G3 = G3
		self.G4 = G4

	def build_gammas(self,t):
		mytotals = np.zeros(shape=(Params.dim-1,Params.dim-1,Params.dim-1),dtype=complex)
		for ii in range(Params.dim-1):
			this_total = np.zeros(shape=(Params.dim-1,Params.dim-1),dtype=complex)
			for jj in range(len(self.peak_params[ii])/3):
				if self.PV_rec[ii][jj] <= 2:
					g = self.peak_params[ii][jj*3]
					l = self.peak_params[ii][jj*3+1]
					O = self.peak_params[ii][jj*3+2]
					orig_PV_rec = copy.deepcopy(self.PV_rec[ii][jj])
					if orig_PV_rec != 1:
						this_first_term = self.first_term(t,g,l,O,self.eps,ii,jj)
						this_diff = abs(this_first_term - self.prev_first_terms[ii][jj])/abs(self.prev_first_terms[ii][jj])
						future_diff = abs(this_first_term - self.long_time_first_term[ii][jj])/abs(self.long_time_first_term[ii][jj])
						if np.max(this_diff) < 1.0*10**(-7) and np.max(future_diff) < 1.0*10**(-7):
							self.PV_rec[ii][jj] += 1
						else:
							self.prev_first_terms[ii][jj] = this_first_term
					else:
						this_first_term = self.prev_first_terms[ii][jj]
					if orig_PV_rec != 2:
						this_second_term = self.second_term(t, g, l, O, self.eps,ii,jj)
						this_diff = abs(this_second_term - self.prev_second_terms[ii][jj])/abs(self.prev_second_terms[ii][jj])
						future_diff = abs(this_second_term - self.long_time_second_term[ii][jj])/abs(self.long_time_second_term[ii][jj])
						if np.max(this_diff) < 1.0*10**(-7) and np.max(future_diff) < 1.0*10**(-7):
							self.PV_rec[ii][jj] += 2
						else:
							self.prev_second_terms[ii][jj] = this_second_term
					else:
						this_second_term = self.prev_second_terms[ii][jj]
					this_total += this_first_term + this_second_term
				else:
					this_total += self.prev_first_terms[ii][jj] + self.prev_second_terms[ii][jj]
			mytotals[ii] = this_total

		for ii in range(len(self.PV_rec)):
			Params.MarkovianRecord[ii] = np.sum(self.PV_rec[ii])/(3*len(self.PV_rec[ii]))

		G1 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		G2 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		G3 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		G4 = np.zeros(shape=(Params.dim,Params.dim,Params.dim,Params.dim),dtype=complex)
		U = self.U
		secular_mat = self.secular_mat
		dim = Params.dim

		code =	"""
				#line 408 "TightBindingTCL2.py"
				using namespace std;
				complex<double> imi(0.0,1.0);
				for (int a = 1 ; a<dim; ++a)
				{
					for (int b = 1; b<dim; ++b)
					{
						for (int c = 1; c<dim; ++c)
						{
							for (int d = 1; d<dim; ++d)
							{
								for (int m = 1; m<dim; ++m)
								{
									if (b == c and secular_mat(a,c,c,d) == 1)
									{
										G1(a,c,c,d) += mytotals(m-1,d-1,c-1) * U(m,a,c) * U(m,c,d);
									}

									if (a == d and secular_mat(c,d,d,b) == 1)
									{
										G4(c,d,d,b) += mytotals(m-1,c-1,d-1) * U(m,c,d) * U(m,d,b);
									}
									if (secular_mat(a,c,d,b) == 1)
									{
										G2(a,c,d,b) += mytotals(m-1,c-1,a-1) * U(m,a,c) * U(m,d,b);
										G3(a,c,d,b) += mytotals(m-1,d-1,b-1) * U(m,a,c) * U(m,d,b);
									}
								}
							}
						}
					}
				}
				"""
		weave.inline(code,
					['G1','G2','G3','G4','mytotals','U','secular_mat','dim'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler, extra_compile_args = Params.Flags,
					verbose = 1)
		self.G1 = G1
		self.G2 = G2
		self.G3 = G3
		self.G4 = G4

	def doublecomm(self, OldState): # This will construct the four parts of the TCL2 term that 
		# come from the double commutator. The four terms are: Ht Hs rho - Ht rho Hs - Hs rho Ht + rho Hs Ht
		term1_abs = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term4_abs = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term2_abs = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term3_abs = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term1_pop = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term4_pop = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term2_pop = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term3_pop = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term1_coh = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term4_coh = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term2_coh = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		term3_coh = np.zeros(shape=(Params.dim,Params.dim),dtype=complex)
		G1 = self.G1
		G2 = self.G2
		G3 = np.conj(self.G3)
		G4 = np.conj(self.G4)
		abs_state = OldState["abs"]
		pop_state = OldState["pop"]
		coh_state = OldState["coh"]
		dim = Params.dim
		# This takes the four gammas and acts on them to obtain the TCL term. 
		code =	"""
				#line 459 "TightBindingTCL2.py"
				using namespace std;
				complex<double> imi(0.0,1.0);
				for (int a = 0 ; a<dim; ++a)
				{
					for (int b = 0; b<dim; ++b)
					{
						for (int c = 0; c<dim; ++c)
						{
							for (int d = 0; d<dim; ++d)
							{
								term1_abs(a,b) += abs_state(d,b) * G1(a,c,c,d);
								term4_abs(a,b) += abs_state(a,c) * G4(c,d,d,b);
								term2_abs(a,b) += abs_state(c,d) * G2(a,c,d,b);
								term3_abs(a,b) += abs_state(c,d) * G3(a,c,d,b);

								term1_pop(a,b) += pop_state(d,b) * G1(a,c,c,d);
								term4_pop(a,b) += pop_state(a,c) * G4(c,d,d,b);
								term2_pop(a,b) += pop_state(c,d) * G2(a,c,d,b);
								term3_pop(a,b) += pop_state(c,d) * G3(a,c,d,b);

								term1_coh(a,b) += coh_state(d,b) * G1(a,c,c,d);
								term4_coh(a,b) += coh_state(a,c) * G4(c,d,d,b);
								term2_coh(a,b) += coh_state(c,d) * G2(a,c,d,b);
								term3_coh(a,b) += coh_state(c,d) * G3(a,c,d,b);
							}
						}
					}
				}
				"""
		weave.inline(code,
					['abs_state','coh_state','pop_state','term1_abs','term4_abs','term2_abs','term3_abs','term1_pop','term4_pop','term2_pop','term3_pop','term1_coh','term4_coh','term2_coh','term3_coh','G1','G2','G3','G4','dim'],
					headers = ["<complex>","<iostream>"],
					global_dict = dict(), 					
					type_converters = scipy.weave.converters.blitz, 
					compiler = Params.Compiler, extra_compile_args = Params.Flags,
					verbose = 1)

		myabs = term1_abs - term2_abs - term3_abs + term4_abs
		mypop = term1_pop - term2_pop - term3_pop + term4_pop
		mycoh = term1_coh - term2_coh - term3_coh + term4_coh

		return [myabs, mypop, mycoh]

	def TCL(self, OldState, t1):
		t0 = self.timesofar
		if t0 != t1:
			if not self.markovian:
				self.useful_exp = np.exp(-2.0 * t1 * np.pi / self.beta)
				self.other_exp = 1/self.useful_exp
				self.exp_itD = np.exp(1j*t1*self.eps)
				self.build_gammas(t1)
			self.timesofar = t1
		return self.doublecomm(OldState)

	def Step(self,OldState,Time,Field = None, AdiabaticMultiplier = 1.0): 
		NewState = OldState.clone()
		NewState.Fill()
		NewState["abs"] += np.dot(self.Hsys,OldState["abs"])
		NewState["abs"] -= np.dot(OldState["abs"],self.Hsys)
		NewState["pop"] += np.dot(self.Hsys,OldState["pop"])
		NewState["pop"] -= np.dot(OldState["pop"],self.Hsys)
		NewState["coh"] += np.dot(self.Hsys,OldState["coh"])
		NewState["coh"] -= np.dot(OldState["coh"],self.Hsys)	
		NewState.MultiplyScalar(complex(0,-1.0))
		TCLterms = self.TCL(OldState,Time)
		NewState["abs"] -= TCLterms[0]
		NewState["pop"] -= TCLterms[1] 
		NewState["coh"] -= TCLterms[2]
		return NewState

