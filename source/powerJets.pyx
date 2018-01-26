# This is a Cython wrapper for the powerJets C++ package
'''Compute the power spectrum H_l from an n-jet model.

	This package is under active development, and it's API is subject to dramatic changes.
'''

########################################################################
########################################################################
########################################################################
# We first declare the C++ functions we intend to call
########################################################################

#~ from libc.stdint cimport int64_t
#~ from libcpp cimport bool # access bool as bool
from libcpp.string cimport string # access std::string as string
#~ from libc.string cimport memcpy
from libcpp.vector cimport vector # access std::vector<T> as vector[T]
from cython.operator cimport dereference as deref 
# Dereference Cython pointers via deref() (e.g. func(deref(pointer)),
# to pass an object as argument. Note: pointer.func() has implicit dereference)

import numpy
cimport numpy

#####################################################################

cdef StdVecToNumpy(const vector[double]& stdVec):
	# Using this "C API" for numpy is deprecated, as the compiler warns,
	# but I don't know any alternative
	cdef numpy.ndarray[double, ndim = 1, mode="c"] numpyVec = numpy.empty(stdVec.size())
	cdef double* vec = <double*>numpyVec.data
	
	# The memcpy appears to the be no faster than the loop, and is less readable
#~ 	memcpy(<void*>vec, <void*>stdVec.data(), stdVec.size()*sizeof(real_t))
	for i in range(stdVec.size()):
		vec[i] = stdVec[i]
	return numpyVec
	
#####################################################################

import kdp
cimport kdp

include "kdp/kdpVectors.hpy"
#~ include "/home/keith/Documents/Projects/libkdp/source/kdp.pyx"

#~ cimport kdp

########################################################################
# import pqRand.hpp

# To allow the objects to have the same name in Cython and C++,
# we import as **_c, giving the full C++ name in quotes
cdef extern from "NjetModel.hpp":
	cdef cppclass Jet:
		Vec4_c p4
		double mass
		
	cdef cppclass ShapedJet:
		Vec4_c p4
		double mass
		
		ShapedJet()
		ShapedJet(const double, const double, const double, const double, 
			const Vec4from2, const vector[double]&) except +
		ShapedJet(const Vec3_c&, const double, const Vec4from2, const vector[double]&) except +
		
	cdef cppclass ShowerParticle:		
		# Because we want to catch and return C++ exceptions (except+),
		# Cython requires a nullary constructor due to its automatic coding paradigm.
		# default assignment (shallow copy) of a binary tree can create pointer errors.
		# There are two solutions. a = b is equivalent to std::swap(a, b), 
		# or switch to shared_ptr for b and c. 
		ShowerParticle(const vector[double]&, const vector[vector[bool]]&) except +
		
		vector[ShapedJet] GetJets() 
				
	cdef cppclass NjetModel_c "NjetModel":
		NjetModel_c(const string& iniFileName) except +
		
		vector[double] H_l(vector[ShapedJet]& const, size_t const, double const) except +
		
cdef extern from "SpectralPower.hpp":
	cdef cppclass PhatF "SpectralPower:PhatF":
		pass	
		
cdef extern from "LHE_Pythia_PowerJets.hpp":
	cdef cppclass Status "LHE_Pythia_PowerJets::Status":
		pass
		
	cdef Status Status_OK "LHE_Pythia_PowerJets::Status::OK"
	cdef Status Status_UNINIT "LHE_Pythia_PowerJets::Status::UNINIT"
	cdef Status Status_END_OF_FILE "LHE_Pythia_PowerJets::Status::END_OF_FILE"
	cdef Status Status_EVENT_MAX "LHE_Pythia_PowerJets::Status::EVENT_MAX"
	cdef Status Status_ABORT_MAX "LHE_Pythia_PowerJets::Status::ABORT_MAX"
			
	cdef cppclass LHE_Pythia_PowerJets:
		LHE_Pythia_PowerJets(const string&) except +
		
		size_t EventIndex()		
		Status GetStatus()
		
		const vector[PhatF]& GetDetected()		
		const vector[double]& Get_H_det()		
		const vector[Jet]& Get_FastJets()
		
		Status Next()
			
########################################################################		
# Now we make the Cython wrapper class of the C++ object

########################################################################		
# Python 3 uses unicode str; Python 2 uses ASCII strings; C++ std::string is ASCII only.
# Thus, Python 2 is a naive conversion between std::string and str, 
# whereas Python 3 needs to explicitly "encode"/"decode" the unicode.
# To allow this package to work with either Python version, we define two hidden helper functions
import sys

if(sys.version_info < (3,0)):
	def _str2string(s):
		return s
	def _string2str(s):
		return str(s)
else:
	def _str2string(s):
		return s.encode()
	def _string2str(s):
		return s.decode()
	
#####################################################################	

from enum import Enum
class JetParamStyle(Enum):
	Momentum = 1
	Splitting = 2

#####################################################################

from enum import Enum	
class JetMomentumStyle(Enum):
	Massless = 1
	Direct = 2

#####################################################################
	
pileup_key = "pileup_frac"

# Cython default initializes the C++ object using the default ctor
# Howerver, engine() causes the default seeding, which we do not always want.
# To ellude the auto-default behavior, we must define __cinit__ and __dealloc__, 
# which in this case requires storing a pointer to the C++ object

cdef class NjetModel:
	cdef NjetModel_c* c_model
	
	#####################################################################
	
	def __cinit__(self, str iniFileName = ""):
		self.c_model = new NjetModel_c(_str2string(iniFileName))
	
	#####################################################################
	
	def __dealloc__(self):
		del self.c_model
	
	#####################################################################
	
	@staticmethod
	cdef vector[ShapedJet] _GetJets_Momentum(numpy.ndarray[double, ndim=1] jetParams, dict jetParamDict, 
		const int nShapeParams) except+:
		
		cdef int numParams_perJet = 4 + nShapeParams
			
		balancingJet = jetParamDict.get('force_CM', True)
		cdef int maxParams = len(jetParams)
		if(balancingJet): 
			maxParams -= 1
		
		momentumStyle = jetParamDict.get('p3_style', JetMomentumStyle.Massless)
		# JetMomentumStyle
		# 		massless:		[p1, p2, p3, gamma] 	(the momentum if the jet is massless)
		# 		direct:		[p1. p2, p3, mass] 	(the momentum is supplied directly)
		# If balancing jet, the final parameter follows the mass convention
						
		cdef vector[ShapedJet] jetVec
		cdef Vec4_c total	
		cdef Vec4from2 style = V4f2_Boost_preserve_E if (momentumStyle == JetMomentumStyle.Massless) else V4f2_Mass
						
		if balancingJet:
			if((len(jetParams) % numParams_perJet) != 1):
				raise ValueError("NjetModel.GetJets: Momentum basis. Looking for 4 params per jet + mass/boost of balancing jet")
				
		elif ((len(jetParams) % numParams_perJet) != 0):
			raise ValueError("NjetModel.GetJets: Momentum basis. Looking for 4 params per jet (and no balancing jet)")
		
		for i in range(0, maxParams, numParams_perJet):
			jetVec.push_back(ShapedJet(
				jetParams[i + 0], jetParams[i + 1], jetParams[i + 2],
				jetParams[i + 3], style, jetParams[i + 4: i + 4 + nShapeParams]))
				
			total += jetVec.back().p4
			
		if balancingJet:
			# For both massless and direct momenta, balancing jet's mass is parameterized by gamma,
			# though this time it preserves momentum. For now, this jet has no shape.
			jetVec.push_back(ShapedJet(-total.p(), jetParams[-1], V4f2_Boost_preserve_p3, vector[double]()))
			
		return jetVec
			
	#####################################################################
	
	@staticmethod
	cdef vector[ShapedJet] _GetJets_Splitting(numpy.ndarray[double, ndim=1] jetParams, dict jetParamDict, 
		const int nShapeParams) except+:
		
		return ShowerParticle(jetParams, jetParamDict.get("addresses", list())).GetJets()
			
	#####################################################################
	
	@staticmethod
	cdef vector[ShapedJet] _GetJets(numpy.ndarray[double, ndim=1] jetParams, dict jetParamDict) except+:
		
		######################
		# discard jet fraction
		if (jetParamDict.get(pileup_key, True)):
			jetParams = jetParams[1:]
			
		#######################
		# determine param style
		paramStyle = jetParamDict.get('param_style', JetParamStyle.Momentum)
		
		cdef int nShapeParams = jetParamDict.get('num_shape_params', 0)
		if(nShapeParams < 0):
			raise ValueError("NjetModel.GetJets: num_shape_params must be non-negative")		
		
		if(paramStyle == JetParamStyle.Momentum):
			return NjetModel._GetJets_Momentum(jetParams, jetParamDict, nShapeParams)
		elif(paramStyle == JetParamStyle.Splitting):
			return NjetModel._GetJets_Splitting(jetParams, jetParamDict, nShapeParams)
		else:
			raise ValueError("NjetModel: unrecognized 'param_style' in jetParamDict")		
			
	#####################################################################
	
	@staticmethod
	def GetJets(jetParams, dict jetParamDict):
		# Use the internal function to get a vector[Jet], then copy the 4-vectors into newly minted Vec4
		cdef vector[ShapedJet] jetVec = NjetModel._GetJets(numpy.asarray(jetParams), jetParamDict)
		
		# I don't know why, but for some reason Cython complains that it can't 
		# convert from Vec4_c to const Vec4_c, so we have to cast the const-ness ... weird
		return [kdp.Vec4.Factory(<const Vec4_c&>jet.p4) for jet in jetVec]
	
	#####################################################################

	# Formerly __call__, the operator() in C++
	def H_l(self, jetParams, jetParamDict = {}, lMax=128, granularity=1e3):
		''' Given jetParams in the correct format (see GetJets),
			return the power specturm H_l_jet.
			
			jetShapeGranularity is the approximate number of random numbers drawn.'''
		if(lMax < 0):
			raise ValueError("NjetModel(): lMax must be non-negative.")
		
		jetFraction = 1.
		if (jetParamDict.get(pileup_key, True)):
			jetFraction -= jetParams[0]
				
		return StdVecToNumpy(self.c_model.H_l(NjetModel._GetJets(numpy.asarray(jetParams), jetParamDict), 
			<size_t> lMax, <double> granularity)) * jetFraction**2
			
	#####################################################################

	def H_l_jet_error_vec(self, jetParams, H_l_obs, jetShapeGranularity, relative = False):
		''' Given jetParams in the correct format (see GetJets) and
			H_l_obs = {H_1, H_2, ..., H_n}, 
			return the vector of residuals (chi_l = H_l_jet - H_l_obs).
			
			jetShapeGranularity is the approximate number of random numbers drawn.
			
			If relative = True, the residuals are relative.'''
			
		H_l_fit = self.H_l(jetParams, len(H_l_obs), jetShapeGranularity)
		
		residualsF = (lambda H_obs, H_fit: (H_fit - H_obs)/H_obs) if relative else \
			(lambda H_obs, H_fit: (H_fit - H_obs))
		
		return numpy.asarray(list(map(residualsF, H_l_obs, H_l_fit)))		
