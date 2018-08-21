#ifndef POWER_JETS
#define POWER_JETS

// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

//~ #define NDEBUG
// The assertions seem to occupy a minimal amount of time

#include "kdp/kdpVectors.hpp"

/*! @file PowerJets.hpp
 *  @brief Defines the PowerJets namespace, which is currently used only 
 *  for a couple of typedef's 
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
*/ 

/*! @mainpage PowerJets
 *  
 *  @brief A C++ and Python package for researching and using the 
 *  dimensionless, angular power spectrum of QCD jets.
 * 
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
 * 
 *  \warning As of 20-Aug-2018, this git repo is in active development, 
 *  and is being checked for completeness and portability on Linux systems.
 *  Expect multiple commits in the coming weeks as more code is added to the repo.
 * 
 *  \note This package requires installing two of my other packages
 *  (libkdp and pqRand), both of which are available on my GitHub page 
 *  (https://www.github.com/keith-pedersen). I have designed my system 
 *  so that "~/local/include" contains soft links to the "include" 
 *  directories of the other packages (renaming them "kdp" and "pqRand", respectively).
 *  Similarly, "~/local/lib" contains soft links to their shared libraries.
 *  
 *  My research into the power spectrum is an attempt to develop 
 *  new tools to analyze particle physics events at the LHC and beyond.
 *  Instead of analyzing QCD by building jets from primarily local information, 
 *  the power spectrum looks towards global correlations for new insights. 
 *  Fundamentally, the power spectrum scans the whole event and 
 *  determines how much energy is separated at different angles
 *  (very much like how cosmologists study the CMB).
 *  Extracting useful information from this spectrum requires 
 *  determining its limits of information, a process described in detail in 
 *  "Harnessing the global correlations of the QCD power spectrum."
 *  It is then possible to fit the result to a jet-like model that 
 *  can easily accommodate important effects like pileup.
 *  
 *  Most of the heavy lifting is done in C++. However, C++ lacks
 *  a good tool for doing non-linear fits without having to 
 *  manually set up the Jacobian and all the overhead.
 *  Thus, the final fit to the jet-like model is accomplished by 
 *  porting the necessary classes to Python (via Cython) 
 *  and performing the fit using scipy.optimize.least_squares
 *  (which still relies quite heavily on the C++ code under-the-hood).
 * 
 *  \note The main working class for calculating the power spectrum is 
 *  PowerSpectrum, which is also a good template for some 
 *  very useful concepts in high-performance computing.
 * 
 *  This package was developed as part of my thesis research, 
 *  and is mostly free under the MIT license.
 *  Code developed from existing source files are licensed appropriately
 *  (e.g. LHE_Pythia_PowerJets).
*/  

namespace PowerJets
{
	typedef double real_t;
	typedef kdp::Vector2<real_t> vec2_t;
	typedef kdp::Vector3<real_t> vec3_t;
	typedef kdp::Vector4<real_t> vec4_t;
}

#endif
