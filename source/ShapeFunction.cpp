// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "ShapeFunction.hpp"
#include <fstream>

////////////////////////////////////////////////////////////////////////
// ShapeFunction
////////////////////////////////////////////////////////////////////////

std::vector<ShapeFunction::real_t> ShapeFunction::hl_Vec(size_t const lMax) const
{
	std::vector<real_t> hl_vec;
	hl_vec.reserve(lMax);
	
	for(size_t l = 1; l <= lMax; ++l)
		hl_vec.push_back(hl(l)); // simply use hl
		
	return hl_vec;
}

////////////////////////////////////////////////////////////////////////
// h_Delta
////////////////////////////////////////////////////////////////////////

std::shared_ptr<ShapeFunction> h_Delta::Clone() const
{
	return Make<h_Delta>();
}

////////////////////////////////////////////////////////////////////////
// h_Measured
////////////////////////////////////////////////////////////////////////

std::vector<h_Measured::real_t> h_Measured::ReadFirstColumn(std::string const& filePath)
{
	std::ifstream data(filePath);
	
	if(not data.is_open())
		throw std::ios_base::failure("h_Measured: Cannot open file <" 
			+ filePath + ">.");
			
	std::vector<real_t> hl_vec;
	std::string line;
	
	// Allow a header commented out with #
	bool inHeader = true;
	
	// NOTE: we assume that l = 0 is not written to this file
	while(std::getline(data, line))
	{
		if(line.empty()) // Stop parsing upon the first empty line
			break;
	
		if(inHeader)
		{
			if(line.front() == '#')
				continue; // Don't parse the header lines
			else // The first line not beginning with # character is the end of the header
				inHeader = false;
		}
		
		// If the line cannot be interpreted as a double, this throws an exception
		hl_vec.emplace_back(std::stod(line, nullptr));
	}
	
	return hl_vec;
}

////////////////////////////////////////////////////////////////////////

h_Measured::real_t h_Measured::hl(size_t const l) const
{
	if(l > lMax())
		throw std::runtime_error("h_Measured: l exceeds supplied coefficients");
	else
	{
		if(l == 0)
			return real_t(1);
		else
			return hl_vec[l-1];
	}
}

////////////////////////////////////////////////////////////////////////

std::shared_ptr<ShapeFunction> h_Measured::Clone() const
{
	return Make<h_Measured>(*this);
}

////////////////////////////////////////////////////////////////////////
// ShapeFunction_Recursive
////////////////////////////////////////////////////////////////////////

ShapeFunction_Recursive::ShapeFunction_Recursive():
	l_current(size_t(-1)), // Initialize to nonsense value to enforce Reset() call by derived ctor
	shape(Shape::NonTrivial) // default to NonTrivial; derived ctor identifies trivial
{}
	
////////////////////////////////////////////////////////////////////////

ShapeFunction_Recursive::real_t ShapeFunction_Recursive::hl_trivial(size_t const l) const
{
	switch(shape)
	{
		case Shape::Isotropic:
			if(l > 0)
				return real_t(0);
		
		// no break, as we can use Delta to return our h_0 = 1 for Isotropic
		case Shape::Delta:
			return real_t(1);
		break;
		
		default:	
			 // This function should only be called on trivial shapes; 
			 // we use an assert to catch a logic error in the code.
			assert(false);
			return real_t(INFINITY); // To suppress warning about no return
	}
}
	
////////////////////////////////////////////////////////////////////////

ShapeFunction_Recursive::real_t ShapeFunction_Recursive::hl(size_t const l) const
{
	if(IsTrivial())
		return hl_trivial(l);
	else
	{
		assert(l_current not_eq size_t(-1)); // Ensure Reset() has been called at least once
		
		// If l == l_current, we skip the work and return the cached value
		if(l not_eq l_current)
		{
			if(l < l_current) // We cannot iterate backwards; only forward
			{
				if(l == 0)
					return real_t(1);
				else if (l <= hl_init.size())
					return hl_init[l - 1]; // hl_init stores hl at index = l - 1
				else
					Reset(); // Reset the recursion to calculate a lower l
			}
			assert(l_current < l); // Sanity check; we should need to call Next() at least once
		
			do
				Next();
			while(l_current < l);
		}
		
		return hl_current;
	}
}

////////////////////////////////////////////////////////////////////////

void ShapeFunction_Recursive::Increment_l() const
{
	++l_current;
	twoLplus1 += real_t(2);
}

////////////////////////////////////////////////////////////////////////

void ShapeFunction_Recursive::Set_l(size_t const l) const
{
	l_current = l;
	twoLplus1 = real_t(2*l + 1);
}

////////////////////////////////////////////////////////////////////////
// h_Cap
////////////////////////////////////////////////////////////////////////

h_Cap::h_Cap(real_t const surfaceFraction):
	twiceSurfaceFraction(real_t(2) * surfaceFraction)
{
	if((surfaceFraction < real_t(0)) or (surfaceFraction > real_t(1)))
		throw std::domain_error("h_Cap: surface fraction (" + std::to_string(surfaceFraction) + 
			") must exist in the unit inverval [0, 1]");
																			GCC_IGNORE_PUSH(-Wfloat-equal)
	// Check for trivial shapes
	if(surfaceFraction == real_t(0))
		shape = Shape::Delta;
	else if(surfaceFraction == real_t(1))
		shape = Shape::Isotropic;
																			GCC_IGNORE_POP	
	Reset();
}

////////////////////////////////////////////////////////////////////////

void h_Cap::Reset() const
{
	Pl_computer.Setup(real_t(1) - twiceSurfaceFraction);
	
	// h_l = (P_{l-1} - P_{l+1}) / (2 A_twr * (2l+1))
	// RecursiveLegendre has access to l, l-1, and l-2, 
	// so we must keep Pl_computer's l one ahead of l_current. 
	// Since Pl_computer starts at l = 1, we can start with l_current = 0
	Set_l(0);
	assert(Pl_computer.l() == 1);
	
	hl_current = real_t(1);
}

////////////////////////////////////////////////////////////////////////

void h_Cap::Next() const
{
	assert(not IsTrivial()); // Next should never be called on a trivial shape
	
	Pl_computer.Next();
	Increment_l();
	
	// Verify that Pl_computer is one l ahead
	assert(Pl_computer.l() == (l_current + 1));	
	
	hl_current = (Pl_computer.P_lm2() - Pl_computer.P_l())/
		(twoLplus1 * twiceSurfaceFraction);
}

////////////////////////////////////////////////////////////////////////

std::shared_ptr<ShapeFunction> h_Cap::Clone() const
{
	return Make<h_Cap>(*this);
}
	
////////////////////////////////////////////////////////////////////////
// h_Unstable
////////////////////////////////////////////////////////////////////////
		
void h_Unstable::Reset() const
{
	stable = true;
	
	if(not IsTrivial())
	{
		assert(hl_init.size()); // We must have values to initialize the recursion
		hl_current = hl_init.back();
		hl_last = (hl_init.size() == 1) ? real_t(1) : hl_init[hl_init.size() - 2];
		Set_l(hl_init.size());
	}
}

////////////////////////////////////////////////////////////////////////
	
void h_Unstable::Next() const
{
	assert(not IsTrivial()); // Next should never be called on a trivial shape
	
	if(stable)
	{
		real_t const hl_next = h_lp1(); // iterate
		
		if(IsUnstable(hl_next))
			stable = false;
		else
		{
			hl_last = hl_current;
			hl_current = hl_next;
		}
	}
		
	// This needs to be done after stable recursion (which uses h_{l+1} = ...),
	// but before unstable recursion (which uses h_l = R_l * h_l-1)
	Increment_l();
	
	if(not stable) // this is not an "else" so that we can flow from stable directly to unstable
	{
		if(hl_current > real_t(0))
		{
			{
				real_t const Rl = R_l();
				assert(Rl < real_t(1));			
				
				hl_last = hl_current;
				hl_current *= Rl;
			}
												GCC_IGNORE_PUSH(-Wfloat-equal)
			if(hl_current == hl_last)
			{
				// If hl_current has a de-normalized mantissa,
				// then R_l may not be small enough to decrement hl_current. 
				// In that case, we need to round down to zero
				// (otherwise the recursion will get stuck at the de-normalized value)
				assert(hl_current <= std::numeric_limits<real_t>::min()); // smallest normalized number
				hl_current = real_t(0);
			}
												GCC_IGNORE_POP
		}
	}
}

////////////////////////////////////////////////////////////////////////
// h_PseudoNormal
////////////////////////////////////////////////////////////////////////

h_PseudoNormal::h_PseudoNormal(real_t const lambda_in):
	lambda(lambda_in),
	lambda2(kdp::Squared(lambda))
{
	if(lambda < real_t(0))
		throw std::runtime_error("h_PseudoNormal: lambda (" + std::to_string(lambda) + 
			") must be positive.");
	
	Setup();
	Reset();
}

////////////////////////////////////////////////////////////////////////

h_PseudoNormal::h_PseudoNormal(real_t const R, real_t const u):
	h_PseudoNormal(SolveLambda(R, u)) {}

////////////////////////////////////////////////////////////////////////

void h_PseudoNormal::Setup()
{
																GCC_IGNORE_PUSH(-Wfloat-equal)
	// If lambda**2 is infinity, we can't calculate hl_init, 
	// so the shape is effectively isotropic
	if(lambda2 == real_t(INFINITY))
		shape = Shape::Isotropic;
	else
	{
		// 1/tanh(1/lambda**2) = (1 + exp(-2/lambda**2))/(1 - exp(-2/lambda**2))
		// (1 + exp(-x))/(1-exp(-x)) = 1 + 2*exp(-x)/(1-exp(-x) = 1 + 2/(exp(x)-1)
		// Note that (1-x)*(1+x) is more accurate when x > 0.5, and 1-x**2 when x < 0.5
		// When lambda**2 > 0.5 * 1/tanh(2/lambda**2), we should do it the first way
		// This occurs when lambda2 is about 0.5

		if(lambda2 > real_t(0.5))
			hl_init = {real_t(1)/std::tanh(real_t(1)/lambda2) - lambda2};
		else
			hl_init = {kdp::Diff2(real_t(1), lambda) + real_t(2)/std::expm1(real_t(2)/lambda2)};
			
		assert(hl_init.front() <= real_t(1));
																			
		// Check for trivial shapes
		if(hl_init.front() == real_t(1))
			shape = Shape::Delta;
		else if(hl_init.front() <= real_t(0))
		{
			hl_init.clear();
			shape = Shape::Isotropic;
		}
	}
																GCC_IGNORE_POP
}

////////////////////////////////////////////////////////////////////////

bool h_PseudoNormal::IsUnstable(real_t const hl_next) const
{
	return ((hl_next < gettingSmall) 
		or (hl_next >= hl_current)); // h_l should be monotonically decreasing
}

////////////////////////////////////////////////////////////////////////

h_PseudoNormal::real_t h_PseudoNormal::R_l() const
{
	real_t const term = twoLplus1 * lambda2;
	return real_t(2)/(term + std::sqrt(kdp::Squared(term) + real_t(4)));				
}

////////////////////////////////////////////////////////////////////////

h_PseudoNormal::real_t h_PseudoNormal::h_lp1() const
{
	return -twoLplus1 * lambda2 * hl_current + hl_last;
}

////////////////////////////////////////////////////////////////////////
	
h_PseudoNormal::real_t h_PseudoNormal::SolveLambda(real_t const R, real_t const u)
{
	// We have a transcendental equation for lambda, which we can solve in a few iterations
	
	if((R < 0.) or (R > M_PI))
		throw std::domain_error("h_PseudoNormal::SolveLambda: R must belong to (0, Pi)");
	if((u <= 0.) or (u > 1.))
		throw std::domain_error("h_PseudoNormal::SolveLambda: u must belong to (0, 1)");
	
	real_t lambda_last, lambda = real_t(0);
	real_t const sinHalfAngle = std::sin(real_t(0.5)*R);
	
	// We hard code in a relatively small max, assuming reasonable values of R and u
	size_t constexpr maxIterations = 1000;
	size_t i = 0;	
													GCC_IGNORE_PUSH(-Wfloat-equal)
	do	
	{
		if(++i > maxIterations)
			throw std::runtime_error("h_PseudoNormal::SolveLambda: lambda not stabilizing after many iterations!");
		
		lambda_last = lambda;
		
		// If R = 0 or u = 1, this expression should immediately stabilize to lambda = 0
		lambda = sinHalfAngle * std::sqrt(-real_t(2)/
			std::log1p(u * std::expm1(-real_t(2)/kdp::Squared(lambda))));
	}
	while(lambda not_eq lambda_last);
													GCC_IGNORE_POP
	return lambda;
}

////////////////////////////////////////////////////////////////////////

std::shared_ptr<ShapeFunction> h_PseudoNormal::Clone() const
{
	return Make<h_PseudoNormal>(*this);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

h_Boost::h_Boost():
	m2(real_t(0)), p2(real_t(0)),	beta(real_t(1))
{
	Setup(); // No need to reset, it's a delta distribution
}	

////////////////////////////////////////////////////////////////////////

h_Boost::h_Boost(vec3_t const& p3, real_t const mass):
	m2(kdp::Squared(mass)),
	p2(p3.Mag2()),
	beta(vec4_t::Beta(p3, mass))
{
	Setup();
	Reset();
}

////////////////////////////////////////////////////////////////////////

void h_Boost::Setup()
{
	assert(beta >= real_t(0));
	assert(beta <= real_t(1));
																			GCC_IGNORE_PUSH(-Wfloat-equal)
	if(beta == real_t(0))
		shape = Shape::Isotropic; // TODO: we will need to adjust this when we add arbitrary shape function
	else if(beta == real_t(1))
		shape = Shape::Delta;
	else
	{
		real_t const term = std::sqrt((m2 + p2)*p2);		
		real_t const h2 = real_t(1) - (real_t(3) * m2)/(real_t(2) * p2) * 
			(real_t(1) - real_t(0.5) * (m2 / term) * 
			std::log1p(real_t(2) * (p2 + term) / m2));
		//~ R_current = hl_current / hl_last;
		
		assert(h2 <= real_t(1));
		
		R_asym = beta / (real_t(1) + std::sqrt(kdp::Diff2(real_t(1), beta)));		
		hl_init = {beta, (h2 < real_t(0)) ? real_t(0) : h2};
	}
																			GCC_IGNORE_POP
}

////////////////////////////////////////////////////////////////////////

bool h_Boost::IsUnstable(real_t const hl_next) const
{
	return ((hl_next < gettingSmall) 
		or (hl_next < R_asym * hl_current) // we should find (hl_next / hl_current) >= R_asym
		or (hl_next >= hl_current)); // h_l should be monotonically decreasing
}

////////////////////////////////////////////////////////////////////////

h_Boost::real_t h_Boost::R_l() const
{
	return real_t(2 * (l_current + 2)) * beta / 
	(twoLplus1 + std::sqrt(kdp::Squared(twoLplus1) - 
		real_t(4) * kdp::Squared(beta) * real_t((l_current - 1) * (l_current + 2))));
}

////////////////////////////////////////////////////////////////////////

h_Boost::real_t h_Boost::h_lp1() const
{
	return (twoLplus1*hl_current - beta*(real_t(l_current + 2)*hl_last))/
		(beta*(real_t(l_current - 1)));
}

////////////////////////////////////////////////////////////////////////

std::shared_ptr<ShapeFunction> h_Boost::Clone() const
{
	return Make<h_Boost>(*this);
}		
