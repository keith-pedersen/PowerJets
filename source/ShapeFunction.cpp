#include "ShapeFunction.hpp"
#include <fstream>

////////////////////////////////////////////////////////////////////////

std::vector<ShapeFunction::real_t> ShapeFunction::hl_Vec(size_t const lMax) const
{
	std::vector<real_t> hl_vec;
	hl_vec.reserve(lMax);
	
	for(size_t l = 1; l <= lMax; ++l)
		hl_vec.push_back(hl(l));
		
	return hl_vec;
}

////////////////////////////////////////////////////////////////////////

std::vector<h_Measured::real_t> h_Measured::ReadFirstColumn(std::string const& filePath)
{
	std::ifstream data(filePath);
	std::vector<real_t> hl_vec;
	std::string line;
	
	if(not data.is_open())
		throw std::ios_base::failure("h_Measured: Cannot open file <" 
			+ filePath + ">.");
	
	// Allow a header commented out with #
	bool header = true;
	
	// NOTE: we assume that l = 0 is not written to this file
	while(std::getline(data, line))
	{
		if(line.empty()) // Stop parsing upon the first empty line
			break;
	
		if(header)
		{
			if(line.front() == '#')
				continue; // Don't parse the header line
			else // The first line not beginning with # character is the end of the header
				header = false;
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
		throw std::runtime_error("h_Measured: l exceeds hl supplied");
	else
	{
		if(l == 0)
			return real_t(1);
		else
			return hl_vec[l-1];
	}
}

////////////////////////////////////////////////////////////////////////

ShapeFunction_Recursive::ShapeFunction_Recursive():
	l_current(size_t(-1)), // Initialize to nonsense value to enforce Reset() call by derived ctor (via assert in hl())
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
		
		// no break, as we can use Delta to return our 1
		case Shape::Delta:
			return real_t(1);
		break;
		
		default:	
			 // This function should only be called on trivial shapes; 
			 // we use an assert to catch a coding error.
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
		assert(l_current not_eq size_t(-1)); // Ensure we've initialized
		
		if(l not_eq l_current)
		{
			if(l < l_current)
			{
				if(l == 0)
					return real_t(1);
				else if (l <= hl_init.size())
					return hl_init[l - 1]; // hl_init stores hl at index = l - 1
				else
				{
					// We assume that we will not be using these classes stupidly, 
					// but we should probably build in something to test for too-frequent reset
					Reset();
				}
			}
			assert(l_current < l); // Sanity check; the control logic says we now need to call Next at least once
		
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
	twoLplus1 = real_t(2*l_current + 1);
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

h_Cap::h_Cap(real_t const surfaceFraction):
	twiceSurfaceFraction(real_t(2) * surfaceFraction)
{
	if((surfaceFraction < real_t(0)) or (surfaceFraction > real_t(1)))
		throw std::runtime_error("h_Cap: surface fraction (" + std::to_string(surfaceFraction) + 
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
	Pl_computer.Next();
	Increment_l();
	
	// Remember, Pl_computer is one l ahead
	assert(Pl_computer.l() == (l_current + 1));	
	
	hl_current = (Pl_computer.P_lm2() - Pl_computer.P_l())/
		(twoLplus1 * twiceSurfaceFraction);
}
	
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
		
void h_Unstable::Reset() const
{
	stable = true;
	
	if(not IsTrivial())
	{
		assert(hl_init.size());
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
		
	// This needs to be done after stable recursion,
	// but before unstable recursion, because h_l = R(l) * h_l-1
	Increment_l();
	
	if(not stable) // this is not an "else" so that we can flow from stable to unstable
	{
		if(hl_current > real_t(0))
		{
			hl_last = hl_current;
			hl_current *= Asym_Ratio();
			
												GCC_IGNORE_PUSH(-Wfloat-equal)
			if(hl_current == hl_last)
			{
				// If hl_current has a de-normalized mantissa,
				// then Asym_Ratio may not be small enough to decrement hl_current. 
				// In that case, we need to round down to zero
				// (otherwise the recursion will get stuck here)
				assert(hl_current <= std::numeric_limits<real_t>::min()); // smallest normalized number
				hl_current = real_t(0);
			}
												GCC_IGNORE_POP
		}
	}		
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::h_Gaussian(real_t const lambda_in):
	lambda(lambda_in),
	lambda2(kdp::Squared(lambda))
{
	if(lambda < real_t(0))
		throw std::runtime_error("h_Gaussian: lambda (" + std::to_string(lambda) + 
			") must be positive.");
	
	Setup();
	Reset();
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::h_Gaussian(real_t const R, real_t const u):
	h_Gaussian(SolveLambda(R, u)) {}

////////////////////////////////////////////////////////////////////////

void h_Gaussian::Setup()
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

bool h_Gaussian::IsUnstable(real_t const hl_next) const
{
	return ((hl_next < gettingSmall) 
		or (hl_next >= hl_current)); // monotonically decreasing
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::real_t h_Gaussian::Asym_Ratio() const
{
	real_t const term = twoLplus1 * lambda2;
	return real_t(2)/(term + std::sqrt(kdp::Squared(term) + real_t(4)));				
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::real_t h_Gaussian::h_lp1() const
{
	return -twoLplus1 * lambda2 * hl_current + hl_last;
}

////////////////////////////////////////////////////////////////////////
	
h_Gaussian::real_t h_Gaussian::SolveLambda(real_t const R, real_t const u)
{
	// We have a transcendental equation for lambda, which we can solve in a few iterations
	
	if((R <= 0.) or (R >= M_PI))
		throw std::runtime_error("h_Gaussian::SolveLambda: R must belong to (0, Pi)");
	if((u <= 0.) or (u >= 1.))
		throw std::runtime_error("h_Gaussian::SolveLambda: u must belong to (0, 1)");
	
	real_t lambda_last, lambda = real_t(0);
	real_t const sinHalfAngle = std::sin(real_t(0.5)*R);
	
	size_t constexpr maxIterations = 1000;
	size_t i = 0;	
													GCC_IGNORE_PUSH(-Wfloat-equal)
	do	
	{
		if(++i > maxIterations)
			throw std::runtime_error("h_Gaussian::SolveLambda: lambda not stabilizing after many iterations!");
		
		lambda_last = lambda;
		
		lambda = sinHalfAngle * std::sqrt(-real_t(2)/
			std::log1p(u * std::expm1(-real_t(2)/kdp::Squared(lambda))));
	}
	while(lambda not_eq lambda_last);
													GCC_IGNORE_POP
	return lambda;
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
	return ((hl_next < gettingSmall) or ((hl_next / hl_current) < R_asym) 
		 or (hl_next >= hl_current)); // monotonically decreasing
}

////////////////////////////////////////////////////////////////////////

h_Boost::real_t h_Boost::Asym_Ratio() const
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

//~ h_Boost_orig::h_Boost_orig(vec3_t const& p3, real_t const mass):
	//~ beta(vec4_t::BetaFrom_Mass_pSquared(mass, p3.Mag2()))
//~ {
	//~ assert(beta >= real_t(0));
	//~ assert(beta <= real_t(1));
//~ }

//~ void h_Boost_orig::Setup(size_t const lMax) const
//~ {
	//~ if(beta == real_t(1))
		//~ onAxis[1] = real_t(1);
	//~ else
	//~ {
		//~ real_t const h1 = real_t(1) + ((real_t(1) - beta)/beta)*
			//~ (real_t(1) - (real_t(1) + beta)/beta * std::atanh(beta));
		
		//~ assert(h1 >= real_t(0));
		//~ assert(h1 < real_t(1));
		
		//~ onAxis[1] = h1;
	//~ }
	
	//~ l=1;
//~ }

//~ h_Boost_orig::real_t h_Boost_orig::Asym_Ratio() const
//~ {
	//~ return real_t(2)*beta*lPlus1 / 
	//~ (twoLplus1 + std::sqrt(kdp::Squared(twoLplus1) -
		//~ kdp::Squared(real_t(2) * beta) * lPlus1*(lPlus1 - real_t(1))));
//~ }

//~ h_Boost_orig::real_t h_Boost_orig::NotIsotropic() const
//~ {
	//~ return (beta > 1e-6);
//~ }

//~ void h_Boost_orig::h_lp1() const
//~ {
	//~ onAxis[l+1] = (twoLplus1*onAxis[l] - beta*lPlus1*onAxis[l-1])/(beta*(lPlus1 - real_t(1)));
//~ }
		
//~ //! @brief The recursive half-power of the particle smeared in Gaussian angle for l=0 to l=lMax
//~ template <typename real_t>
//~ std::vector<real_t> Q_l(size_t const lMax, real_t const lambda, 
	//~ real_t const threshold = real_t(1e-6))
//~ {
	//~ real_t const lambda2 = kdp::Squared(lambda);
	
	//~ std::vector<real_t> Q;
	//~ Q.resize(lMax + 1, real_t(0)); // Fill with zeroes
	
	//~ Q[0] = real_t(1);	
		
	//~ if(lambda < 1e3) // Otherwise, the smear angle is so large Q[1] is numerically unstable
	//~ {
		//~ // 1/tanh(1/lambda**2) = (1 + exp(-2/lambda**2))/(1 - exp(-2/lambda**2))
		//~ // (1 + exp(-x))/(1-exp(-x)) = 1 + 2*exp(-x)/(1-exp(-x) = 1 + 2/(exp(x)-1)
		//~ // Note that (1-x)*(1+x) is more accurate when x > 0.5, and 1-x**2 when x < 0.5
		//~ // When lambda**2 > 0.5 * 1/tanh(2/lambda**2), we should do it the first way
		//~ // This occurs when lambda2 is about 0.5
		
		//~ if(lambda2 > real_t(0.5))
			//~ Q[1] = real_t(1)/std::tanh(real_t(1)/lambda2) - lambda2;
		//~ else
			//~ Q[1] = kdp::Diff2(real_t(1), lambda) + real_t(2)/std::expm1(real_t(2)/lambda2);
		
		//~ // l = 1, emplace l + 1 every iteration
		//~ size_t l = 1;
		//~ real_t twoLplus1 = real_t(3); // 2*1 + 1
		
		//~ bool stable = true;
		
		//~ while(l < lMax)
		//~ {
			//~ if(stable)
			//~ {
				//~ Q[l+1] = -twoLplus1*lambda2*Q[l] + Q[l-1];
				
				//~ assert(Q[l+1] > real_t(0));
				//~ if(Q[l+1] < threshold)
					//~ stable = false; // Switch to stable within this iteration
			//~ }
			
			//~ // This needs to be done after stable recusion, 
			//~ // but before unstable recursion, because Q_l = R(l) * Q_l-1
			//~ twoLplus1 += real_t(2);
			//~ ++l;
			
			//~ if(not stable)
			//~ {
				//~ real_t const term = twoLplus1 * lambda2;
				//~ real_t const R_l = real_t(2)/(term + std::sqrt(kdp::Squared(term) + real_t(4)));
				
				//~ Q[l] = R_l * Q[l-1];
				//~ if(Q[l] == real_t(0))
					//~ break;
			//~ }
		//~ }
	//~ }
	
	//~ return Q;
//~ }

//~ template <typename real_t>
//~ std::vector<real_t> Q_l_squared(size_t const lMax, real_t const lambda, 
	//~ real_t const threshold = real_t(1e-6))
//~ {
	//~ std::vector<real_t> Q = Q_l(lMax, lambda, threshold);
	
	//~ for(auto& Q_l : Q)
		//~ Q_l = kdp::Squared(Q_l);
		
	//~ return Q;
//~ }
