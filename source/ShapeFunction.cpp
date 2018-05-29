#include "ShapeFunction.hpp"
#include <iostream>

// To check that Reset is called, we can initialize l_current to a non-sensical value, 
// then assert that l_current is sensible when hl is called

////////////////////////////////////////////////////////////////////////

std::vector<ShapeFunction::real_t> ShapeFunction::hl_Vec(size_t const lMax) const
{
	// The whole point of this class is that resting and repeating the recursion is
	// better than trying to cache it, so we will alter the state of this object
	
	std::vector<real_t> hl_vec;
	hl_vec.reserve(lMax);
	
	// Use the implicit Reset inside hl
	for(size_t l = 1; l <= lMax; ++l)
		hl_vec.push_back(hl(l));
		
	return hl_vec;
}

////////////////////////////////////////////////////////////////////////

ShapeFunction_Recursive::ShapeFunction_Recursive():
	l_current(size_t(-1)) {} // Initialize to nonsense value to enforce Reset() call by derived ctor (via assert in hl())
	
////////////////////////////////////////////////////////////////////////

ShapeFunction_Recursive::real_t ShapeFunction_Recursive::hl(size_t const l) const
{
	assert(l_current not_eq size_t(-1));
	
	if(l not_eq l_current)
	{
		if(l < l_current)
		{
			if(l == 0)
				return real_t(1);
			else if (l <= hl_init.size())
			{
				// hl_init stores hl at index = l - 1
				return hl_init[l - 1];
			}
			else
			{
				// We assume that we will not be using these classes stupidly, 
				// but we should probably build in something to test for too-frequent reset
				Reset();
			}
		}
		assert(l_current < l); // Sanity check; the control logic says we now need to call Next at least once
	
		//~ std::cout << "calculating...\n";
	
		do
			Next();
		while(l_current < l);
	}
	
	return hl_current;
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
	assert(surfaceFraction < 1);
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
	
	hl_current = real_t(1);
}

////////////////////////////////////////////////////////////////////////

void h_Cap::Next() const
{
	Pl_computer.Next();
	Increment_l();
	
	assert(Pl_computer.l() == (l_current + 1));
	
	// We need 2l + 1 for the new l
	hl_current = (Pl_computer.P_lm2() - Pl_computer.P_l())/
		(twoLplus1 * twiceSurfaceFraction);
}
	
////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////
		
void h_Unstable::Reset() const
{
	//~ R_current = real_t(1);
	stable = true;
	
	if(Isotropic())
	{
		Set_l(1);
		hl_current = real_t(0);
		hl_last = real_t(1);
	}
	else
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
	if(Isotropic()) // All zeroes after h_0
	{
		Increment_l();
		assert(hl_current = real_t(0));
	}
	else
	{
		if(stable)
		{
			real_t const hl_next = h_lp1(); // iterate
			
			if(hl_next < 1e-5) // It starts to get bad around 1e-8, so use a buffer
				stable = false;
						
			// The less arbitrary instability detection does not work for on-the-fly recursion, 
			// because hl for the previous iteration is still inaccurate, 
			// but is not corrected.
			
			//~ real_t const R_next = hl_next / hl_current;
			
			//~ if((R_next < real_t(0)) or (R_next > R_current)) // detect instability
			//~ {
				
				
				//~ // Reset and re-iterate to a smaller l that was stable
				//~ size_t const l_stable = size_t(0.8*real_t(l_current));
				//~ assert(l_stable >= 2);
				
				//~ // Now we use a little recursive trickery.
				//~ // We do/set some things explicitly in case hl is using h_init
				//~ Reset();
				//~ hl_current = hl(l_stable);
				//~ stable = false;
				//~ // The do-while loop in hl() should ensure that Next() is called repeatedly, 
				//~ // now using the unstable Asym_Ratio code, until l increments to where we want.
			//~ }
			else
			{
				hl_last = hl_current;
				hl_current = hl_next;
				//~ R_current = R_next;
			}
		}
			
		// This needs to be done after stable recursion,
		// but before unstable recursion, because h_l = R(l) * h_l-1
		Increment_l();
		
		if(not stable) // this is not an else so that we can flow from stable to unstable
			hl_current *= Asym_Ratio();
	}
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::h_Gaussian(real_t const lambda_in):
	lambda(lambda_in),
	lambda2(kdp::Squared(lambda)) 
{
	assert(lambda > 0.);
	
	Setup();
	Reset();
}

////////////////////////////////////////////////////////////////////////

void h_Gaussian::Setup()
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
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::real_t h_Gaussian::Asym_Ratio() const
{
	real_t const term = twoLplus1 * lambda2;
	return real_t(2)/(term + std::sqrt(kdp::Squared(term) + real_t(4)));				
}

////////////////////////////////////////////////////////////////////////

bool h_Gaussian::Isotropic() const
{
	return (lambda > 1e3);
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::real_t h_Gaussian::h_lp1() const
{
	return -twoLplus1 * lambda2 * hl_current + hl_last;
}

////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////

h_Boost::h_Boost(vec3_t const& p3, real_t const mass):
	m2(kdp::Squared(mass)),
	p2(p3.Mag2()),
	beta(vec4_t::BetaFrom_Mass_pSquared(mass, p2))
{
	assert(beta >= real_t(0));
	assert(beta <= real_t(1));
	
	Setup();
	Reset();
}

////////////////////////////////////////////////////////////////////////

void h_Boost::Setup()
{
	if(Isotropic())
		hl_init = {beta, real_t(0)};
	else
	{
		real_t const term = std::sqrt((m2 + p2)*p2);		
		real_t const h2 = real_t(1) - (real_t(3) * m2)/(real_t(2) * p2) * 
			(real_t(1) - real_t(0.5) * (m2 / term) * 
			std::log1p(real_t(2) * (p2 + term) / m2));
		//~ R_current = hl_current / hl_last;
		
		assert(h2 >= real_t(0));
		assert(h2 <= real_t(1));
		
		hl_init = {beta, h2};
	}
}

////////////////////////////////////////////////////////////////////////

h_Boost::real_t h_Boost::Asym_Ratio() const
{
	return real_t(2 * (l_current + 2)) * beta / 
	(twoLplus1 + std::sqrt(kdp::Squared(twoLplus1) - 
		real_t(4) * kdp::Squared(beta) * real_t((l_current - 1) * (l_current + 2))));
}

////////////////////////////////////////////////////////////////////////

bool h_Boost::Isotropic() const
{
	// Zero-initiliazed ShapedJet
	return (p2 == 0.) or (beta < 1e-5);
}

////////////////////////////////////////////////////////////////////////

h_Boost::real_t h_Boost::h_lp1() const
{
	// This should be safe for beta = 1. for sufficiently small l, 
	// since the integers will be exactly represented
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
