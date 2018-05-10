#include "ShapeFunction.hpp"
#include <iostream>

std::vector<ShapeFunction::real_t> const& ShapeFunction::OnAxis(size_t const lMax) const
{
	if(onAxis.size() < lMax)
		this->Fill_OnAxis(lMax);

	assert(onAxis.size() >= lMax);
	
	return onAxis;
}

////////////////////////////////////////////////////////////////////////

h_Cap::h_Cap(real_t const surfaceFraction):
		twiceSurfaceFraction(real_t(2) * surfaceFraction)
{
	assert(surfaceFraction < 1);
}

void h_Cap::Fill_OnAxis(size_t const lMax) const
{
	Pl_computer.Setup(real_t(1) - twiceSurfaceFraction);
	onAxis.assign(lMax, 0.);
	
	// h_l     = (P_{l-1} - P_{l+1}) / (2 A_twr * (2l+1))
	// h_{l-1} = (P_{l-2} - P_{l})   / (2 A_twr * (2l-1))
	
	// The first l we want to set is l = 1, so we start at l = 2
	Pl_computer.Next();
	assert(Pl_computer.l() == 2);
			
	real_t twolm1 = real_t(2*Pl_computer.l() - 1); // start with l = 0
		
	// We set an l that is one less than the l we keep Pl_computer at
	// But we also don't want to set l=0, so we store it at lSet - 1
	for(size_t lSet = 1; lSet <= lMax; ++lSet)
	{
		onAxis[lSet - 1] = (Pl_computer.P_lm2() - Pl_computer.P_l())/
			(twolm1 * twiceSurfaceFraction);
			
		Pl_computer.Next();
		twolm1 += real_t(2);
	}
	assert(onAxis.size() == lMax);
}
	
////////////////////////////////////////////////////////////////////////
		
h_Unstable::h_Unstable():
	R_last(real_t(1)) {}
	
void h_Unstable::Reset(size_t const lMax) const
{
	onAxis.assign(lMax + 1, real_t(0)); // Fill with zeroes
	
	// Initialize
	onAxis[0] = real_t(1);
	R_last = real_t(1);
	
	Setup(lMax); // Set initial values, initialize l to the largest set value
	assert(l >= 1);
			
	lPlus1 = real_t(l + 1);
	twoLplus1 = real_t(2*l + 1); // 2*1 + 1
	
	stable = true;
}

void h_Unstable::Fill_OnAxis(size_t const lMax) const
{
	Reset(lMax);
	
	if(NotIsotropic()) // Otherwise, all zeroes after h_0
	{
		while(l < lMax) // we set l+1
		{
			if(stable)
			{
				h_lp1(); // iterate
				
				real_t const R = onAxis[l+1] / onAxis[l];
				
				if((R < real_t(0)) or (R > R_last))
				{
					// Move back a little bit
					l -= size_t(0.3*real_t(l));
					assert(l >= 2);
					lPlus1 = real_t(l + 1);
					twoLplus1 = real_t(2*l + 1);
				
					stable = false;
				}
				else
					R_last = R;
			}
			
			// This needs to be done after stable recursion, 
			// but before unstable recursion, because h_l = R(l) * h_l-1
			lPlus1 += real_t(1);
			twoLplus1 += real_t(2);
			++l;
			
			if(not stable)
			{
				onAxis[l] = Asym_Ratio() * onAxis[l-1];
				if(onAxis[l] == real_t(0))
					break;
			}
		}
	}
	
	// We need h0 = 1 to do the recursion without a conditional branch,
	// but now that we're done calculating, we don't want that h0 coefficient anymore
	onAxis.erase(onAxis.begin());
	assert(onAxis.size() == lMax);
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::h_Gaussian(real_t const lambda_in):
	h_Unstable(), 
	lambda(lambda_in),
	lambda2(kdp::Squared(lambda)) 
{
	assert(lambda > 0.);
}

void h_Gaussian::Setup(size_t const lMax) const
{
	// 1/tanh(1/lambda**2) = (1 + exp(-2/lambda**2))/(1 - exp(-2/lambda**2))
	// (1 + exp(-x))/(1-exp(-x)) = 1 + 2*exp(-x)/(1-exp(-x) = 1 + 2/(exp(x)-1)
	// Note that (1-x)*(1+x) is more accurate when x > 0.5, and 1-x**2 when x < 0.5
	// When lambda**2 > 0.5 * 1/tanh(2/lambda**2), we should do it the first way
	// This occurs when lambda2 is about 0.5

	if(lambda2 > real_t(0.5))
		onAxis[1] = real_t(1)/std::tanh(real_t(1)/lambda2) - lambda2;
	else
		onAxis[1] = kdp::Diff2(real_t(1), lambda) + real_t(2)/std::expm1(real_t(2)/lambda2);
		
	l = 1;
}

h_Gaussian::real_t h_Gaussian::Asym_Ratio() const
{
	real_t const term = twoLplus1 * lambda2;
	return real_t(2)/(term + std::sqrt(kdp::Squared(term) + real_t(4)));				
}

h_Gaussian::real_t h_Gaussian::NotIsotropic() const
{
	return (lambda < 1e3);
}

void h_Gaussian::h_lp1() const
{
	onAxis[l+1] = -twoLplus1*lambda2*onAxis[l] + onAxis[l-1];
}

////////////////////////////////////////////////////////////////////////

h_Boost::h_Boost(vec3_t const& p3, real_t const mass):
	m2(kdp::Squared(mass)),
	p2(p3.Mag2()),
	beta(vec4_t::BetaFrom_Mass_pSquared(mass, p2))
{
	assert(beta >= real_t(0));
	assert(beta <= real_t(1));
}

void h_Boost::Setup(size_t const lMax) const
{
	if(beta == real_t(1))
	{
		// fill lMax + 1, because we erase the first element
		onAxis.assign(lMax + 1, 1.);
		l = lMax + 1;
	}
	else
	{
		real_t const term = std::sqrt((m2 + p2)*p2);
		
		onAxis[1] = beta;
		
		real_t const h2 = real_t(1) - (real_t(3) * m2)/(real_t(2) * p2) * 
			(real_t(1) - real_t(0.5) * (m2 / term) * 
			std::log1p(real_t(2) * (p2 + term) / m2));
				
		assert(h2 >= real_t(0));
		assert(h2 < real_t(1));
		
		onAxis[2] = h2;
		
		l = 2;
	}
}

h_Boost::real_t h_Boost::Asym_Ratio() const
{
	return real_t(2) * (lPlus1 + real_t(1)) * beta / 
	(twoLplus1 + std::sqrt(kdp::Squared(twoLplus1) - 
		real_t(4) * kdp::Squared(beta) * (real_t(l) - real_t(1)) * (real_t(l) + real_t(2))));
}

h_Boost::real_t h_Boost::NotIsotropic() const
{
	return (beta > 1e-6);
}

void h_Boost::h_lp1() const
{
	// This should be safe for all beta = 1. for sufficiently small l, 
	// since the integers will be exactly represented
	onAxis[l+1] = (twoLplus1*onAxis[l] - beta*(lPlus1 + real_t(1))*onAxis[l-1])/
		(beta*(real_t(l) - real_t(1)));
}

////////////////////////////////////////////////////////////////////////

h_Boost_orig::h_Boost_orig(vec3_t const& p3, real_t const mass):
	beta(vec4_t::BetaFrom_Mass_pSquared(mass, p3.Mag2()))
{
	assert(beta >= real_t(0));
	assert(beta <= real_t(1));
}

void h_Boost_orig::Setup(size_t const lMax) const
{
	if(beta == real_t(1))
		onAxis[1] = real_t(1);
	else
	{
		real_t const h1 = real_t(1) + ((real_t(1) - beta)/beta)*
			(real_t(1) - (real_t(1) + beta)/beta * std::atanh(beta));
		
		assert(h1 >= real_t(0));
		assert(h1 < real_t(1));
		
		onAxis[1] = h1;
	}
	
	l=1;
}

h_Boost_orig::real_t h_Boost_orig::Asym_Ratio() const
{
	return real_t(2)*beta*lPlus1 / 
	(twoLplus1 + std::sqrt(kdp::Squared(twoLplus1) -
		kdp::Squared(real_t(2) * beta) * lPlus1*(lPlus1 - real_t(1))));
}

h_Boost_orig::real_t h_Boost_orig::NotIsotropic() const
{
	return (beta > 1e-6);
}

void h_Boost_orig::h_lp1() const
{
	onAxis[l+1] = (twoLplus1*onAxis[l] - beta*lPlus1*onAxis[l-1])/(beta*(lPlus1 - real_t(1)));
}
		
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
