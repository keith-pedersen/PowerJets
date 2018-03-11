#include "ShapeFunction.hpp"


std::vector<ShapeFunction::real_t> const& ShapeFunction::OnAxis(size_t const lMax)
{
	if(onAxis.size() not_eq (lMax + 1))
		this->Fill_OnAxis(lMax);
	
	return onAxis;
}

////////////////////////////////////////////////////////////////////////

h_Cap::h_Cap(real_t const surfaceFraction):
		twiceSurfaceFraction(real_t(2) * surfaceFraction)
{
	assert(surfaceFraction < 1);
}

void h_Cap::Fill_OnAxis(size_t const lMax)
{
	Pl_computer.z.front() = real_t(1) - twiceSurfaceFraction;
	Pl_computer.Reset();
	
	onAxis.assign(lMax + 1, 0.);
	onAxis.front() = real_t(1);
	
	real_t twolp1 = real_t(1); // start with l = 0
	
	// Move Pl_computer to l == 1
	Pl_computer.Next();
	assert(Pl_computer.l() == 1);
	
	for(size_t l = 1; l <= lMax; ++l)
	{
		real_t P_lm1 = Pl_computer.P_lm1().front(); // Make sure this get's called first
		onAxis[l] = (P_lm1 - Pl_computer.Next().front())/
			((twolp1 += real_t(2)) * twiceSurfaceFraction);
	}
}
	
////////////////////////////////////////////////////////////////////////
		
h_Unstable::h_Unstable(real_t const threshold_in):
	threshold(threshold_in) {}
	
void h_Unstable::Reset(size_t const lMax)
{
	onAxis.assign(lMax + 1, real_t(0)); // Fill with zeroes
	
	// Initialize
	onAxis.front() = real_t(1);
	onAxis[1] = h_1();
		
	// l = 1, emplace l + 1 every iteration
	l = 1;
	lPlus1 = real_t(2);
	twoLplus1 = real_t(3); // 2*1 + 1
	
	stable = true;
}

void h_Unstable::Fill_OnAxis(size_t const lMax)
{
	Reset(lMax);
	
	if(NotIsotropic()) // Otherwise, all zeroes after h_0
	{
		while(l < lMax)
		{
			if(stable)
			{
				h_lp1();
				
				if(std::fabs(onAxis[l+1]) < threshold)
					stable = false; // Switch to stable within this iteration
			}
			
			// This needs to be done after stable recusion, 
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
}

////////////////////////////////////////////////////////////////////////

h_Gaussian::h_Gaussian(real_t const lambda_in, real_t threshold_in):
	h_Unstable(threshold_in), 
	lambda(lambda_in),
	lambda2(kdp::Squared(lambda)) 
{
	assert(lambda > 0.);
}

h_Gaussian::real_t h_Gaussian::h_1() const
{
	// 1/tanh(1/lambda**2) = (1 + exp(-2/lambda**2))/(1 - exp(-2/lambda**2))
	// (1 + exp(-x))/(1-exp(-x)) = 1 + 2*exp(-x)/(1-exp(-x) = 1 + 2/(exp(x)-1)
	// Note that (1-x)*(1+x) is more accurate when x > 0.5, and 1-x**2 when x < 0.5
	// When lambda**2 > 0.5 * 1/tanh(2/lambda**2), we should do it the first way
	// This occurs when lambda2 is about 0.5

	if(lambda2 > real_t(0.5))
		return real_t(1)/std::tanh(real_t(1)/lambda2) - lambda2;
	else
		return kdp::Diff2(real_t(1), lambda) + real_t(2)/std::expm1(real_t(2)/lambda2);
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

void h_Gaussian::h_lp1()
{
	onAxis[l+1] = -twoLplus1*lambda2*onAxis[l] + onAxis[l-1];
}

////////////////////////////////////////////////////////////////////////

h_Boost::h_Boost(real_t const beta_in, real_t threshold_in):
	h_Unstable(threshold_in), 
	beta(beta_in)
{
	assert(beta >= real_t(0));
	assert(beta < real_t(1));
}

h_Boost::real_t h_Boost::h_1() const
{
	real_t const val = real_t(1) + ((real_t(1) - beta)/beta)*
		(real_t(1) - (real_t(1) + beta)/beta * std::atanh(beta));
	
	assert(val >= real_t(0));
	assert(val < real_t(1));
	
	return val;
}

h_Boost::real_t h_Boost::Asym_Ratio() const
{
	return real_t(2)*beta*lPlus1 / 
	(twoLplus1 + std::sqrt(kdp::Squared(twoLplus1) -
		kdp::Squared(real_t(2) * beta) * lPlus1*(lPlus1 - real_t(1))));
}

h_Boost::real_t h_Boost::NotIsotropic() const
{
	return (beta > 1e-6);
}

void h_Boost::h_lp1()
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
