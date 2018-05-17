#ifndef SHAPE_FUNCTION
#define SHAPE_FUNCTION

#include "PowerJets.hpp"
#include "RecursiveLegendre.hpp"
#include "kdp/kdpTools.hpp"

GCC_IGNORE(-Wpadded)

/*! @brief Calculate the on-axis coefficient for the particle shape function.
 * 
 *  The ShapeFunction is a state machine whose state (l) is mutable. 
 *  This property is useful because, while a user may be able to change the state, 
 *  they cannot alter the answer they will get. Hence, calling hl(23)
 *  is equivalent to looking up the value hl_Vec[22], just without 
 *  explicitly storing the vector.
 * 
 *  @warning ShapeFunction is not thread safe
*/ 
class ShapeFunction
{
	public:
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t;
		using vec4_t = PowerJets::vec4_t;
		
	protected:
		mutable real_t hl_current;
		mutable size_t l_current;
		mutable real_t twoLplus1;
		std::vector<real_t> hl_init; // The first few coefficients. Filled by derived ctor and possibly reused by Reset
		
		virtual void Reset() const = 0; // Called to reset recursion. MUST be called by derived ctor
		virtual void Next() const = 0; // Increment l and store h_l in hl_current
		void Increment_l() const;
		void Set_l(size_t const l) const;
		
		ShapeFunction();
				
	public:
		virtual ~ShapeFunction() {}
	
		//! @brief Return h_l for the given l. Advance l if necessary
		real_t hl(size_t const l) const;
	
		//! @brief Get the vector of on-axis coefficients for l=1 to l=lMax
		std::vector<real_t> hl_Vec(size_t const lMax) const;
		
		virtual ShapeFunction* Clone() const = 0;
};

//! @brief A particle uniformly distributed across a 
// circular cap of solid angle Omega = surfaceFraction * 4 Pi
class h_Cap : public ShapeFunction
{
	protected:
		mutable RecursiveLegendre<real_t> Pl_computer;
		real_t twiceSurfaceFraction;
	
		void Reset() const;
		void Next() const;
			
	public:
		h_Cap(real_t const surfaceFraction);
		
		ShapeFunction* Clone() const
		{
			return static_cast<ShapeFunction*>(new h_Cap(*this));
		}		
};

/*! A shape function with an unstable recursion that can be 
 *  corrected by extrapolating its ratio. The extrapolation begins when 
 *  hl drops below the threshold.
*/
class h_Unstable : public ShapeFunction
{
	protected:
		mutable real_t hl_last;
		//~ real_t R_current;
		mutable bool stable;
		
		virtual void Reset() const;		
		virtual void Next() const;
		virtual void Setup() = 0;
		
		//! @brief The ratio of h_l / h_(l-1), to be used once h_l < threshold 
		virtual real_t Asym_Ratio() const = 0;
						
		//! @brief Return the l + 1 coefficient
		virtual real_t h_lp1() const = 0;
		
		virtual bool Isotropic() const = 0;
		
		//h_Unstable();
		
	public:
		virtual ~h_Unstable() {}
};

//! @brief A particle distributed by a pseudo-Gaussian.
class h_Gaussian : public h_Unstable
{
	private:
		real_t lambda;
		real_t lambda2;
	
	protected:
		void Setup();
		
		real_t Asym_Ratio() const;
		real_t h_lp1() const;
		bool Isotropic() const;
		
	public:
		h_Gaussian(real_t const lambda_in);
		
		ShapeFunction* Clone() const
		{
			return static_cast<ShapeFunction*>(new h_Gaussian(*this));
		}
};

//! @brief A scalar decay boosted into the lab frame.
class h_Boost : public h_Unstable
{
	private:
		real_t m2;
		real_t p2;
		real_t beta;
	
	protected:
		void Setup();
		
		real_t Asym_Ratio() const;
		real_t h_lp1() const;
		bool Isotropic() const;
		
	public:
		h_Boost(vec3_t const& p3, real_t const mass);
		
		ShapeFunction* Clone() const
		{
			return static_cast<ShapeFunction*>(new h_Boost(*this));
		}
};

GCC_IGNORE_END


//! @brief A scalar decay boosted into the lab frame, 
//  but forgetting to account for the energy transformation
//~ class h_Boost_orig : public h_Unstable
//~ {
	//~ private:
		//~ real_t beta;
	
	//~ protected:
		//~ void Setup(size_t const lMax) const;		
		//~ real_t Asym_Ratio() const;	
		//~ real_t NotIsotropic() const;		
		//~ void h_lp1() const;
		
	//~ public:
		//~ h_Boost_orig(vec3_t const& p3, real_t const mass);
//~ };


		
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

#endif
