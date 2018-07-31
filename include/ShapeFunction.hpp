#ifndef SHAPE_FUNCTION
#define SHAPE_FUNCTION

#include "PowerJets.hpp"
#include "RecursiveLegendre.hpp"
#include "kdp/kdpTools.hpp"

GCC_IGNORE_PUSH(-Wpadded)

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
		//~ ShapeFunction() {}
		
	public:
		virtual ~ShapeFunction() = default;
	
		//! @brief Return h_l for the given l.
		virtual real_t hl(size_t const l) const = 0;
	
		/*! @brief Get the vector of on-axis coefficients for l=1 to l=lMax
		 * 
		 *  The whole point of this class is that repeating the recursion is
		 *  better than trying to cache it --- generate a fresh vector each time. 
		*/
		std::vector<real_t> hl_Vec(size_t const lMax) const;
		
		virtual ShapeFunction* Clone() const = 0;
};

//! @brief A delta-distribution, whose "up" coefficient is always 1.
class h_Delta : public ShapeFunction
{
	public:
		h_Delta() {}			
																						GCC_IGNORE_PUSH(-Wunused-parameter)
		real_t hl(size_t const l) const final {return real_t(1);}
																						GCC_IGNORE_POP		
		ShapeFunction* Clone() const {return new h_Delta();}
};

/*! @brief Return the "up" coefficient measured in an experiment.
 * 
 *  This is useful when an empirical shape function is used (e.g. pileup).
 * 
 *  \throws Throws std::runtime_error when l exceeds lMax
*/ 
class h_Measured : public ShapeFunction
{
	private:
		std::vector<real_t> hl_vec;
		
		static std::vector<real_t> ReadFirstColumn(std::string const& filePath);
	
	public:
		h_Measured(std::vector<real_t> const hl_in = std::vector<real_t>()):
			hl_vec(hl_in) {}
			
		//! @brief Read hl from the first column of a file
		h_Measured(std::string const& filePath):
			h_Measured(ReadFirstColumn(filePath)) {}
			
		real_t hl(size_t l) const final;
		size_t lMax() const {return hl_vec.size();}
		
		ShapeFunction* Clone() const final {return new h_Measured(hl_vec);}
};

/*! @brief Calculate the on-axis coefficient for the particle shape function.
 * 
 *  The ShapeFunction_Recursive is a state machine whose state (l) is mutable. 
 *  This property is useful because, while a user may be able to change the state, 
 *  they cannot alter the answer they will get. Hence, calling hl(23)
 *  is equivalent to looking up the value hl_Vec[22], just without 
 *  explicitly storing the vector.
 * 
 *  @warning ShapeFunction is not thread safe
*/ 
class ShapeFunction_Recursive : public ShapeFunction
{
	public:
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t;
		using vec4_t = PowerJets::vec4_t;
		
	protected:
		// We can intercept trivial shapes before doing lots of work
		enum class Shape {NonTrivial=0, Isotropic, Delta};
		
		mutable size_t l_current;
		Shape shape;
		mutable real_t hl_current;
		mutable real_t twoLplus1;
		std::vector<real_t> hl_init; // The first few coefficients. Filled by derived ctor and possibly reused by Reset
		
		bool IsTrivial() const {return bool(shape);}
		real_t hl_trivial(size_t const l) const;
		
		virtual void Reset() const = 0; // Called to reset recursion. MUST be called by derived ctor
		virtual void Next() const = 0; // Increment l and store h_l in hl_current
		void Increment_l() const;
		void Set_l(size_t const l) const;
		
		ShapeFunction_Recursive();
				
	public:
		virtual ~ShapeFunction_Recursive() = default;
	
		//! @brief Return h_l for the given l. Advance l if necessary
		real_t hl(size_t const l) const final;
};

//! @brief A particle uniformly distributed across a 
//! circular cap of solid angle Omega = surfaceFraction * 4 Pi
class h_Cap : public ShapeFunction_Recursive
{
	protected:
		mutable RecursiveLegendre<real_t> Pl_computer;
		real_t twiceSurfaceFraction;
	
		void Reset() const;
		void Next() const;
			
	public:
		h_Cap(real_t const surfaceFraction = real_t(0)); // Default to delta-distribution
		
		ShapeFunction* Clone() const
		{
			return static_cast<ShapeFunction*>(new h_Cap(*this));
		}
};

/*! A shape function with an unstable recursion that can be 
 *  corrected by extrapolating its ratio. The extrapolation begins when 
 *  hl drops below the threshold.
*/
class h_Unstable : public ShapeFunction_Recursive
{
	protected:
		mutable real_t hl_last;
		mutable bool stable;
		
		// When hl get small, the recursion becomes unstable.
		// This seems to occur some time before 2^-(half the mantissa bits)
		// for different shapes and different parameters
		static constexpr real_t gettingSmall = 
			std::exp2(real_t(-0.33 * std::numeric_limits<real_t>::digits));
		
		virtual void Reset() const; //!< @brief reset the recursion
		virtual void Next() const; //!< @brief advance the recursion
		virtual void Setup() = 0; //!< @brief setup the recursion
		
		//! @brief Detect when the recursion becomes unstable
		virtual bool IsUnstable(real_t const hl_next) const = 0; 
		
		//! @brief The ratio of h_l / h_(l-1), to be used once hl becomes unstable
		virtual real_t Asym_Ratio() const = 0;
						
		//! @brief Return the l + 1 coefficient
		virtual real_t h_lp1() const = 0;
		
	public:
		virtual ~h_Unstable() = default;
};

//! @brief A particle distributed by a pseudo-Gaussian.
class h_Gaussian : public h_Unstable
{
	private:
		real_t lambda; // Keep around for calls to Lambda()
		real_t lambda2;
	
	protected:
		void Setup();
		bool IsUnstable(real_t const hl_next) const;
		
		real_t Asym_Ratio() const;
		real_t h_lp1() const;		
		
	public:
		//! @brief Construct a pseudo-normal with a given lambda (default to delta-distribution)
		h_Gaussian(real_t const lambda_in = real_t(0));
		
		//! @brief Construct a pseudo-normal where a circular cap of radius R
		//! contains a fraction u of the particle
		h_Gaussian(real_t const R, real_t const u);
		
		ShapeFunction* Clone() const
		{
			return static_cast<ShapeFunction*>(new h_Gaussian(*this));
		}
		
		real_t Lambda() const {return lambda;}
		
		/*! @brief Solve for the lambda where a fraction u of the distribution
		 *  occupies a circular cap of given radius.
		*/ 
		static real_t SolveLambda(real_t const R, real_t const u);
};

//! @brief A scalar decay boosted into the lab frame.
class h_Boost : public h_Unstable
{
	private:
		real_t m2;
		real_t p2;
		real_t beta;
		// The asymptotic ratio of adjacent values stabilizes to beta / (1 + sqrt(1-beta**2))
		// We can detect instability by detecting h_next / h_current dropping below this ratio
		real_t R_asym;
	
	protected:
		void Setup();
		bool IsUnstable(real_t const hl_next) const;
		
		real_t Asym_Ratio() const;
		real_t h_lp1() const;
		
	public:
		h_Boost(); // Default to delta-distribution
		h_Boost(vec3_t const& p3, real_t const mass);
		
		ShapeFunction* Clone() const
		{
			return static_cast<ShapeFunction*>(new h_Boost(*this));
		}
};

GCC_IGNORE_POP

#endif
