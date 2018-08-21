#ifndef SHAPE_FUNCTION
#define SHAPE_FUNCTION

// Copyright (C) 2018 by Keith Pedersen (Keith.David.Pedersen@gmail.com)
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
// The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#include "PowerJets.hpp"
#include "RecursiveLegendre.hpp"
#include "kdp/kdpTools.hpp"
#include <memory> // shared_ptr

GCC_IGNORE_PUSH(-Wpadded)

/*! @file ShapeFunction.hpp
 *  @brief Supplies a number of shape functions used to by PowerSpectrum
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
*/ 

/*! @brief Calculate the "up" coefficient for the particle shape function.
 * 
 *  The ShapeFunction concept is explained in more detail in 
 *  "Harnessing the global correlations of the QCD power spectrum."
 * 
 *  A ShapeFunction \f$ h(\hat{r}) \f$ is a particle's spatial probability distribution.
 *  An ensemble's PowerSpectrum is only a complete decomposition when
 *  the ensemble is built from \em extensive particles (they do not occupy only 
 *  one spatial location, but are distributed in space about their direction of travel \f$ \hat{p} \f$).
 *  The ShapeFunction records the fraction of the particle that occupies 
 *  some portion of the unit sphere (a differential solid angle \f$ {\rm d}\Omega \f$):
 *  	\f[ {\rm d}P = h(\hat{r}) {\rm d}\Omega \f]
 *  Hence, a ShapeFunction is normalized when integrated over the whole sphere.
 * 
 *  The ShapeFunction's defined thus far are always azimuthally symmetric 
 *  about the particle's direction of travel \f$ \hat{p} \f$.
 *  This simplifies the PowerSpectrum calculation so that it only needs 
 *  each ShapeFunction's "up" coefficient \f$ \bar{h}_\ell \f$ (ShapeFunction::hl()).
 *  The "up" coefficient is defined by rotating the shape function
 *  such that \f$ \hat{p} \parallel \hat{z} \f$, 
 *  then calculating the Legendre integral
 *  	\f[ \bar{h}_\ell = \int_{-1}^1 {\rm d}z\, P_l(z)\,h(z) \f]
 *  Thus, the direction of the shape function is not important, 
 *  only its topology and the size of its spatial extent.
 * 
 *  @warning ShapeFunction(s) are NOT thread safe
*/ 
class ShapeFunction
{
	public:
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t;
		using vec4_t = PowerJets::vec4_t;
			
	public:
		virtual ~ShapeFunction() = default;
	
		//! @brief Return the "up" coefficient h_l for the given l.
		virtual real_t hl(size_t const l) const = 0;
	
		//! @brief Get the vector of "up" coefficients for l=1 to l=lMax
		std::vector<real_t> hl_Vec(size_t const lMax) const;
		
		//! @brief Clone the shape function into a shared_ptr		 
		virtual std::shared_ptr<ShapeFunction> Clone() const = 0;
		
		//! @brief Make a shared_ptr in-place using the forwarded arguments
		template<class T, typename... Args>
		static std::shared_ptr<ShapeFunction> Make(Args&&... args)
		{
			// Automatically convert to a base-class pointer in the return
			return std::make_shared<T>(std::forward<Args>(args)...);
		}
};

////////////////////////////////////////////////////////////////////////

/*! @brief A curiously recurring template pattern (CRTP) used to define Clone().
 *  
 *  We must also include the Intermediate class so that Intermediate is a Direct base.
 *  This allows us to extend ShapeFunction with as many levels of 
 *  abstract classes as necessary, then add the Clone definition in the last step.
 * 
 *  class h_PseudoNormal : public ShapeFunction_Cloneable<h_Unstable, h_PseudoNormal>
 *  { 
 *   ... 
 *  };
 * 
 *  Ultimately, this class was not used because each instantiation
 *  made the binary 2-10 kB larger. This seems pretty inefficient, 
 *  considering that the Clone() function is a single line.
*/ 
template <class Intermediate, class Derived>
class ShapeFunction_Cloneable : public Intermediate
{
	public:
		virtual std::shared_ptr<ShapeFunction> Clone() const final
		{
			return ShapeFunction::Make<Derived>(static_cast<Derived const &>(*this));
		}
};

////////////////////////////////////////////////////////////////////////

//! @brief A delta-distribution, whose "up" coefficient is always 1.
class h_Delta : public ShapeFunction
{
	public:
		h_Delta() {}
																						GCC_IGNORE_PUSH(-Wunused-parameter)
		virtual real_t hl(size_t const l) const final {return real_t(1);}
																						GCC_IGNORE_POP		
		virtual std::shared_ptr<ShapeFunction> Clone() const final;
};

////////////////////////////////////////////////////////////////////////

/*! @brief Return the "up" coefficient measured in an experiment.
 * 
 *  This is useful when an empirical shape function is used (e.g. pileup).
 *  
 *  \warning We \em assume that l=0 is not written in the file or supplied vector
 *  \throws hl() throws std::runtime_error when l exceeds lMax
*/ 
class h_Measured : public ShapeFunction
{
	private:
		std::vector<real_t> hl_vec;
		
		/*! @brief Convert the first column of a file to floating point, 
		 *  and return as a vector.
		 * 
		 *  Skips header lines commented with '#'. Used std::stod under the hood.
		 * 
		 *  \warning Stops parsing on the first empty line.
		 *  \throws Throws std::ios_base if file cannot be read.
		 *  \throws Throws std::invalid_argument if a first-column value cannot be parsed
		 */		
		static std::vector<real_t> ReadFirstColumn(std::string const& filePath);
	
	public:
		h_Measured(std::vector<real_t> const hl_in = std::vector<real_t>()):
			hl_vec(hl_in) {}
			
		//! @brief Read hl from the first column of a file
		h_Measured(std::string const& filePath):
			h_Measured(ReadFirstColumn(filePath)) {}
			
		virtual real_t hl(size_t l) const final;
		size_t lMax() const {return hl_vec.size();}
		
		virtual std::shared_ptr<ShapeFunction> Clone() const final;
};

////////////////////////////////////////////////////////////////////////

/*! @brief Recursively calculate the "up" coefficient for the particle shape function.
 * 
 *  The ShapeFunction_Recursive is a state machine whose state (l_current) is mutable. 
 *  This property is useful because, while a user may be able to change the state, 
 *  they cannot alter the answer they will get. Hence, calling hl(23)
 *  is equivalent to looking up the value hl_Vec[22], just without 
 *  explicitly storing the vector.
 * 
 *  @warning ShapeFunction_Recursive is not thread safe
*/ 
class ShapeFunction_Recursive : public ShapeFunction
{
	public:
		using real_t = PowerJets::real_t;
		using vec3_t = PowerJets::vec3_t;
		using vec4_t = PowerJets::vec4_t;
		
	protected:
		/*! @brief Shape is used to identify trivial shapes before doing lots of calculation.
		 * 
		 *  Isotropic distribution: h_0 = 1, h_(l>1) = 0
		 *  Delta distribution: h_l = 1
		 *  
		 *  Setting NonTrivial=0 permits a convenient definition of IsTrivial
		*/
		enum class Shape {NonTrivial=0, Isotropic, Delta};
		
		mutable size_t l_current;
		mutable real_t hl_current;
		mutable real_t twoLplus1;
		//! @brief The first few coefficients. Filled by derived ctor and possibly reused by Reset().
		std::vector<real_t> hl_init;
		Shape shape; //!< @brief A flag to signal a trivial shape that doesn't require calculation
		
		bool IsTrivial() const {return bool(shape);}
		//! @brief Return the hl for a trivial shape (Isotropic or Delta)
		real_t hl_trivial(size_t const l) const;
		
		//! @brief Reset the recursion. MUST be called by derived ctor.
		virtual void Reset() const = 0;
		//! @brief Increment the recursion, storing h_l in hl_current
		virtual void Next() const = 0;
		void Increment_l() const; //!< @brief Increment l_current and twoLplus1
		void Set_l(size_t const l) const; //!< @brief Set l_current and twoLplus1
		
		ShapeFunction_Recursive();
				
	public:
		virtual ~ShapeFunction_Recursive() = default;
	
		virtual real_t hl(size_t const l) const final;
};

////////////////////////////////////////////////////////////////////////

/*! @brief A particle uniformly distributed across a 
 *  circular cap of solid angle \f$ \Omega = {\tt surfaceFraction}\,4\pi \f$
*/ 
class h_Cap : public ShapeFunction_Recursive
{
	protected:
		mutable RecursiveLegendre<real_t> Pl_computer;
		real_t twiceSurfaceFraction; // Used directly by recursive definition
	
		virtual void Reset() const final;
		virtual void Next() const final;
			
	public:
		//! @brief Default to a delta-distribution
		h_Cap(real_t const surfaceFraction = real_t(0));
		
		real_t SurfaceFraction() const {return real_t(0.5)*twiceSurfaceFraction;}
		
		virtual std::shared_ptr<ShapeFunction> Clone() const final;
};

////////////////////////////////////////////////////////////////////////

/*! @brief A shape function with an unstable recursion that can be 
 *  corrected by extrapolating the ratio R_l of adjacent h_l for large l.
 * 
 *  Extrapolation begins when IsUnstable() returns true.
*/
class h_Unstable : public ShapeFunction_Recursive
{
	protected:
		mutable real_t hl_last;
		mutable bool stable;
		
		/*! @brief When hl get small, the recursion becomes unstable.
		 *  
		 *  This seems to occur some time before 2^(-(half the mantissa bits))
		 *  for different shapes and different parameters
		*/
		static constexpr real_t gettingSmall = 
			std::exp2(real_t(-0.33 * std::numeric_limits<real_t>::digits));
		
		virtual void Reset() const final;
		virtual void Next() const final;
		virtual void Setup() = 0; //!< @brief Setup the initial recursion
		
		//! @brief Detect when the recursion becomes unstable
		virtual bool IsUnstable(real_t const hl_next) const = 0; 
		
		/*! @brief The ratio of h_l / h_(l-1), generally accurate for "large" l
		 * 
		 *  Once the main hl recursion becomes unstable, 
		 *  we can approximate h_l = R_l * h_(l-1)
		*/ 
		virtual real_t R_l() const = 0;
						
		//! @brief Return the l + 1 coefficient
		virtual real_t h_lp1() const = 0;
		
	public:
		virtual ~h_Unstable() = default;
};

////////////////////////////////////////////////////////////////////////

/*! @brief A particle spatially distributed by an azimuthally symmetric distribution
 *  whose radial distribution is pseudo-normal
*/ 
class h_PseudoNormal : public h_Unstable
{
	private:
		real_t lambda; // Keep around for calls to Lambda()
		real_t lambda2;
	
	protected:
		virtual void Setup() final;
		virtual bool IsUnstable(real_t const hl_next) const final;
		
		virtual real_t R_l() const final;
		virtual real_t h_lp1() const final;
		
	public:
		//! @brief Construct a pseudo-normal with a given lambda (default to delta-distribution)
		h_PseudoNormal(real_t const lambda_in = real_t(0));
		
		//! @brief Construct a pseudo-normal where a circular cap of radius R
		//! contains a fraction u of the particle
		h_PseudoNormal(real_t const R, real_t const u);
		
		virtual std::shared_ptr<ShapeFunction> Clone() const final;	
		
		real_t Lambda() const {return lambda;}
		
		/*! @brief Solve for the lambda where a fraction u of the distribution
		 *  occupies a circular cap of angular radius R.
		 * 
		 *  \throws Throws std::domain_error if (R < 0 or R > Pi) or (u <= 0 or u > 1)
		 *  (u must be non-zero because \em some fraction must be specified).
		*/ 
		static real_t SolveLambda(real_t const R, real_t const u);
};

////////////////////////////////////////////////////////////////////////

//! @brief A scalar decay boosted into the lab frame.
class h_Boost : public h_Unstable
{
	private:
		real_t m2;
		real_t p2;
		real_t beta;
		// The asymptotic ratio of adjacent values stabilizes to: beta / (1 + sqrt(1-beta**2))
		// We can detect instability by detecting h_next / h_current dropping below this ratio
		real_t R_asym;
	
	protected:
		virtual void Setup() final;
		virtual bool IsUnstable(real_t const hl_next) const final;
		
		virtual real_t R_l() const final;
		virtual real_t h_lp1() const final;
		
	public:
		h_Boost(); //! @brief Default to delta-distribution
		h_Boost(vec3_t const& p3, real_t const mass);
		
		virtual std::shared_ptr<ShapeFunction> Clone() const final;
};

GCC_IGNORE_POP

#endif
