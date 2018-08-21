#ifndef RECURSIVE_LEGENDRE
#define RECURSIVE_LEGENDRE

//~ #include <type_traits>
#include <cmath>
#include <cstddef>
#include <assert.h>

/*! @file RecurisveLegendre.hpp
 *  @brief Defines two classes to recursively calculate Legendre polynomials at given z-values
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com, https://wwww.hepguy.com)
*/ 

/*! @brief Recursively calculate the Legendre polynomial P_l for an array/vector of z values.
 * 
 *  \f$ (l+1)P_{l+1}(z) = (2l+1)z\,P_l(z) - l\,P_{l-1}(z) \f$
 * 
 *  Operating recursively on an array of values allows efficient auto-vectorization
 *  to SIMD instructions on the CPU, with high cache-hit efficiency. 
 *  
 *  \tparam T	
 *  The container_t (a std::vector or a std::array)
*/
template<class T>
class RecursiveLegendre_Increment
{
	public:
		typedef T container_t;
		typedef typename container_t::value_type real_t;
		
		static_assert(std::is_floating_point<real_t>::value, 
			"RecursiveLegendre_Increment: container_t::value_type must be a floating point type");
				
	private:
		// Named aliases for the three P_l arrays (last iteration, this iteration, next iteration).
		// Use __restrict to tell the compiler that they'll never collide (for optimization).
		// (__restrict works for GCC, Clang, VCC and Intel, while __restric__ is just GCC and Clang).
		container_t* __restrict Pl_last;
		container_t* __restrict Pl_this;
		container_t* __restrict Pl_next;
		
		// The "coeff" is the numerical pre-factor in the P_l recursion (e.g. (2l + 1))
		real_t Pl_next_coeff, Pl_this_coeff, Pl_last_coeff;
		
		// As we process the P_l recursion in the l-loop,
		// we need to swap Pl_this -> Pl_last, Pl_next -> Pl_this.
		// But std::array::swap is VERY slow (versus std::vector), 
		// b/c its an element-wise swap (versus swapping the std::vector.data() pointer). 
		// The solution: create 3 x Pl worker arrays which we alias with pointers, 
		// then swap the pointers (essentially mimic std::vector swap,
		// just using a fixed width array(for better compiler optimization, 
		// since we know exactly how long each increment will be).
		container_t Pl_A, Pl_B, Pl_C; // Pl_last, Pl_this and Pl_next will be stored here.
			
		size_t l_this; //!< @brief The Legendre index l for Pl_this
		size_t size; //!< @brief The active size (in case there's lot's of unused space in z)
		
		void Fill(container_t& vec, real_t const val)
		{
			if(vec.size() not_eq z.size())
			{
				// This resizes std::vector, but will still compile for std::array
				// (i.e., we can't use vec.resize() on std::vector)
				// Slightly wasteful, but initialization doesn't happen too often
				vec = z;
				assert(vec.size() == z.size());
			}
			for(size_t i = 0; i < vec.size(); ++i)
				vec[i] = val;
		}
		
	public:
		/*! @brief The z-values for which P_l is calculated
		 *  
		 *  The public can access z so they are not forced to Setup()
		 *  by passing z as an argument (e.g. if some function() takes z by reference to fill it).
		 *  
		 *  \warning The public must then call Reset() before calling Next(), 
		 *  otherwise there will be undefined behavior.
		*/ 
		container_t z;
		
		//! @brief Initialize the object in a valid (though mostly useless) state.
		RecursiveLegendre_Increment()
		{
			Fill(z, real_t(0));
			Reset(z.size());
		}
		
		~RecursiveLegendre_Increment(){} // Suppress inlining failure warnings
		
		size_t l() const {return l_this;} //!< @brief The current Legendre index l.
		
		size_t ActiveSize() const {return size;} //!< @brief The active size of all containers
		
		// The next time we call Next(), it will overwrite Pl_next,
		// but until then it provides access to the second-to-last value
		inline container_t const& P_lm2() const {return *Pl_next;} //!< @brief \f$ P_{l-2}(z) \f$
		inline container_t const& P_lm1() const {return *Pl_last;} //!< @brief \f$ P_{l-1}(z) \f$
		inline container_t const& P_l() const {return *Pl_this;} //!< @brief \f$ P_l(z) \f$
		
		/*! @brief Reset \ref z to \p z_new and \p l to 1.
		 * 
		 *  \param activeSize
		 *  The active size of \p z_new (e.g. the indices which will actually be used).
		 *  No index larger than activeSize will be altered during Next(), 
		 *  and accessing \ref z, P_l(), P_lm1(), or P_lm2() outside of this range is undefined behavior.
		 *  \throws Throws out_for_range if size > z_new.size()
		*/ 
		void Setup(container_t const& z_new, size_t const activeSize)
		{
			z = z_new;
			Reset(activeSize);
		}
		
		//! @brief Reset \ref z to \p z_new and \p l to 1.
		void Setup(container_t const& z_new)
		{
			Setup(z_new, z_new.size());
		}
				
		/*! @brief Reset \ref l_this to 1, but use the existing \ref z.
		 * 
		 *  \param activeSize
		 *  The active size of \p z_new (e.g. the indices which will actually be used).
		 *  No index larger than activeSize will be altered during Next(), 
		 *  and accessing \ref z, P_l(), P_lm1(), or P_lm2() outside of this range is undefined behavior.
		 *  \throws Throws out_for_range if size > z_new.size()
		*/
		void Reset(size_t const activeSize)
		{			
			Pl_last = &Pl_A;
			Pl_this = &Pl_B; 
			Pl_next = &Pl_C;
						
			// Start with l = 1
			l_this = 1;
			
			if(activeSize > z.size())
				throw std::out_of_range("RecursiveLegendre_Increment: activeSize > z.size()");
			size = activeSize;
			
			Pl_last_coeff = -real_t(l_this); 		// (-l), must put negative outside of real_t, because size_t
			Pl_this_coeff = real_t(2*l_this + 1); 	// (2l + 1)
			Pl_next_coeff = real_t(l_this + 1); 		// (l + 1)
			
			*Pl_this = z; // P_1(x) = z
			
			// Initialize the last values P_0(z) = 1
			Fill(*Pl_last, real_t(1));
			// There is no second-to-last value, fill Pl_next with NAN
			Fill(*Pl_next, real_t(NAN));
		}
		
		//! @brief Reset \p l to 1, but use the existing \ref z.
		void Reset()
		{
			Reset(z.size());
		}
		
		//! @brief Increment \p l and return the \p P_l array.
		container_t const& Next()
		{
			// NOTE: gcc won't auto-vectorize if the iterator has a 
			// smaller data type than the data type of the loop bound, 
			// because it can't ensure that the loop will terminate 
			// (from iterator overflow). Must use size_t.
			for(size_t i = 0; i < size; ++i)
			{
				// To auto-vectorize a loop, don't try to mess with registers 
				(*Pl_next)[i] = 
					(Pl_this_coeff * z[i] * (*Pl_this)[i] 
					+ Pl_last_coeff * (*Pl_last)[i])/Pl_next_coeff;
			}
			
			// We use the P_l+1 recursion, so we post-increment l
			++l_this;
			Pl_next_coeff += real_t(1); // (l + 1)
			Pl_this_coeff += real_t(2); // (2l + 1)
			Pl_last_coeff -= real_t(1); // (-l)
			
			{
				// last = this
				// this = next
				// next = last (now second-to-last value)				
				
				container_t* __restrict const Pl_secondLast = Pl_last;
				Pl_last = Pl_this; // Pl_this -> Pl_last
				Pl_this = Pl_next; // Pl_next -> Pl_this
				Pl_next = Pl_secondLast; // Pl_last -> Pl_next
			}
			
			return P_l();
		}
};

/*! @brief Recursively calculate the Legendre polynomial P_l for a single z-value.
 * 
 *  \f$ (l+1)P_{l+1}(z) = (2l+1)z\,P_l(z) - l\,P_{l-1}(z) \f$
 * 
 *  
 *  \tparam real_t
 *  The floating-point type
*/
template<typename real_t>
class RecursiveLegendre
{
	private:
		static_assert(std::is_floating_point<real_t>::value, 
			"RecursiveLegendre_Increment: container_t::value_type must be a floating point type");
			
		real_t Pl_last, Pl_this, Pl_next;
	
		// The "coeff" is the numerical pre-factor in the P_l recursion (e.g. (2l + 1))
		real_t Pl_next_coeff, Pl_this_coeff, Pl_last_coeff;
		
		size_t l_this; //!< @brief The Legendre index l for Pl_this
	
	public:
		real_t z; //!< @brief The z-value
		
		//! @brief Initialize the object in a valid state.
		RecursiveLegendre()
		{
			z = real_t(0);
			Reset();
		}
		
		size_t l() {return l_this;} //!< @brief The current Legendre index l.
		
		// The next time we call Next(), it will overwrite Pl_next,
		// but until then it provides access to the second-to-last value
		real_t const& P_lm2(){return Pl_next;} //!< @brief \f$ P_{l-2}(z) \f$
		real_t const& P_lm1(){return Pl_last;} //!< @brief \f$ P_{l-1}(z) \f$
		real_t const& P_l(){return Pl_this;} //!< @brief \f$ P_{l-2}(z) \f$
		
		//! @brief Reset \ref z to \p z_new and \p l to 0
		void Setup(real_t const z_new)
		{
			z = z_new;
			Reset();
		}
		
		//! @brief Reset \p l to 1, but use the existing \ref z.
		void Reset()
		{			
			// Start with l = 0
			l_this = 1;
			
			Pl_last_coeff = -real_t(l_this); 		// (-l)
			Pl_this_coeff = real_t(2*l_this + 1); 	// (2l + 1)
			Pl_next_coeff = real_t(l_this + 1); 		// (l + 1)
			
			Pl_this = z; // P_0(x) = 1
			
			// Initialize the last values P_0(z) = 1
			Pl_last = real_t(1);
			// There is no second-to-last value, fill Pl_next with NAN
			Pl_next = NAN;
		}
		
		//! @brief Increment \p l and return the \p P_l array.
		real_t const& Next()
		{
			Pl_next = (Pl_this_coeff * z * Pl_this	+ Pl_last_coeff * Pl_last)/Pl_next_coeff;
			++l_this;
			
			Pl_next_coeff += real_t(1); // (l + 1)
			Pl_this_coeff += real_t(2); // (2l + 1)
			Pl_last_coeff -= real_t(1); // (-l)
			
			{
				// last = this
				// this = next
				// next = last (now second-to-last value)
				
				real_t const Pl_secondLast = Pl_last;
				Pl_last = Pl_this; // Pl_this -> Pl_last
				Pl_this = Pl_next; // Pl_next -> Pl_this
				Pl_next = Pl_secondLast; // Pl_last -> Pl_next
			}
			
			return P_l();
		}
};

#endif
