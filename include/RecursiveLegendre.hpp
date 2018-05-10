#ifndef RECURSIVE_LEGENDRE
#define RECURSIVE_LEGENDRE

#include <type_traits>
#include <cmath>
#include <cstddef>
#include <assert.h>

/*! @brief Recursively calculate the Legendre polynomial P_l for an array/vector of z values.
 * 
 *  \f$ (l+1)P_{l+1}(z) = (2l+1)z\,P_l(z) - l\,P_{l-1}(z) \f$
 * 
 *  Operating recursively on an array of values allows efficient auto-vectorization
 *  to SIMD instructions, with high cache-hit efficiency. As such,
 *  \p arraySize should be chosen to be some power-of-2 which is 
 *  large enough to span several cache lines.
 * 
 *  \note The non-virtual version is significantly smaller
 *  
 *  \tparam real_t 		the floating-point type used by P_l (a real-valued function)
 *  \tparam arraySize 		the size of the base array (could be any value, even 1, but powers of 2 [especially 32, 64, 128] work best).
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
			
		size_t l_this; // The Legendre index l for Pl_this
		
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
			for(size_t i = 0; i < z.size(); ++i)
				vec[i] = val;
		}
		
	public:
		/*! @brief The z-values for which P_l is calculated
		 *  
		 *  The public can access z so they are not forced to Setup()
		 *  by passing z as an argument (e.g. if some f() takes z by reference to fill it).
		 *  
		 *  \warning The public must then call Reset() before calling Next(), 
		 *  otherwise the results will be nonsense. 
		*/ 
		container_t z;
		
		//! @brief Initialize the object in a valid state.
		RecursiveLegendre_Increment()
		{
			Fill(z, real_t(0));
			Reset();
		}
		
		size_t l() {return l_this;} //!< @brief The current Legendre index l.
		
		inline container_t const& P_lm2(){return *Pl_next;} //!< @brief \p P_{l-2}
		inline container_t const& P_lm1(){return *Pl_last;} //!< @brief \p P_{l-1}
		inline container_t const& P_l(){return *Pl_this;} //!< @brief \p P_l
		
		//! @brief Reset \ref z to \p z_new and \p l to 0.
		void Setup(container_t const& z_new)
		{
			z = z_new;
			Reset();
		}
		
		//! @brief Reset \ref l_this to 1, but use the existing \ref z.
		void Reset()
		{			
			Pl_last = &Pl_A;
			Pl_this = &Pl_B; 
			Pl_next = &Pl_C;
						
			// Start with l = 1
			l_this = 1;
			
			Pl_last_coeff = -real_t(l_this); 		// (-l)
			Pl_this_coeff = real_t(2*l_this + 1); 	// (2l + 1)
			Pl_next_coeff = real_t(l_this + 1); 		// (l + 1)
			
			*Pl_this = z; // P_1(x) = z
			
			// FORMERALY IMPORTANT: before we begin, we MUST initialize Pl_last,
			// even though Pl_last_coeff = 0 in the first iteration.
			// This is because 0.*nan => nan, contamination all subsequent iterations.
			// We do this every time we Reset(), in case the last z passed in a nan, 
			// or if Pl_last was never initialized (so that it randomly contains a 
			// numbers with the special nan exponent). 
			// Of course, this does not protect against nan in the current z.
			Fill(*Pl_last, real_t(1));
			// Now that we've added access to P_lm2, fill Pl_next with NAN
			Fill(*Pl_next, NAN);
		}
		
		//! @brief Increment \p l and return the \p P_l array.
		container_t const& Next()
		{
			// NOTE: gcc won't auto-vectorize if the iterator has a 
			// smaller data type than the data type of the loop bound, 
			// because it can't ensure that the loop will terminate 
			// (from iterator overflow). Must use size_t.
			for(size_t i = 0; i < z.size(); ++i)
			{
				// To auto-vectorize a loop, don't try to mess with registers 
				(*Pl_next)[i] = 
					(Pl_this_coeff * z[i] * (*Pl_this)[i] 
					+ Pl_last_coeff * (*Pl_last)[i])/Pl_next_coeff;
			}
			
			++l_this;
			Pl_next_coeff += real_t(1); // (l + 1)
			Pl_this_coeff += real_t(2); // (2l + 1)
			Pl_last_coeff -= real_t(1); // (-l)
			
			{
				// last = this
				// this = next
				// next = last
				// The next time Next() is called, we overwrite next,
				// but until then it provides access to the second-to-last value 
				
				container_t* __restrict const Pl_secondLast = Pl_last;
				Pl_last = Pl_this; // Pl_this -> Pl_last
				Pl_this = Pl_next; // Pl_next -> Pl_this
				Pl_next = Pl_secondLast; // Pl_last -> Pl_next
			}
			
			return P_l();
		}
};

// In this case, it is better to repeat yourself and merely double-check the math
template<typename real_t>
class RecursiveLegendre
{
	private:
		static_assert(std::is_floating_point<real_t>::value, 
			"RecursiveLegendre_Increment: container_t::value_type must be a floating point type");
			
		real_t Pl_last, Pl_this, Pl_next;
	
		// The "coeff" is the numerical pre-factor in the P_l recursion (e.g. (2l + 1))
		real_t Pl_next_coeff, Pl_this_coeff, Pl_last_coeff;
		
		size_t l_this; // The Legendre index l for Pl_this
	
	public:
		real_t z;
		
		//! @brief Initialize the object in a valid state.
		RecursiveLegendre()
		{
			z = real_t(0);
			Reset();
		}
		
		size_t l() {return l_this;} //!< @brief The current Legendre index l.
		
		real_t const& P_lm2(){return Pl_next;} //!< @brief \p P_{l-2}
		real_t const& P_lm1(){return Pl_last;} //!< @brief \p P_{l-1}
		real_t const& P_l(){return Pl_this;} //!< @brief \p P_l
		
		//! @brief Reset \ref z to \p z_new and \p l to 0
		void Setup(real_t const z_new)
		{
			z = z_new;
			Reset();
		}
		
		//! @brief Reset \ref l_this to 1, but use the existing \ref z.
		void Reset()
		{			
			// Start with l = 0
			l_this = 1;
			
			Pl_last_coeff = -real_t(l_this); 		// (-l)
			Pl_this_coeff = real_t(2*l_this + 1); 	// (2l + 1)
			Pl_next_coeff = real_t(l_this + 1); 		// (l + 1)
			
			Pl_this = z; // P_0(x) = 1
			
			// IMPORTANT: before we begin, we MUST initialize Pl_last,
			// even though Pl_last_coeff = 0 in the first iteration.
			// This is because 0.*nan => nan, contamination all subsequent iterations.
			// We do this every time we Reset(), in case the last z passed in a nan, 
			// or if Pl_last was never initialized (so that it randomly contains a 
			// numbers with the special nan exponent). 
			// Of course, this does not protect against nan in the current z.
			Pl_last = real_t(1);
			// Now that we've added access to P_lm2, set it to NAN
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
				// next = last
				// The next time Next() is called, we overwrite next,
				// but until then it provides access to the second-to-last value 
				
				real_t const Pl_secondLast = Pl_last;
				Pl_last = Pl_this; // Pl_this -> Pl_last
				Pl_this = Pl_next; // Pl_next -> Pl_this
				Pl_next = Pl_secondLast; // Pl_last -> Pl_next
			}
			
			return P_l();
		}
};

#endif
