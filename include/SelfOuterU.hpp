#ifndef SELF_OUTER_U
#define SELF_OUTER_U

/*!
 *  @file SelfOuterU.hpp
 *  @brief Tools to calculate the symmetric, self outer product (SOP) \f$ |v\rangle \langle v| \f$
 *  @author Copyright (C) 2017 Keith Pedersen (Keith.David.Pedersen@gmail.com)
 *  @date August 2017
 * 
 *  @anchor SOP
 *  Because the self outer product (SOP) of a vector of numerical data \f$ v_i \f$
 *  \f[
 *      \text{SOP}_{ij} = v_i \times v_j
 *  \f]
 *  is symmetric, it can be compactly packed into upper-triangular form U 
 *  (a 1D array that doesn't store the redundant lower half L).
 *  Splitting the calculation into square tiles speeds up the calculation by:
 *   -# Reducing memory latency through efficient caching.
 *   -# Naturally permitting SSE/AVX SIMD instructions (e.g. via GCC option -ftree-vectorize).
 *   -# Lending itself to multi-threaded calculation.
 * 
 *  The U-form of a \em tiled SOP is depicted below,
 *  for a vector of length 8 and tiles of width 3.
 *  Matrix positions for diagonal tiles use (a,b) and off-diagonal tiles use [a,b].
 *  \verbatim
    
                                   Center <== | ==> Right Edge 
      /-------------------+-------------------+------------------\
      | (0,0) (0,1) (0,2) | [0,3] [0,4] [0,5] | [0,6] [0,7]  X   |
      | (1,0) (1,1) (1,2) | [1,3] [1,4] [1,5] | [1,6] [1,7]  X   |
      | (2,0) (2,1) (2,2) | [2,3] [2,4] [2,5] | [2,6] [2,7]  X   |
      +-------------------+-------------------+------------------+ 
      |   X     X     X   | (3,3) (3,4) (3,5) | [3,6] [3,7]  X   |  
      |   X     X     X   | (4,3) (4,4) (4,5) | [4,6] [4,7]  X   | 
      |   X     X     X   | (5,3) (5,4) (5,5) | [5,6] [5,7]  X   | 
      +-------------------+-------------------+------------------+ 
      |   X     X     X   |   X     X     X   | (6,6) (6,7)  X   | <== Final Diagonal
      |   X     X     X   |   X     X     X   |   X   (7,7)  X   |
      |   X     X     X   |   X     X     X   |   X     X    X   |
      \-------------------+-------------------+------------------/
    \endverbatim
 * 
 *  There are three classes of tiles:
 *   -# Center: Always full tiles. These are the most efficient to 
 *      calculate. Note that diagonal tiles contain 
 *      redundant/double-counted elements (e.g. (0,1) and (1,0)),
 *      while off-diagonal tiles (and the last diagonal tile) do not. 
 *      This is because diagonal tiles are not ragged, a choice made for 
 *      speed and uniformity, as the time to calculate 
 *      redundant (but full) diagonal tiles is smaller than the time to create 
 *      non-redundant (but ragged) tiles (more control logic, fewer SIMD).
 *   -# Right Edge: These tiles may be missing columns on the right, 
 *      because the vector does not perfectly align with tileWidth.
 *      These are still fairly efficient to calculate,
 *      because each column is a full tileWidth.
 *   -# Final Diagonal: Unlike diagonal Center tiles, the final diagonal tile 
 *      does not extend past the diagonal (i.e. has no redundant elements). 
 *      Making it ragged improves performance dramatically when 
 *      \f$ N < tileWidth \f$.
 *  
 *  @anchor special-U-form
 *  The tiled SOP is output in special upper-triangular, column-major U-form:
 *   - The tiles themselves are output in standard column-major U-form
 *     {tile(0,0), tile[0,1], tile(1,1), tile[0,2], ...}.
 *   - The contents of each individual tile are in standard column-major form.
 *   - Center tiles are full, Right Edge tiles have full columns, 
 *     and the Final Diagonal tile is in standard column-major U-form.
 *   - An offDiagonalFactor (e.g. 2) can be multiplied by the elements in 
 *     off-diagonal tiles (and the Final Diagonal tile) to match the 
 *     explicit double-counting in full diagonal tiles.
 * 
 *  Here we show the above example matrix in special-U-form 
 *  (with each tile on its own row):
 *  \verbatim
      SOP = {(0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2),
             [0,3], [1,3], [2,3], [0,4], [1,4], [2,4], [0,5], [1,5], [2,5],
             (3,3), (4,3), (5,3), (3,4), (4,4), (5,4), (3,5), (4,5), (5,5),
             [0,6], [1,6], [2,6], [0,7], [1,7], [2,7],
             [3,6], [4,6], [5,6], [3,7], [4,7], [5,7],
             (6,6), (6,7), (7,7)}
    \endverbatim
 * 
 *  If the vector is not very large, there is no benefit in tiling; use SelfOuterU().
 *  As tiling becomes beneficial, use TiledSelfOuterU(), 
 *  but note that the memory overhead O(N^2) can become excessive.
 *  If portions of the SOP can be used independently, 
 *  then it is highly advantageous to use a TiledSelfOuterU_Incremental object, 
 *  which outputs increments of the SOP of length tileSize**2.
 *  All three require selecting the correct \ref OperatorClass.
 * 
 *  All methods were validated via tester_TiledSelfOuterU.cpp.
*/

#include <vector>
#include <array>
#include <assert.h>
#include "kdp/kdpTools.hpp"

/*! @defgroup OperatorClass 
 *  When the \ref SOP is emplaced in target, we may want to do either
 *  (target = SOP) or (target += SOP). These options are delineated by
 *  using two minimal classes which wrap the chosen operation, 
 *  and are then passed as template arguments for the OuterU tools.
 *  This slims down the machine code by removing a conditional branch 
 *  inside the main calculation loop (which would otherwise have to 
 *  use a bool to choose between the two operations at run time). 
 *  The wrapper class also informs whether the target should be resized.
 *  Perhaps this is not the most elegant way to do this, but it works.
*/ 

/*! @ingroup OperatorClass
 *  @brief An \ref OperatorClass wrapper for the equals (=) operation.
*/ 
template<typename T>
struct Equals
{
	typedef T value_type;
	
	//! @brief We should resize the vector before setting.
	static constexpr bool doResize = true;
	
	//! @brief lhs = rhs
	inline static T& Operate(T& lhs, T const rhs)
	{
		return lhs = rhs;
	}
};

/*! @ingroup OperatorClass
 *  @brief An \ref OperatorClass wrapper for the plus-equals (+=) operation.
*/ 
template<typename T>
struct PlusEquals
{
	typedef T value_type;
	
	//! @brief Do not resize the vector before setting.
	static constexpr bool doResize = false;
	
	//! @brief lhs += rhs
	inline static T& Operate(T& lhs, T const rhs)
	{
		// We could probably adopt this to use std::fma
		return lhs += rhs;
	}
};

////////////////////////////////////////////////////////////////////////

/*! @brief Emplace a \em portion of an \ref SOP (in U-form) into \p target
 *  using the un-tiled algorithm, starting at \p col_begin and \p row_begin 
 *  and ending at \p col_end.
 * 
 *  @warning This function is a worker function for other SelfOuterU functions,
 *  and is not intended to be called directly by the user.
 *  @warning There is no access to the target object; it must be resized/trimmed externally.
 *  @note No tiling; the \p target is set in \em standard U-form.
 *  
 *  @tparam T 		The data type
 *  @tparam operation	The operation to perform when setting \p target (see \ref OperatorClass).
 *  @tparam source_t 	The iterator type to the source
 *  @tparam target_t    The iterator type to the target
 *  @param source_begin 	The first index in the source.
 *  @param source_end 	One past the last index in the source.
 *  @param target		The output location where the SOP is set (in upper-triangular U-form).
 *  @param offDiagonalFactor	Multiplied by off-diagonal elements.
*/
template<template<typename T> class operation, typename T, class source_it, class target_it>
target_it SelfOuterU_Worker(source_it const source_begin, source_it const source_end, 
	target_it target, T const offDiagonalFactor)
{
	static_assert(std::is_same<typename std::iterator_traits<source_it>::value_type, T>::value, 
		"SelfOuterU: source_it must have value_type as operation");
	static_assert(std::is_same<typename std::iterator_traits<target_it>::value_type, T>::value, 
		"SelfOuterU: target_it must have value_type as operation");
	assert(source_begin <= source_end);
			
	for(auto j = source_begin; j < source_end; ++j)
	{
		// Apply the off-diagonal factor to off-diagonal elements
		for(auto i = source_begin; i < j; ++i)
			operation<T>::Operate(*(target++), offDiagonalFactor * ((*i) * (*j)));
		
		// Each column's last/diagonal element
		operation<T>::Operate(*(target++), (*j) * (*j));
	}
	return target; // Inform the caller where target ended.
}

////////////////////////////////////////////////////////////////////////

/*! @brief Emplace the \em entire \ref SOP (in U-form) of \p source into \p target 
 *  using the un-tiled algorithm.
 * 
 *  @note No tiling; the \p target is set in \em standard U-form.
 *  @note When requested by \p operation::doResize, \p target is trimmed to fit.
 *  
 *  @tparam operation	The operation to perform when setting \p target (see \ref OperatorClass).
 *  @param source 	The input vector.
 *  @tparam T 	The data type.
 *  @param target		The output vector where the SOP is set (in upper-triangular U-form).
 *  @param offDiagonalFactor	Multiplied by off-diagonal elements.
 */
template<template<typename T> class operation, typename T>
void SelfOuterU(std::vector<T> const& source, std::vector<T>& target, 
	T const offDiagonalFactor)
{
	if(operation<T>::doResize)
		target.resize(kdp::GaussSum(source.size()));
	
	auto const target_end_actual = 
		SelfOuterU_Worker<operation, T>
			(source.cbegin(), source.cend(), target.begin(), offDiagonalFactor);
	
	assert(target_end_actual == target.end());
}

////////////////////////////////////////////////////////////////////////

/*! @brief Emplace the \em entire \ref SOP (in \ref special-U-form)
 *  of \p source into \p target using the tiled algorithm.
 *  
 *  @note When requested by \p operation::doResize, \p target is trimmed to fit.
 *  
 *  @tparam operation	The operation to perform when setting \p target (see \ref OperatorClass).
 *  @tparam tileWidth	The width of the tiles (must be a power of 2).
 *  @tparam T 	The data type.
 *  @param source 	The input vector.
 *  @param target		The output vector where the SOP is set (in \ref special-U-form).
 *  @param offDiagonalFactor	Multiplied by off-diagonal elements.
*/ 
template<template<typename T> class operation, size_t tileWidth, typename T>
void TiledSelfOuterU(std::vector<T> const& source, 
	std::vector<T>& target, T const offDiagonalFactor)
{
	static_assert(kdp::IsPowerOfTwo(tileWidth), "TiledSelfOuterU: tileWidth must be a power of two");
	
	size_t const numFullTiles = source.size() / tileWidth;
	size_t const centralSpan = numFullTiles * tileWidth;
	size_t const overSpill = source.size() - centralSpan;
	
	// 1 past the last element in the central span
	auto const center_end = source.begin() + centralSpan;
	
	size_t const newSize = 
		kdp::GaussSum(numFullTiles) * tileWidth * tileWidth // full tiles
			+ overSpill * centralSpan // right edge
				+ kdp::GaussSum(overSpill); // final diagonal
	
	if(operation<T>::doResize) // When we set, we must resize the target.
		target.resize(newSize);
	else // When we add to, the target should already be the correct size
		assert(target.size() == newSize);
	
	// The "local" pointer corresponds to each tile's first member
	// Each vectorized setter-loop will place it's data here.
	auto target_local = target.begin();
	// target_local should be incremented after each settwe-loop.
	
	// Fill center (full) tiles
	
	// Outer loop over columns
	for(auto j_local = source.begin(); j_local < center_end; j_local += tileWidth)
	{
		// Inner loop over rows
		for(auto i_local = source.begin(); i_local <= j_local; i_local += tileWidth)
		{
			// Diagonal tiles have (i_local == j_local)
         T const symmetryFactor = (i_local == j_local) ? 1 : offDiagonalFactor;

			// The inner loop is vectorized
         for(size_t j = 0; j < tileWidth; ++j)
         {
            for(size_t i = 0; i < tileWidth; ++i)
            {
					operation<T>::Operate(target_local[i], 
						symmetryFactor * (i_local[i] * j_local[j]));
            }
            target_local += tileWidth;
         }
      }
   }
   
   // Right edges (partially full) tiles ... always off-diagonal (symmetryFactor = offDiagonalFactor)
   
	for(auto i_local = source.begin(); i_local < center_end; i_local += tileWidth)
	{
		// When vectorizing, keep the ragged (i.e. non-aligned) loop on the outside
		for(auto j = center_end; j < source.end(); ++j)
		{
			for(size_t i = 0; i < tileWidth; ++i)
			{
				operation<T>::Operate(target_local[i], 
					offDiagonalFactor * (i_local[i] * (*j)));
			}
			target_local += tileWidth;
		}
	}
	
	// Final diagonal tile
	
	auto const target_end = 
		SelfOuterU_Worker<operation, T>
		(center_end, source.end(), target_local, offDiagonalFactor);
	
	assert(target_end == target.end());	
}

/*! @brief Calculate increments of the \ref SOP of \p source, in \ref special-U-form, 
 *  using the tiled algorithm.
 * 
 *  Each increment (of length \p tileWidth**2) is written to \p target (std::array) via Next().
 *   - Center tiles each fill an entire increment.
 *   - Right Edge tiles are not full (missing columns) and cannot fill an entire increment.
 *     This space is filled by pulling the next tile into the current increment
 *     (see example below).
 *   - The Final Diagonal tile can occur in two places:
 *      - If the final Right Edge increment has room for the entire
 *        Final Diagonal, it is placed there.
 *      - Otherwise (no room), the final Right Edge increment is padded 
 *        with zeroes and the Final Diagonal occurs in the next increment
 *        (also padded with zeroes).
 *   - After the final increment, additional calls to Next() return immedietely.
 * 
 *  Here we depict the increments for the example matrix from \ref special-U-form
 *  (with each increment on its own row, and tiles seperated by ; ):
 *  \verbatim
      SOP =  (0,0), (1,0), (2,0), (0,1), (1,1), (2,1), (0,2), (1,2), (2,2)%
             [0,3], [1,3], [2,3], [0,4], [1,4], [2,4], [0,5], [1,5], [2,5]%
             (3,3), (4,3), (5,3), (3,4), (4,4), (5,4), (3,5), (4,5), (5,5)%
             [0,6], [1,6], [2,6], [0,7], [1,7], [2,7]% [3,6], [4,6], [5,6],
             [3,7], [4,7], [5,7]% (6,6), (6,7), (7,7)%   0      0      0 
    \endverbatim
 * 
 *  TiledSelfOuterU_Incremental is \em not thread safe, but can be made so
 *  by guarding each call to Setup() or Next() with a unique_lock (in case of exception).
 *  Handling thread safety externally gives more freedom to the user 
 *  (e.g. perhaps 4 separate TiledSelfOuterU_Incremental need to call Next(), 
 *  but also must remain synchronized with each other, so all 4 should use the same mutex).
 * 
 *  @tparam T 	The data type.
 *  @tparam operation	The operation to perform when setting \p target (see \ref OperatorClass).
 *  @tparam tileWidth	The width of the tiles (must be a power of 2).
*/
template<typename T, template<typename T> class operation, size_t tileWidth>
class TiledSelfOuterU_Incremental
{
	static_assert(kdp::IsPowerOfTwo(tileWidth), "TiledSelfOuterU_Incremental: tileWidth must be a power of two");
	
	public:
		//! @brief the size of one increment = tileWidth**2 (one full tile's worth of information).
		static size_t constexpr incrementSize = tileWidth * tileWidth;
				
		//! @brief The source is a std::vector, because it has undetermined length.
		typedef typename std::vector<T> source_t;
		
		//! @brief The target is a std::array (superior for optimizing SIMD)
		//! because incrementSize is known.
		typedef typename std::array<T, incrementSize> target_t;
	
	private:
		typedef typename source_t::const_iterator source_iter_t;
		typedef typename target_t::iterator target_iter_t;		
		
		// The number of redundant elements in diagonal center tiles
		static size_t constexpr redundantSize = (tileWidth * (tileWidth - 1))/2;
		 
		/* A type-safe enum for the internal state of the outer product.
		 *  - Center:  calculating full, square tiles in the center of the matrix.
		 *  - RightEdge:  partially-full tiles along the right edge of the matrix.
		 *  - FinalDiagonal_piggyback:  the ragged, Final Diagonal tile 
		 *    os piggybacking on the final increment containing the Right Edge.
		 *  - FinalDiagonal_standalone:  the ragged, Final Diagonal tile 
		 *    will be in its own increment.
		 *  - Done: calculation is complete, calls to Next() immediately return.
		*/ 
		enum class OuterState {Center, RightEdge, 
			FinalDiagonal_piggyback, FinalDiagonal_standalone, Done};

		// We must store progress between increments; i = row, j = col
		// "local" positions mark the first member in each tile
		source_iter_t i_local, j_local, center_end;
		source_t const* source; // Use a pointer to rebind source every Setup()
		T offDiagonalFactor; // Scale off-diagonal elements by 
		OuterState state; // The current state of the outer product
		OuterState finalDiagonalState; // The state to use for the Final Diagonal
		
		void BeDone(); // Enter a "Done" state; calls to Next() return immedietely.
		 
	public:
		/*! @brief Initialize the object so that calling Next() does nothing.
		 * 
		 *  This makes it safe to call Next() before Setup() 
		 *  (even though doing so makes no sense).
		*/ 
		TiledSelfOuterU_Incremental() {BeDone();}
		
		/*! @brief Estimate the number of increments required to calculate
		 *  \p source's SOP, without actually binding to \p source.
		 * 
		 *  The estimate does not account for the exact position of the 
		 *  FinalDiagonal, so may be short by one.
		 * 
		 *  @param source		The input vector.
		 *  @return The estimated number of increments in the SOP of \p source.
		*/
		static size_t EstimateIncrements(source_t const& source);
		
		/*! @brief Bind to \p source, in preparation to calculate its SOP,
		 *  but don't calculate anything yet.
		 * 
		 *  @warning No guarantee is made that the last SOP was completed.
		 * 
		 *  @param source		The input vector.
		 *  @param offDiagonalFactor	Multiplied by off-diagonal elements.
		 *  @return The number of increments that will be returned by Next().
		*/ 
		size_t Setup(source_t const& source, T const offDiagonalFactor);

		/*! @brief Emplace the next increment of the SOP into \p target (in special-U-form).
		 * 
		 *  When the increment is not full (potentially the last two),
		 *  \p target is padded with zeroes on the right.
		 *  When the SOP is already done, \p target is unaltered.
		 *  Hence, the return informs the state of the SOP calculation
		 *  (so that while(obj.Next()) is a good way to make a loop):
		 *  - normal (full) increment ... return == \p incrementSize
		 *  - final (usually partial) increment ... 0 < return <= \p incrementSize
		 *  - SOP done ... return == 0 (\p target unaltered)
		 * 
		 *  @param target		The output array where the SOP increment is emplaced (in \ref special-U-form).
		 *  @return The size of the increment emplaced in \p target (not counting zero fill).
		 */
		size_t Next(target_t& target);
};

template<typename T, template<typename T> class operation, size_t tileWidth>
void TiledSelfOuterU_Incremental<T, operation, tileWidth>::BeDone()
{
	// i_local = j_local = center_end = source_iter_t(); // A singular value, so we can't derefernce anyway, so why set it at all.
	//~ source = nullptr; // Also probably unnecessary
	//~ offDiagonalFactor = 1.; // Also probably unnecessary
	//~ finalDiagonalState = OuterState::standalone; // Also probably unnecessary
	state = OuterState::Done; // <== Only thing that actual matters
}

template<typename T, template<typename T> class operation, size_t tileWidth>
size_t TiledSelfOuterU_Incremental<T, operation, tileWidth>::EstimateIncrements(source_t const& source_in)
{
	// (The size of the untiled SOP) + (the number of full tiles) * redundantSize
	return MinPartitions(GaussSum(source_in.size()) + (source_in.size() / tileWidth) * redundantSize, tileWidth);
}

template<typename T, template<typename T> class operation, size_t tileWidth>
size_t TiledSelfOuterU_Incremental<T, operation, tileWidth>::Setup
	(source_t const& source_in, T const offDiagonalFactor_in)
{
	if(source_in.size() == 0)
	{
		// If the vector is empty, so is the SOP
		BeDone();
		return 0;
	}
	else
	{		
		source = &source_in;
		j_local = (i_local = source->cbegin());
		offDiagonalFactor = offDiagonalFactor_in;
		
		size_t const numFullTiles = source->size() / tileWidth;
		size_t const centralSpan = numFullTiles * tileWidth;
		
		// the center ends after the central span
		center_end = source->cbegin() + centralSpan;
		assert(center_end <= source->cend());
		
		if(numFullTiles == 0)
		{
			state = OuterState::FinalDiagonal_standalone; // no need to alter finalDiagonalState
			
			return 1; // 1 increment (the final diagonal)
		}
		else
		{	
			state = OuterState::Center;
			size_t numIncrements = kdp::GaussSum(numFullTiles); // the number of center tiles
			
			size_t const overSpill = source->size() - centralSpan;
			
			if(overSpill > 0)
			{
				size_t const rightEdgeSize = centralSpan * overSpill;
				size_t const rightEdgeIncrements = kdp::MinPartitions(rightEdgeSize, incrementSize);
				
				numIncrements += rightEdgeIncrements;
			
				// How much room is there for the final diagonal in the 
				// last increment filled by the right edge.
				size_t const roomForFinalDiagonal = rightEdgeIncrements * incrementSize - rightEdgeSize;
				
				// If there is enough room for the entire final diagonal, 
				// we will place it in the final increment of the right edge.
				// Otherwise we will place it in a new increment of its own
				// (since we'll need one more increment anyway).
				if(roomForFinalDiagonal >= kdp::GaussSum(overSpill))
					finalDiagonalState = OuterState::FinalDiagonal_piggyback;
				else
				{
					finalDiagonalState = OuterState::FinalDiagonal_standalone;
					numIncrements += 1;
				}
			}
			return numIncrements;
		}
	}
}

template<typename T, template<typename T> class operation, size_t tileWidth>
size_t TiledSelfOuterU_Incremental<T, operation, tileWidth>::Next
	(target_t& target)
{
	if(state == OuterState::Done)
		return 0; // Don't alter target
	else
	{
		auto target_local = target.begin();
		
		if(state == OuterState::Center) // Fill an entire increment
		{
			// Diagonal tiles have the same local index
			bool const diagonalTile = (i_local == j_local);
			{
				T const symmetryFactor = diagonalTile ? 1. : offDiagonalFactor;

				for(size_t j = 0; j < tileWidth; ++j)
				{
					for(size_t i = 0; i < tileWidth; ++i)
					{
						operation<T>::Operate(target_local[i], 
							symmetryFactor * (i_local[i] * j_local[j]));
					}
					target_local += tileWidth;
				}
			}
			
			if(not diagonalTile)
				i_local += tileWidth; // Move down to the next tile in the column
			else
			{
				j_local += tileWidth; // Move right, to the next column of tiles
				i_local = source->cbegin(); // Start at the top of the column			
				
				// Detect if we're done with the Center
				if(j_local == center_end)
				{
					if(center_end < source->end()) // Test for a right edge
						state = OuterState::RightEdge; // i_local is in place
					else // Perfect alignment with tileWidth, we're done
						BeDone();
				}
			}
			assert(target_local == target.end());
		}//end Center
		else // j_local will now be treated as j
		{
			// switch is the correct control structure because 
			// sometimes we need to flow from RightEdge to FinalDiagonal
			switch(state)
			{
				case OuterState::RightEdge:
				{
					while(target_local < target.end()) // Fill till full increment
					{
						for(size_t i = 0; i < tileWidth; ++i) // Fill full columns from the tile
						{
							// The RightEdge is always off-diagonal
							operation<T>::Operate(target_local[i], 
								offDiagonalFactor * (i_local[i] * (*j_local)));
						}
						target_local += tileWidth;
						++j_local; // Move to the next column in the tile
											
						if(j_local == source->end()) // no more columns in tile
						{
							i_local += tileWidth; // Move down to the next tile
							j_local = center_end; // Start at its first column
							
							if(i_local == center_end) // Just filled last tile in right edge
							{
								// It is time to move to the final diagonal
								state = finalDiagonalState;
								// From now on, i_local and j_local will not be used
																
								break; // out of while loop
							}
						}
					}// end while loop (fill a full incrememnt)
				}// end RightEdge
				
				// Move directly to FinalDiagonal IFF piggybacking, otherwise break
				if(state not_eq OuterState::FinalDiagonal_piggyback)
					break; // out of case
				
				case OuterState::FinalDiagonal_standalone:
				{
					assert(target_local <= target.end());
					
					target_local = 
						SelfOuterU_Worker<operation, T>
							(center_end, source->end(), target_local, offDiagonalFactor);
									
					assert(target_local <= target.end());
					BeDone();
				}
				break;
				
				default:
					throw std::runtime_error("TiledSelfOuterU_Incremental: Unanticipated control logic!");
				break;
			}// end case
						
			// Pad any remaining spaces with zeroes
			if(target_local < target.end())
			{
				size_t const lengthSet = (target_local - target.begin());
				assert(lengthSet < incrementSize);
				
				while(target_local < target.end())
					*(target_local++) = 0.;
					
				return lengthSet;
			}			
		}
		return incrementSize; // Catch-all return for full tileWidth
	}
}

#endif
