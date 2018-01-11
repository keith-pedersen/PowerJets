#ifndef HELPER_TOOLS
#define HELPER_TOOLS

/*!
 *  @file helperTools.hpp
 *  @brief Some short utlity functions.
 * 
 *  A constexpr function can be evaluated at compile time.
 * 
 *  @author Copyright (C) 2018 Keith Pedersen (Keith.David.Pedersen@gmail.com)
 *  @date Jan 2018
*/

#include <sys/stat.h> // POSIX stat()
#include <assert.h>
#include <string>
#include <vector>
#include <array>
//~ #include <QtCore/QSettings>
#include <sstream>
#include <cmath>
#include <limits>

////////////////////////////////////////////////////////////////////////

template<typename real_t>
real_t Diff2(real_t const a, real_t const b)
{
	return (a - b)*(a + b);
}
 
////////////////////////////////////////////////////////////////////////

// Note: Commented out because it is not a template, 
// so if this file is included in multiple files which build object files, 
// it will receive multiple definitions, and cause a linking error 
// when the shared library is built
/*! @brief Quickly check if a file is visible.
 * 
 *  Mainly useful for control logic before sending file to another function for internal use.
*/
// Based on: http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
//~ bool FileIsVisible(const std::string& name) 
//~ {
	//~ // per: http://stackoverflow.com/questions/23329382/function-and-struct-having-the-same-name-in-c,
	//~ // A struct and a function/variable/etc can have the same name. 
	//~ // The function/variable/etc hides the struct, 
	//~ // which is identified by the struct prefix
	//~ struct stat buffer; // POSIX uses this name hiding paradigm to associate structs with functions
	//~ // stat::Return(): Upon successful completion, 0 shall be returned. 
	//~ // Otherwise, -1 shall be returned and errno set to indicate the error.
	//~ return (stat(name.c_str(), &buffer) == 0); // Any error means we can't read the file.
//~ }

////////////////////////////////////////////////////////////////////////

/*! @brief Parse a string (e.g. from an INI file) to read the angle,
 *  allowing input in both radians and degrees.
 *  
 *  Assume radians if the angle is purely numeric. Accept suffixes containing 
 *  two known substrings:
 *   - "deg" means degrees (so {degs, degree, or degrees} will also work).
 *   - "rad" means radians (so {rads, radian, or radians} will also work).
 * 
 *  \returns the parsed angle, in radians
 *  
 *  \note There must be a space between the numbers and the suffix.
*/
template<typename real_t>
real_t ReadAngle(std::string const& toParse)
{
	// static means we need only one instance of the variable for every call to the function,
	// which in this case is true and useful
	// constexpr means that the value is known at compile time,
	// so the compiler can potentially optimize the variable away.
	static constexpr char const* degreeKey = "deg"; // a std::string cannot be a constexpr
	static constexpr char const* radianKey = "rad";
	static constexpr real_t degreesToRadians = M_PI / 180.;
	
	real_t angle;
	std::stringstream parser(toParse);
	parser >> angle;
		
	if(not parser)
		throw std::runtime_error(("ReadAngle: cannot parse <" + toParse + ">.").c_str());
	
	// Angles with no units are assumed radians; attempt to parse the units
	std::string units;
	parser >> units;
		
	if(units.length() > 0) // A suffix existed
	{
		// Look for degreeKey in the suffix (starting at position 0). 
		if(units.find(degreeKey, 0) not_eq std::string::npos)
		{
			printf("Converted degrees to radians.\n");
			angle *= degreesToRadians;
		}
		// Else if we find radians, nothing to do (no conversion needed)
		else if(units.find(radianKey, 0) not_eq std::string::npos) {}
		// A non-empty string which doesn't contain 'rad' or 'deg' has an unsupported unit
		else
		{
			throw std::runtime_error(("ReadAngle: unsupported angular units ... '" + units + "'."
				 + " Only a string containing 'deg' or 'rad' work (e.g. 'deg', 'degs', 'degree' or 'degrees'"
				 + " and the same variations on 'rad')."
				 + " Radians are assumed in the absence of a qualified degree keyword,"
				  + " but we don't know what to do with this.").c_str());
		}
		// Otherwise we found radians 
	}
	// else no units to extract, return angle (in radians) "as is"
	
	return angle;
}

////////////////////////////////////////////////////////////////////////

//! @brief Partition n things into partitions of size m. How many do you need?
template<typename uint_t>
uint_t MinPartitions(uint_t const n, uint_t const m)
{
	return (n - 1)/m + 1;
}

////////////////////////////////////////////////////////////////////////

//! @brief Round a value to the nearest pitch (e.g. if pitch is 3, round value = 10 to 9). 
template<typename real_t>
real_t RoundToNearestPitch(real_t const value, real_t const pitch)
{
	return pitch * std::round(value / pitch);
}

////////////////////////////////////////////////////////////////////////

/*! @brief Unbiased relative difference of \p a and \p b.
 * 
 *  Returns a value in [-1, 1] IFF both arguments have the same sign (or at most one is zero). 
*/
template<typename real_t>
real_t RelDiff(real_t const a, real_t const b)
{
	return (a - b)/(a + b);
}

////////////////////////////////////////////////////////////////////////

//! @brief sum of {1, 2, ..., n}
template<typename uint_t>
uint_t GaussSum(uint_t const n)
{
	return (n*(n + 1))/2;
}

////////////////////////////////////////////////////////////////////////

//! @brief Return the value of the smallest set bit (2**index, not index)
template<typename uint_t>
constexpr uint_t SmallestSetBit(uint_t const x)
{
	static_assert(not std::numeric_limits<uint_t>::is_signed, "SmallestBit: type must be UN-signed");
	
	// Developed from 
	// http://www.exploringbinary.com/ten-ways-to-check-if-an-integer-is-a-power-of-two-in-c/
	return (x bitand (compl(x) + 1));
}

////////////////////////////////////////////////////////////////////////

/*! @brief Return the value of the largest set bit (2**index, not index)
 * 
 *  Does a stupid search (not x86 BSR instruction), 
 *  b/c x86 tutorials are difficult to understand (and not portable to other arch).
 *  A binary search *might* be faster, but uses much more code.
*/ 
template<typename uint_t>
uint_t LargestBit(uint_t x)
{
	static_assert(not std::numeric_limits<uint_t>::is_signed, "LargestBit: type must be UN-signed");
	
	if(x == 0) return 0;
	
	// We can start with the smallest set bit and move on from there
	uint_t largestBit = SmallestSetBit(x);
	x /= (2 * largestBit);
	
	while(x)
	{
		x /= 2;
		largestBit *= 2;
	}
	return largestBit;
}

////////////////////////////////////////////////////////////////////////

/*! @brief Determines if the argument is an exact power of 2
*/ 
template<typename uint_t>
constexpr bool IsPowerOfTwo(uint_t const x)
{
	static_assert(not std::numeric_limits<uint_t>::is_signed, "IsPowerOfTwo: type must be UN-signed");
	
	return ((x not_eq 0) and (x == SmallestSetBit(x)));
}

////////////////////////////////////////////////////////////////////////

/*! @brief Binary accumulate a std::array with \p size = (exact power of 2).
 * 
 *  @warning The array (passed by ref) is accumulated in place (i.e. destructively altered).
*/
template<typename T, size_t arraySize>
T BinaryAccumulate_Destructive(std::array<T, arraySize>& vec)
{
	// We can use a static assert because std::array has static size
	static_assert(IsPowerOfTwo(arraySize), "BinaryAccumulate(std::array): array size must be an exact power of 2");
		
	size_t size = arraySize;
	
	while(size > 1)
	{
		size /= 2; // Equivalent to >>=, but more readible
		for(size_t i = 0; i < size; ++i)
			vec[i] += vec[size + i];
	}
	
	return vec.front();
}

////////////////////////////////////////////////////////////////////////

/*! @brief Binary accumulate a std::array with size = (exact power of 2).
 * 
 *  The accumulation is non-destructive (array is passed by value, 
 *  then passed to BinaryAccumulate_Destructive()).
*/
template<typename T, size_t arraySize>
T BinaryAccumulate(std::array<T, arraySize> vec)
{
	return BinaryAccumulate_Destructive(vec);
}

////////////////////////////////////////////////////////////////////////

/*! @brief Binary accumulate a std::vector of any size.
 * 
 *  @warning The array (passed by ref) is accumulated in place (i.e. destructively altered).
 *  For efficiency, the first reduction does not span 100% of elements, 
 *  but makes the vector's new size an exact power of two. This speeds up all 
 *  subsequent reductions, which do not require an oddness check
 *  and will have good alignment in memory.
*/  
template<typename T>
T BinaryAccumulate_Destructive(std::vector<T>& vec)
{
	// 1. Find the smallest power-of-two less than vec.size()
	size_t size = LargestBit(vec.size());
	
	{
		// 2. Determine how many values are past the power of two
		size_t const overflow = vec.size() - size;
		assert(overflow < size); // unsigned arithmetic, so also asserts overflow >= 0
	
		// 3. Add the overflow to the front of the list (assert guarantees enough room)
		for(size_t i = 0; i < overflow; ++i)
			vec[i] += vec[size + i];
	}
	
	// Now we can do a fast, power-of-two accumulate (no oddness check)
	vec.resize(size); // <== Is this helpfull?
	
	while(size > 1)
	{
		size /= 2; // Equivalent to >>=, but more readible
		for(size_t i = 0; i < size; ++i)
			vec[i] += vec[size + i];
	}
	
	return vec.front();
}

////////////////////////////////////////////////////////////////////////

/*! @brief Binary accumulate a std::vector of any size.
 * 
 *  The accumulation is non-destructive (array is passed by value, 
 *  then passed to BinaryAccumulate_Destructive()).
*/
template<typename T>
T BinaryAccumulate(std::vector<T> vec)
{
	return BinaryAccumulate_Destructive(vec);
}

// What is this left over from?
//~ template<typename real_t, size_t incrementSize>
//~ using incrementArray_t = std::array<real_t, incrementSize>;
#endif
