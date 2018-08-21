#include "RecursiveLegendre.hpp"
#include <fstream>
#include <assert.h>
#include <vector>
#include <cmath>
#include <QtCore/QSettings>

/*! @file tester_RecursiveLegendre.cpp
 *  @brief Validate RecursiveLegendre 
 * 
 *  The simplest way to test RecursiveLegendre is to ensuring that P_l(z_0) = 0
 *  for a collection of roots z_0 found by Mathematica with WorkingPrecision->50.
 *  Because of rounding error, we cannot find exact zeroes. 
 *  Additionally, because of the reduced precision near 1, 
 *  the exact root may not be representable. Thus, we merely ensure that 
 *  P_l(z_0) ~= 0. This requires comparing the value at the root 
 *  to the local slope, which we can compute numerically.
*/ 

int main()
{
	QSettings parsedINI("tester_RecursiveLegendre.ini", QSettings::IniFormat);
	
	double const limit = parsedINI.value("limit", 1e-8).toDouble();
		
	using container_t = std::vector<double>;
	RecursiveLegendre_Increment<container_t> Pl_computer;
	
	// Mathematica has calculated zeroes for l = 1 to l = lMax
	// using arbitrary precision arithmetic (more than quad precision)	
	size_t const lMax = [&]()
	{
		std::ifstream LegendreZeroes("./LegendreZeroes.dat");
		std::string line;
		
		// The first line in the file is lMax
		getline(LegendreZeroes, line);
		size_t const lMax_ = std::stoull(line);
		
		while(getline(LegendreZeroes, line))
		{
			double const z = std::stod(line);
			
			// To compute the slope near the root, we insert an adjacent value
			Pl_computer.z.push_back(z);
			Pl_computer.z.push_back((z == 0.) ? 1e-8 : z*(1.-1e-12));
		}
		
		return lMax_;
	}();
	
	Pl_computer.Reset();
	container_t const& z = Pl_computer.z;
	
	size_t failures = 0;
	
	printf(" l   zeroes found\n");
	
	for(size_t l = 1; l <= lMax; ++l)
	{
		size_t zeroesFound = 0;
		
		assert(l == Pl_computer.l());
		container_t const& P_l = Pl_computer.P_l();
		
		for(size_t i = 0; i < Pl_computer.z.size(); i += 2)
		{
			double const slope = (P_l[i+1] - P_l[i])/(z[i+1] - z[i]);
						
			if(std::fabs(P_l[i]/slope) < limit)
				++zeroesFound;
		}
		
		if(zeroesFound < l)
		{
			++failures;
			printf("%3lu %3lu\n", l, zeroesFound);
		}
		
		Pl_computer.Next();				
	}
	
	printf("%lu moments analyzed, %lu failures\n", lMax, failures);
		
	return 0;
}
