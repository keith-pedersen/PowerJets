#include "pqRand/pqRand.hpp"
#include "SelfOuterU.hpp"
#include "helperTools.hpp"
#include <stdio.h>
#include <numeric>

int main()
{
	pqRand::engine gen;
	
	size_t constexpr tileWidth = 32;
	TiledSelfOuterU_Incremental<double, Equals, tileWidth> incrementer;
	
	std::vector<double> source;
	std::vector<double> target_untiled;
	std::vector<double> target_tiled;
	std::vector<double> target_incremental;
	std::array<double, incrementer.incrementSize> increment;
	
	size_t const numTrials = 1 << 14;
	size_t const maxSize = 256;
	
	for(size_t trial = 0; trial < numTrials; ++trial)
	{
		source.clear();
		target_incremental.clear();
		
		size_t const vecSize = size_t(gen.U_even() * maxSize) + 1;
		//printf("%u\n", vecSize);
		
		for(size_t val = 0; val < vecSize; ++val)
			source.push_back(gen.U_uneven());
		
		SelfOuterU<Equals>(source, target_untiled, 2.);
		TiledSelfOuterU<Equals, tileWidth>(source, target_tiled, 2.);
		incrementer.Setup(source, 2.);
		
		size_t lengthSet;
		
		while(bool((lengthSet = incrementer.Next(increment))))
		{
			// Emplacing the entire increment tests the zero-padding routine
			target_incremental.insert(target_incremental.end(), 
				increment.begin(), increment.begin() + increment.size());
		}
		
		bool fail = false;
		
		double const sourceSum_binary = BinaryAccumulate(source);
		double const sourceSum_std = std::accumulate(source.begin(), source.end(), 0.);
		
		// Test that the BinaryAccumulate gives nearly the same answer as std::accumulate
		if(std::fabs(RelDiff(sourceSum_binary, sourceSum_std) > 1e-15))
			fail = true;
		
		auto& vec = target_incremental;
		auto randomPos = vec.begin() + (gen() % vec.size());
		// vec.erase(randomPos, randomPos + 1); // This causes failure very quickly,
		// which validates the following test.
		double const targetSum2_untiled = BinaryAccumulate(target_untiled);
		double const targetSum2_tiled = BinaryAccumulate(target_tiled);
		double const targetSum2_incremental = BinaryAccumulate(target_incremental);
		
		// Test that sqrt(sum(outer product)) matches the sum of the source
		// This is a fairly good way to ensure that the outer product is correctly calculated
		if(std::fabs(RelDiff(sourceSum_binary, std::sqrt(targetSum2_untiled))) > 1e-15)
			fail = true;
			
		if(std::fabs(RelDiff(sourceSum_binary, std::sqrt(targetSum2_tiled))) > 1e-15)
			fail = true;
			
		if(std::fabs(RelDiff(sourceSum_binary, std::sqrt(targetSum2_incremental))) > 1e-15)
			fail = true;
		
		std::array<double, 64> test;
		for(size_t i = 0; i < test.size(); ++i)
			test[i] = gen.U_uneven();
		
		// Test the std::array version of the code	
		if(std::fabs(RelDiff(BinaryAccumulate(test), std::accumulate(test.begin(), test.end(), 0.))) > 1e-15)
			fail = true;
			
		if(fail)
		{
			printf("FAIL!\n");
			return 1;
		}
	}
	printf("ALL PASS!\n");
	
	return 0;
}
